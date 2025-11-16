# mask2former_train_continue.py
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import Mask2FormerForUniversalSegmentation

# -----------------------
# Аргументы
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)             # сколько ДОУЧИТЬ
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--data_dir", type=str, default="covid_data")
    p.add_argument("--weights_in", type=str, default="mask2former_head_best.pth")  # откуда грузим
    p.add_argument("--weights_out", type=str, default="mask2former_head_best.pth") # куда сохраняем лучший
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_submit", action="store_true")
    p.add_argument("--submission", type=str, default="submission.csv")
    return p.parse_args()

# -----------------------
# Утилиты
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed)
        except Exception: pass

def normalize_ct(x: np.ndarray):
    x = np.clip(x, -1000, 400).astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x

# -----------------------
# Датасеты
# -----------------------
class CovidDataset(Dataset):
    def __init__(self, imgs, masks, tf):
        self.imgs, self.masks, self.tf = imgs, masks, tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = normalize_ct(self.imgs[i])
        if img.ndim == 2: img = np.expand_dims(img, -1)
        mask = self.masks[i].astype(np.float32)  # (H,W,2)
        out = self.tf(image=img, mask=mask)
        x = out["image"].float()    # (1,H,W)
        y = out["mask"].float()     # (H,W,2)
        return x, y

class CovidTestDataset(Dataset):
    def __init__(self, imgs, tf):
        self.imgs, self.tf = imgs, tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = normalize_ct(self.imgs[i])
        if img.ndim == 2: img = np.expand_dims(img, -1)
        x = self.tf(image=img)["image"].float()
        return x

# -----------------------
# Модель: Mask2Former base + 1x1 conv head (Q -> 2)
# -----------------------
class M2FHead(nn.Module):
    def __init__(self, base_model, num_queries=100, out_channels=2):
        super().__init__()
        self.base = base_model
        self.head = nn.Conv2d(num_queries, out_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):  # x: [B,3,H,W]
        out = self.base(pixel_values=x)
        mq = out.masks_queries_logits  # [B,Q,h,w]
        return self.head(mq)           # [B,2,h,w]

# -----------------------
# Лоссы / метрики
# -----------------------
class DiceBCELoss(nn.Module):
    def __init__(self): super().__init__(); self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets, smooth=1e-6):  # [B,2,h,w] vs [B,2,h,w]
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p = probs.reshape(probs.size(0), -1)
        t = targets.reshape(targets.size(0), -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)
        return 0.5 * bce + 0.5 * (1 - dice.mean())

@torch.no_grad()
def dice_coef(logits, targets, thr=0.35):  # [B,2,H,W]
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).reshape(preds.size(0), -1).sum(1)
    union = preds.reshape(preds.size(0), -1).sum(1) + targets.reshape(targets.size(0), -1).sum(1)
    return (2 * inter / (union + 1e-6)).mean().item()

# -----------------------
# Тренировка (как раньше, только подгружаем веса в начале)
# -----------------------
def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    set_seed(args.seed)

    # Данные
    images = np.load(os.path.join(args.data_dir, "images_medseg.npy"))            # (100,512,512,1)
    masks  = np.load(os.path.join(args.data_dir, "masks_medseg.npy"))[..., :2]    # (100,512,512,2)
    print(f"📦 Data shapes: images={images.shape}, masks={masks.shape}")

    idx = np.arange(len(images)); np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_images, va_images = images[tr_idx], images[va_idx]
    tr_masks,  va_masks  = masks[tr_idx],  masks[va_idx]

    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(0.05, 0.05, 15, p=0.5),
        ToTensorV2()
    ])
    val_tf = A.Compose([ToTensorV2()])

    train_loader = DataLoader(CovidDataset(tr_images, tr_masks, train_tf),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(CovidDataset(va_images, va_masks, val_tf),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Модель
    print("⚙️ Loading Mask2Former (facebook/mask2former-swin-base-coco-panoptic)...")
    base = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic", ignore_mismatched_sizes=True
    ).to(device)
    with torch.no_grad():
        try: Q = base.model.decoder.num_queries
        except Exception: Q = 100
    model = M2FHead(base, num_queries=Q, out_channels=2).to(device)

    # >>> ВОТ ЗДЕСЬ ГРУЗИМ СТАРЫЕ ВЕСА <<<
    if os.path.exists(args.weights_in):
        state = torch.load(args.weights_in, map_location=device)
        # поддержка как state_dict модели, так и сохранённого целиком M2FHead
        if isinstance(state, dict) and "base.base_model.encoder.embeddings.patch_embeddings.projection.weight" in state:
            # кто-то сохранил весь .state_dict()
            model.load_state_dict(state, strict=False)
        else:
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                # если сохранялся именно .state_dict(), всё равно загрузим что совпало
                model.load_state_dict(state, strict=False)
        print(f"🔁 Loaded weights from: {args.weights_in}")
    else:
        print(f"⚠️ No weights found at {args.weights_in}, training from scratch.")

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)              # x: [B,1,H,W], y: [B,H,W,2]
            x = x.repeat(1, 3, 1, 1)                       # -> [B,3,H,W]
            y_chw = y.permute(0, 3, 1, 2).contiguous()     # -> [B,2,H,W]

            # как раньше: считаем лосс на апсемпленных логитах, но это работало у тебя
            logits_small = model(x)                        # [B,2,h,w]
            logits = torch.nn.functional.interpolate(
                logits_small, size=y_chw.shape[-2:], mode="bilinear", align_corners=False
            )  # [B,2,H,W]

            loss = criterion(logits, y_chw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()

        # валидация
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = x.repeat(1, 3, 1, 1)
                y_chw = y.permute(0, 3, 1, 2).contiguous()
                logits_small = model(x)
                logits = torch.nn.functional.interpolate(
                    logits_small, size=y_chw.shape[-2:], mode="bilinear", align_corners=False
                )
                val_dice += dice_coef(logits, y_chw, thr=args.thr)
        val_dice /= max(1, len(val_loader))

        print(f"📉 Epoch {epoch}: loss={total/len(train_loader):.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict(), args.weights_out)
            print(f"💾 Saved best! dice={best_val:.4f} -> {args.weights_out}")

    print(f"✅ Done. Best Val Dice: {best_val:.6f}")

# -----------------------
# Сабмит (тот же head)
# -----------------------
@torch.no_grad()
def submit(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    test = np.load(os.path.join(args.data_dir, "test_images_medseg.npy"))  # (10,512,512,1)
    tf = A.Compose([ToTensorV2()])
    dl = DataLoader(CovidTestDataset(test, tf), batch_size=1, shuffle=False)

    base = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic", ignore_mismatched_sizes=True
    ).to(device)
    try: Q = base.model.decoder.num_queries
    except Exception: Q = 100
    model = M2FHead(base, num_queries=Q, out_channels=2).to(device)

    if os.path.exists(args.weights_out):
        model.load_state_dict(torch.load(args.weights_out, map_location=device), strict=False)
        print(f"🔁 Loaded: {args.weights_out}")
    elif os.path.exists(args.weights_in):
        model.load_state_dict(torch.load(args.weights_in, map_location=device), strict=False)
        print(f"🔁 Loaded: {args.weights_in}")
    else:
        print("⚠️ Нет весов, будет сабмит с предобученных фичей головы. Не рекомендую.")

    model.eval()
    preds = []
    for x in tqdm(dl, desc="Predict"):
        x = x.to(device).repeat(1, 3, 1, 1)
        logits_small = model(x)
        logits = torch.nn.functional.interpolate(
            logits_small, size=test.shape[1:3], mode="bilinear", align_corners=False
        )
        probs = torch.sigmoid(logits)
        pred = (probs > args.thr).long().squeeze(0).cpu().numpy()  # [2,512,512]
        preds.append(pred)

    preds = np.stack(preds, 0)  # [10,2,512,512]
    flat = preds.reshape(-1).astype(np.int64)
    idxs = np.arange(flat.size, dtype=np.int64)
    pd.DataFrame({"Id": idxs, "Predicted": flat}).set_index("Id").to_csv(args.submission)
    print(f"📤 Saved submission: {args.submission} ({flat.size:,} rows)")

# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {dev}")
    print(f"⚙️ Config: epochs={args.epochs} bs={args.batch_size} lr={args.lr} thr={args.thr}")

    if args.do_train:
        train(args)
    if args.do_submit:
        submit(args)
