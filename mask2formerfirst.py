import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import Mask2FormerForUniversalSegmentation

# =====================
# Аргументы
# =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--data_dir", type=str, default="covid_data")
    p.add_argument("--out_model", type=str, default="mask2former_head_best.pth")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_submit", action="store_true")
    p.add_argument("--submission", type=str, default="submission.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# =====================
# Утилиты
# =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

def normalize_ct(x: np.ndarray):
    x = np.clip(x, -1000, 400).astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x

# =====================
# Датасет
# =====================
class CovidDataset(Dataset):
    def __init__(self, imgs, masks, tf):
        self.imgs, self.masks, self.tf = imgs, masks, tf
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        img = normalize_ct(self.imgs[i])
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        mask = self.masks[i].astype(np.float32)
        aug = self.tf(image=img, mask=mask)
        x, y = aug["image"].float(), aug["mask"].float()
        return x, y

class CovidTestDataset(Dataset):
    def __init__(self, imgs, tf):
        self.imgs, self.tf = imgs, tf
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        img = normalize_ct(self.imgs[i])
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        aug = self.tf(image=img)
        x = aug["image"].float()
        return x

# =====================
# Модель: Mask2Former + наш 1x1 head (Q→2)
# =====================
class M2FHead(nn.Module):
    """
    Берём masks_queries_logits: [B, Q, h, w]
    Прогоняем через 1x1 conv (in=Q, out=2), далее апсемплим снаружи.
    """
    def __init__(self, base_model, num_queries=100, out_channels=2):
        super().__init__()
        self.base = base_model
        self.head = nn.Conv2d(num_queries, out_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):  # x: [B,3,H,W]
        out = self.base(pixel_values=x)
        mq = out.masks_queries_logits  # [B,Q,h,w]
        if mq.ndim != 4:
            raise RuntimeError(f"Unexpected masks_queries_logits shape: {tuple(mq.shape)}")
        logits_small = self.head(mq)    # [B,2,h,w]
        return logits_small

# =====================
# Лоссы/метрики
# =====================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets, smooth=1e-6):
        # logits/targets: [B,2,H,W]
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p = probs.reshape(probs.size(0), -1)
        t = targets.reshape(targets.size(0), -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)
        return 0.5 * bce + 0.5 * (1 - dice.mean())

def dice_coef(logits, targets, thr=0.35):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).reshape(preds.size(0), -1).sum(1)
    union = preds.reshape(preds.size(0), -1).sum(1) + targets.reshape(targets.size(0), -1).sum(1)
    return (2 * inter / (union + 1e-6)).mean().item()

# =====================
# Тренировка
# =====================
def train_mask2former(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    set_seed(args.seed)

    # Данные
    images = np.load(os.path.join(args.data_dir, "images_medseg.npy"))           # (100,512,512,1)
    masks  = np.load(os.path.join(args.data_dir, "masks_medseg.npy"))[..., :2]   # (100,512,512,2)
    print(f"📦 Data shapes: images={images.shape}, masks={masks.shape}")

    idx = np.arange(len(images))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]
    train_images, val_images = images[train_idx], images[val_idx]
    train_masks,  val_masks  = masks[train_idx],  masks[val_idx]

    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ToTensorV2()
    ])
    val_tf = A.Compose([ToTensorV2()])

    train_loader = DataLoader(CovidDataset(train_images, train_masks, train_tf),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(CovidDataset(val_images,  val_masks,  val_tf),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Модель
    print("⚙️ Loading Mask2Former (facebook/mask2former-swin-base-coco-panoptic)...")
    base = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic", ignore_mismatched_sizes=True
    ).to(device)

    # Узнаём количество query
    with torch.no_grad():
        try:
            Q = base.model.decoder.num_queries
        except Exception:
            Q = 100  # дефолт
    model = M2FHead(base, num_queries=Q, out_channels=2).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            x = x.repeat(1, 3, 1, 1)  # 1->3 каналов

            # forward
            logits_small = model(x)  # [B,2,h,w]
            y_chw = y.permute(0, 3, 1, 2).contiguous()      # [B,2,H,W]
            target_hw = y_chw.shape[-2:]

            logits = torch.nn.functional.interpolate(
                logits_small, size=target_hw, mode="bilinear", align_corners=False
            )  # [B,2,H,W]

            loss = criterion(logits, y_chw)
            optimizer.zero_grad(set_to_none=True)
            try:
                loss.backward()
            except RuntimeError:
                # MPS иногда кукожит strides — лечим
                loss = loss.contiguous().clone().detach().requires_grad_(True)
                loss.backward()
            optimizer.step()
            run_loss += loss.item()

        # Валидация
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = x.repeat(1, 3, 1, 1)
                logits_small = model(x)
                y_chw = y.permute(0, 3, 1, 2).contiguous()
                target_hw = y_chw.shape[-2:]

                logits = torch.nn.functional.interpolate(
                    logits_small, size=target_hw, mode="bilinear", align_corners=False
                )
                val_dice += dice_coef(logits, y_chw, thr=args.thr)
        val_dice /= len(val_loader)

        print(f"📉 Epoch {epoch}: loss={run_loss/len(train_loader):.4f} | val_dice={val_dice:.4f}")
        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict(), args.out_model)
            print(f"💾 Saved best! dice={best_val:.4f}")

    print("✅ Training done. 🏁 Best Val Dice:", best_val)

# =====================
# Инференс + сабмишн
# =====================
def predict_and_submit(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    set_seed(args.seed)

    # Тест
    test_images = np.load(os.path.join(args.data_dir, "test_images_medseg.npy"))  # (10,512,512,1)
    print(f"🧪 Test shape: {test_images.shape}")

    # Модель
    base = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic", ignore_mismatched_sizes=True
    ).to(device)
    with torch.no_grad():
        try:
            Q = base.model.decoder.num_queries
        except Exception:
            Q = 100
    model = M2FHead(base, num_queries=Q, out_channels=2).to(device)
    if os.path.exists(args.out_model):
        model.load_state_dict(torch.load(args.out_model, map_location=device))
        print(f"🔁 Loaded weights: {args.out_model}")
    model.eval()

    test_tf = A.Compose([ToTensorV2()])
    test_ds = CovidTestDataset(test_images, test_tf)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    preds_all = []
    with torch.no_grad():
        for x in tqdm(test_dl, desc="Predicting"):
            x = x.to(device)              # [1,1,512,512]
            x = x.repeat(1, 3, 1, 1)      # -> [1,3,512,512]
            logits_small = model(x)       # [1,2,h,w]
            logits = torch.nn.functional.interpolate(
                logits_small, size=(test_images.shape[1], test_images.shape[2]),
                mode="bilinear", align_corners=False
            )  # [1,2,512,512]
            probs = torch.sigmoid(logits)
            pred = (probs > args.thr).long()  # [1,2,512,512], 0/1
            preds_all.append(pred.squeeze(0).cpu().numpy())  # [2,512,512]

    # Собираем сабмишн
    # Kaggle ждёт CSV: колонки Id,Predicted, всего 10*2*512*512 строк = 5_242_880
    preds_np = np.stack(preds_all, axis=0)  # [10,2,512,512]
    flat = preds_np.reshape(-1).astype(np.int64)
    ids = np.arange(flat.size, dtype=np.int64)

    df = pd.DataFrame({"Id": ids, "Predicted": flat})
    df.set_index("Id").to_csv(args.submission)
    print(f"📤 Saved submission: {args.submission}  ({len(df):,} rows)")

# =====================
# Main
# =====================
if __name__ == "__main__":
    args = parse_args()
    # девайс лог
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {dev}")
    print(f"⚙️ Config: epochs={args.epochs} bs={args.batch_size} lr={args.lr} thr={args.thr}")

    if args.do_train:
        train_mask2former(args)

    if args.do_submit:
        predict_and_submit(args)
