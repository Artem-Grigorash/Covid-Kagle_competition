import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segmentation_models_pytorch import UnetPlusPlus

# ========= конфиг =========
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"🔥 Using device: {device}")

EPOCHS = 30
BATCH_SIZE = 6
LR = 1e-5
WEIGHT_DECAY = 1e-5
THRESH = 0.35

# ========= дата =========
def load_data():
    medseg_img = np.load("covid_data/images_medseg.npy")         # (100, 512,512,1)
    medseg_msk = np.load("covid_data/masks_medseg.npy")[..., :2] # два класса

    radio_img = np.load("covid_data/images_radiopedia.npy")      # (829, 512,512,1)
    radio_msk = np.load("covid_data/masks_radiopedia.npy")[..., :2]
    # берём только позитивные срезы из radiopedia
    pos = radio_msk.sum(axis=(1,2,3)) > 0
    radio_img, radio_msk = radio_img[pos], radio_msk[pos]

    imgs = np.concatenate([medseg_img, radio_img], axis=0)
    msks = np.concatenate([medseg_msk, radio_msk], axis=0)
    return imgs, msks

class SafeTorchAugDataset(Dataset):
    """Без OpenCV: только torch.flip и rot90. Форматы: X-> [1,512,512], y-> [2,512,512]."""
    def __init__(self, imgs, msks, train=True):
        self.imgs = imgs
        self.msks = msks
        self.train = train

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32)   # (512,512,1)
        msk = self.msks[i].astype(np.float32)   # (512,512,2)

        # нормализация CT
        img = np.clip(img, -1000, 400)
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-6)

        # в тензоры
        x = torch.from_numpy(img).permute(2,0,1)      # [1,H,W]
        y = torch.from_numpy(msk).permute(2,0,1)      # [2,H,W]

        if self.train:
            # горизонтальный флип
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            # вертикальный флип
            if torch.rand(1) < 0.3:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            # rot90 k times
            k = torch.randint(0, 4, (1,)).item()
            if k:
                x = torch.rot90(x, k, dims=[1,2])
                y = torch.rot90(y, k, dims=[1,2])

        return x.float(), y.float()

# ========= модель/лоссы =========
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, target, smooth=1e-6):
        probs = torch.sigmoid(logits)
        bce = self.bce(logits, target)
        inter = (probs * target).sum(dim=(1,2,3))
        denom = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2*inter + smooth) / (denom + smooth)
        return 0.5*bce + 0.5*(1 - dice.mean())

@torch.no_grad()
def dice_score(logits, target, thr=0.35):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * target).sum(dim=(1,2,3))
    denom = preds.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return (2*inter / (denom + 1e-6)).mean().item()

# ========= обучение =========
def main():
    imgs, msks = load_data()
    # простое разбиение 85/15
    idx = np.arange(len(imgs))
    np.random.shuffle(idx)
    split = int(0.85 * len(idx))
    tr, va = idx[:split], idx[split:]
    train_ds = SafeTorchAugDataset(imgs[tr], msks[tr], train=True)
    val_ds   = SafeTorchAugDataset(imgs[va], msks[va], train=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=2
    ).to(device)

    # грузим лучшие веса
    state = torch.load("unetpp_covid.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    print("✅ Loaded weights: unetpp_covid.pth")

    # можно заморозить энкодер на первые 5 эпох, затем разморозить
    for p in model.encoder.parameters():
        p.requires_grad = False

    crit = DiceBCELoss()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best = -1.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0.0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total += loss.item()

        # разморозить энкодер после 5 эпох
        if epoch == 6:
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS-epoch+1)
            print("🔓 Unfroze encoder")

        model.eval()
        val_d, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_d += dice_score(out, yb, thr=THRESH)
                val_n += 1
        val_d /= max(1, val_n)
        sch.step()

        print(f"📉 Epoch {epoch}/{EPOCHS} | Train Loss: {total/len(train_dl):.4f} | Val Dice: {val_d:.4f}")

        if val_d > best:
            best = val_d
            torch.save(model.state_dict(), "unetpp_covid_plus.pth")
            print(f"💾 Saved best (Dice={best:.4f})")

    print(f"🏁 Done. Best Val Dice: {best:.4f}")

if __name__ == "__main__":
    main()
