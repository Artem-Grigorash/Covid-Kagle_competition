import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import UnetPlusPlus

# === Настройки ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = 512
THRESHOLD = 0.35

# === Загружаем данные ===
images_medseg = np.load("covid_data/images_medseg.npy")
masks_medseg = np.load("covid_data/masks_medseg.npy")

print(f"📦 Data shapes: images={images_medseg.shape}, masks={masks_medseg.shape}")

# Берём только два канала (ground glass + consolidation)
masks_medseg = masks_medseg[..., :2]

# === Разделение на train/val ===
idx = np.arange(len(images_medseg))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

train_images, val_images = images_medseg[train_idx], images_medseg[val_idx]
train_masks, val_masks = masks_medseg[train_idx], masks_medseg[val_idx]

# === Аугментации ===
train_transform = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
])

val_transform = A.Compose([
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])

# === Dataset ===
class CovidSegDataset(Dataset):
    def __init__(self, imgs, masks, transform=None):
        self.imgs = imgs
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)

        img = np.clip(img, -1000, 400)
        img = (img - img.min()) / (img.max() - img.min())

        # Добавляем канал, если он потерялся (например, стал (512,512))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Убедимся, что формат [C,H,W]
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3 and img.shape[0] != 1:
            img = img.permute(2, 0, 1)
        if mask.ndim == 3 and mask.shape[0] != 2:
            mask = mask.permute(2, 0, 1)

        return img, mask


train_loader = DataLoader(CovidSegDataset(train_images, train_masks, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CovidSegDataset(val_images, val_masks, val_transform), batch_size=BATCH_SIZE, shuffle=False)

# === Модель ===
model = UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=2
).to(device)

# === Функции потерь ===
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, targets, smooth=1e-6):
        preds = torch.sigmoid(preds)
        bce = self.bce(preds, targets)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return 0.5 * bce + 0.5 * (1 - dice)

criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# === Тренировка ===
def dice_coef(preds, masks, thr=0.35):
    preds = (torch.sigmoid(preds) > thr).float()
    intersection = (preds * masks).sum((1,2,3))
    union = preds.sum((1,2,3)) + masks.sum((1,2,3))
    return (2 * intersection / (union + 1e-6)).mean().item()

best_val_dice = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_dice = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            val_dice += dice_coef(preds, masks)
    val_dice /= len(val_loader)
    scheduler.step()

    avg_loss = train_loss / len(train_loader)
    print(f"📉 Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "unetpp_covid.pth")
        print(f"💾 Model saved (best dice={val_dice:.4f})")

print("✅ Training finished")
print(f"Best Val Dice: {best_val_dice:.4f}")
