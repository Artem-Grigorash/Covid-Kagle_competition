import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from segmentation_models_pytorch import UnetPlusPlus

# ===== конфиг =====
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"🔥 Device: {device}")

WEIGHTS_PATH = "unetpp_covid_plus.pth"
THRESH = 0.35

# ===== модель =====
model = UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=2
).to(device)

state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"✅ Weights loaded: {WEIGHTS_PATH}")

# ===== данные =====
test_images = np.load("covid_data/test_images_medseg.npy")  # (10,512,512,1)
print(f"📂 Test shape: {test_images.shape}")

def normalize_ct(img):
    img = np.clip(img, -1000, 400)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-6)

@torch.no_grad()
def predict_single(img):
    img = normalize_ct(img.astype(np.float32))
    x = torch.from_numpy(np.expand_dims(img, 0)).permute(0, 3, 1, 2).to(device)
    pred = torch.sigmoid(model(x)).cpu().numpy()[0]  # (2,512,512)
    return pred

# ===== предсказания =====
all_preds = []
for i in tqdm(range(len(test_images)), desc="Predicting"):
    pred = predict_single(test_images[i])
    # бинаризация
    pred_bin = (pred > THRESH).astype(np.uint8)
    all_preds.append(pred_bin)

all_preds = np.stack(all_preds, axis=0)  # (10,2,512,512)
print(f"✅ Predictions ready: {all_preds.shape}")

# ===== CSV формат =====
# Kaggle требует (Id,Predicted)
flat = all_preds.transpose(0, 2, 3, 1).ravel().astype(int)
ids = np.arange(len(flat))

df = pd.DataFrame({"Id": ids, "Predicted": flat})
df.to_csv("submission_final.csv", index=False)
print("💾 Saved: submission_final.csv")
print(f"🧾 Rows: {len(df)} (must be 10*512*512*2 = {10*512*512*2})")
