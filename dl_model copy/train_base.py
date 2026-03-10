import os
import csv
import gc
import math
import json
import random
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch.amp import autocast, GradScaler
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model import TransformerUNet  # Make sure model.py is correct and in the same directory

# Config
DATA_DIR = "/data/MF_2023_lcdb_stacks/Stacks_resampled_cogs_sorted"
MODEL_SAVE_DIR = "/data/dl_model/models_wetlands_12band"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

PATCH_SIZE = 256
STRIDE = 256
BATCH_SIZE = 4
BASE_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 42

FEATURE_BANDS = list(range(12))  # First 12 bands
TARGET_BAND = 13  # Label is in band 14
CSV_LOG_PATH = os.path.join(MODEL_SAVE_DIR, "training_log.csv")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===== Utility Functions =====
def list_tifs(folder: str) -> List[str]:
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".tif")]

def compute_band_stats(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(FEATURE_BANDS)
    sum_b, sum_sq_b, count_b = np.zeros(n), np.zeros(n), np.zeros(n)
    for p in tqdm(paths, desc="Computing band stats"):
        try:
            with rasterio.open(p) as src:
                data = src.read()
                bands = np.stack([data[i] for i in FEATURE_BANDS]).astype(np.float64)
            bands = np.nan_to_num(bands)
            sum_b += bands.sum(axis=(1, 2))
            sum_sq_b += (bands**2).sum(axis=(1, 2))
            count_b += bands.shape[1] * bands.shape[2]
        except:
            continue
    mean = sum_b / np.maximum(count_b, 1)
    std = np.sqrt(np.maximum(0.0, sum_sq_b / np.maximum(count_b,1) - mean**2)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def estimate_pos_weight(paths: List[str], sample_limit: int = 200) -> torch.Tensor:
    pos = neg = 0
    for p in paths[:sample_limit]:
        try:
            with rasterio.open(p) as src:
                y_raw = src.read(TARGET_BAND + 1).astype(np.int16)
            valid = (y_raw == 1) | (y_raw == 2)
            if not valid.any(): continue
            y01 = (y_raw == 2)
            pos += int((y01 & valid).sum())
            neg += int(((~y01) & valid).sum())
        except:
            continue
    return torch.tensor([max(neg, 1) / max(pos, 1)], dtype=torch.float32, device=DEVICE)

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# --------- Custom Dataset ---------
class WetlandTileDataset(Dataset):
    def __init__(self, samples, band_means, band_stds, patch_size, stride, augment=False):
        self.samples = samples
        self.band_means = band_means
        self.band_stds = band_stds
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.aug = self._get_aug() if augment else None

    def _get_aug(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-10, 10), p=0.5),
        ], additional_targets={"masks0": "mask", "masks1": "mask"})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, r, c, h, w, _has_pos = self.samples[idx]
        try:
            with rasterio.open(path) as src:
                win = Window(c, r, w, h)
                data = src.read(window=win)

            x = np.stack([data[i] for i in FEATURE_BANDS]).astype(np.float32)

            for i in range(x.shape[0]):
                lo = self.band_means[i] - 3 * self.band_stds[i]
                hi = self.band_means[i] + 3 * self.band_stds[i]
                x[i] = np.clip(x[i], lo, hi)
            x = (x - self.band_means[:, None, None]) / (self.band_stds[:, None, None] + 1e-6)
            x = np.nan_to_num(x)

            y_raw = data[TARGET_BAND].astype(np.int16)
            valid = ((y_raw == 1) | (y_raw == 2)).astype(np.uint8)
            y01 = (y_raw == 2).astype(np.uint8)

            if self.augment:
                out = self.aug(image=x.transpose(1, 2, 0), masks=[y01, valid])
                x = out["image"].transpose(2, 0, 1).copy().astype(np.float32)
                y01, valid = [m.copy().astype(np.uint8) for m in out["masks"]]

            return (
                torch.from_numpy(x),
                torch.from_numpy(y01),
                torch.from_numpy(valid.astype(bool)),
                torch.tensor(_has_pos, dtype=torch.bool),
            )
        except Exception:
            # fallback to another random sample
            return self.__getitem__(np.random.randint(0, len(self.samples)))

# --------- Dataset Splitting ---------
def extract_valid_patches(paths, patch_size, stride):
    samples = []
    for path in tqdm(paths, desc="Extracting patches"):
        try:
            with rasterio.open(path) as src:
                H, W = src.height, src.width
                yband = src.read(TARGET_BAND + 1)
                for r in range(0, H - patch_size + 1, stride):
                    for c in range(0, W - patch_size + 1, stride):
                        patch = yband[r:r+patch_size, c:c+patch_size]
                        has_valid = ((patch == 1) | (patch == 2)).any()
                        samples.append((path, r, c, patch_size, patch_size, has_valid))
        except Exception:
            continue
    return samples

# --------- Losses ---------
class MaskedBCE(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, targets, valid_mask):
        loss = self.loss(logits.squeeze(1), targets.float())
        return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1)

def dice_loss(logits, targets, valid_mask, eps=1e-6):
    probs = torch.sigmoid(logits.squeeze(1))
    probs = probs * valid_mask
    targets = targets.float() * valid_mask
    inter = (probs * targets).sum()
    union = (probs + targets).sum()
    return 1 - (2 * inter + eps) / (union + eps)

# --------- Evaluation ---------
@torch.no_grad()
def evaluate(model, loader, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    TP = FP = FN = TN = 0

    for xb, yb, vb, _ in loader:
        xb, yb, vb = xb.to(DEVICE), yb.to(DEVICE), vb.to(DEVICE)
        logits = model(xb)
        total_loss += loss_fn(logits, yb, vb).item()

        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs > threshold).int()
        TP += ((preds == 1) & (yb == 1) & vb).sum().item()
        TN += ((preds == 0) & (yb == 0) & vb).sum().item()
        FP += ((preds == 1) & (yb == 0) & vb).sum().item()
        FN += ((preds == 0) & (yb == 1) & vb).sum().item()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    avg_loss = total_loss / len(loader)
    return avg_loss, precision, recall, f1, iou

@torch.no_grad()
def find_best_threshold(model, loader):
    best_iou = -1
    best_thr = 0.5
    best_metrics = None
    for t in np.linspace(0.3, 0.7, 21):
        val_loss, P, R, F1, IoU = evaluate(model, loader, MaskedBCE(), threshold=t)
        if IoU > best_iou:
            best_iou = IoU
            best_thr = t
            best_metrics = {"loss": val_loss, "precision": P, "recall": R, "f1": F1, "iou": IoU}
    return best_thr, best_metrics

# --------- Build Loaders ---------
def build_loaders():
    files = list_tifs(DATA_DIR)
    band_means, band_stds = compute_band_stats(files)
    samples = extract_valid_patches(files, PATCH_SIZE, STRIDE)

    samples_with_pos = [s for s in samples if s[-1]]
    samples_no_pos = [s for s in samples if not s[-1]]
    random.shuffle(samples_with_pos)

    train_pos, val_pos = train_test_split(samples_with_pos, test_size=0.1, random_state=SEED)
    val_samples = val_pos + samples_no_pos[:len(val_pos)]
    train_samples = train_pos + samples_no_pos[len(val_pos):]

    train_ds = WetlandTileDataset(train_samples, band_means, band_stds, PATCH_SIZE, STRIDE, augment=True)
    val_ds = WetlandTileDataset(val_samples, band_means, band_stds, PATCH_SIZE, STRIDE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, files


# ===== You must include these below for the script to be complete =====
# - WetlandTileDataset
# - MaskedBCE
# - dice_loss
# - evaluate
# - find_best_threshold
# - build_loaders

# ===== Training Entry Point =====
def main():
    print(f"Training on device: {DEVICE}")
    train_loader, val_loader, tr_files = build_loaders()

    model = TransformerUNet(in_channels=12, num_classes=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    pos_weight = estimate_pos_weight(tr_files)
    bce = MaskedBCE(pos_weight=pos_weight)

    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "f1", "iou", "pos_weight"])

    best_iou = -1.0
    start_epoch = 0

    if os.path.exists(BEST_MODEL_PATH):
        print(f"Resuming from best model: {BEST_MODEL_PATH}")
        ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)

    for epoch in range(start_epoch, BASE_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{BASE_EPOCHS}")

        for xb, yb, vb, _ in pbar:
            xb, yb, vb = xb.to(DEVICE), yb.to(DEVICE), vb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type):
                logits = model(xb)
                loss = bce(logits, yb, vb) + 0.5 * dice_loss(logits, yb, vb)

            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            del xb, yb, vb, logits, loss
            gc.collect()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, P, R, F1, IoU = evaluate(model, val_loader, bce)

        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"P {P:.3f} R {R:.3f} F1 {F1:.3f} IoU {IoU:.3f}")

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, os.path.join(MODEL_SAVE_DIR, f"trained_epoch_{epoch+1}.pth"))
        if IoU > best_iou:
            best_iou = IoU
            torch.save(state, BEST_MODEL_PATH)
            print(f"🌟 Best updated (IoU={IoU:.3f}) at epoch {epoch+1}")

        with open(CSV_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{P:.6f}", f"{R:.6f}", f"{F1:.6f}", f"{IoU:.6f}",
                             float(pos_weight.item())])

    best_thr, metrics = find_best_threshold(model, val_loader)
    with open(os.path.join(MODEL_SAVE_DIR, "best_threshold.json"), "w") as f:
        json.dump({"best_threshold": best_thr, "metrics": metrics}, f, indent=2)
    print(f"Best threshold on val = {best_thr:.2f} | {metrics}")

if __name__ == "__main__":
    main()
