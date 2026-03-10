
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from model import WetlandUNet
from dataset import WetlandTileDataset
from utils import compute_band_stats, evaluate, compute_class_weights
from tqdm import tqdm
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256
STRIDE = 128
NUM_CLASSES = 4  # Adjust based on your task
BATCH_SIZE = 16
LOG_PATH = "training_log_extended.csv"
BEST_MODEL_PATH = "best_model.pth"

# Load data and compute stats
all_files = sorted(Path("/data/tiles").rglob("*.tif"))
band_means, band_stds = compute_band_stats(all_files)

# Split data
val_split = 0.2
np.random.seed(42)
shuffled = np.random.permutation(all_files)
split_idx = int(len(all_files) * (1 - val_split))
tr_files = shuffled[:split_idx]
val_files = shuffled[split_idx:]

train_ds = WetlandTileDataset(tr_files, band_means, band_stds, PATCH_SIZE, STRIDE, augment=True)
val_ds = WetlandTileDataset(val_files, band_means, band_stds, PATCH_SIZE, STRIDE, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model
model = WetlandUNet(in_channels=12, num_classes=NUM_CLASSES).to(DEVICE)

# Resume best model
best_iou = 0
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print("✅ Loaded best model for continued training.")

# Training components
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Load previous training log if exists
if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    start_epoch = log_df["epoch"].max() + 1
    best_iou = log_df["iou"].max()
else:
    log_df = pd.DataFrame()
    start_epoch = 1

# Training loop
for epoch in range(start_epoch, 801):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/800")
    for xb, yb, vb, _ in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with autocast(enabled=(DEVICE.type == "cuda")):
            preds = model(xb)
            loss = criterion(preds, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(pbar)
    val_loss, metrics = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | P {metrics['precision']:.3f} R {metrics['recall']:.3f} F1 {metrics['f1']:.3f} IoU {metrics['iou']:.3f}")

    # Save best model
    if metrics["iou"] > best_iou:
        best_iou = metrics["iou"]
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"🌟 Best updated (IoU={best_iou:.3f}) at epoch {epoch}")

    # Append log
    log_df = pd.concat([log_df, pd.DataFrame([{
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        **metrics
    }])])
    log_df.to_csv(LOG_PATH, index=False)
