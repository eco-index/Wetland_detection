# predict.py — aligned with train.py normalization & thresholding
import os, json, warnings
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from tqdm import tqdm

from model import TransformerUNet  # same as training

# ====== CONFIG (edit if needed) ======
INPUT_DIR   = "MF_2023_lcdb_stacks/Stacks_resampled_cogs_sorted"
OUTPUT_DIR  = "preds"
WEIGHTS     = "models_wetlands_12band/best_model.pth"
BANDS       = list(range(12))     # FEATURE_BANDS in train.py
PATCH       = 256
STRIDE      = 256                 # training used PATCH_SIZE==STRIDE; keep same
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_SIGMA  = 3.0                 # same clip used in dataset __getitem__
SKIP_IF_DONE = True
# =====================================

def list_tifs(folder):
    return sorted([str(Path(folder, f)) for f in os.listdir(folder) if f.lower().endswith(".tif")])

def load_best_threshold(weights_path, fallback=0.50):
    cand = Path(weights_path).with_name("best_threshold.json")
    if cand.exists():
        try:
            data = json.loads(cand.read_text())
            return float(data.get("best_threshold", fallback))
        except Exception:
            pass
    return fallback

def compute_global_band_stats(paths, band_idx):
    """Matches train.py: compute over ALL pixels of FEATURE_BANDS with np.nan_to_num, no masks."""
    n = len(band_idx)
    sum_b   = np.zeros(n, dtype=np.float64)
    sum_sq  = np.zeros(n, dtype=np.float64)
    count_b = np.zeros(n, dtype=np.float64)
    for p in tqdm(paths, desc="Computing band stats"):
        try:
            with rasterio.open(p) as src:
                data = src.read()  # [C,H,W]; 0-based indexing for numpy slice
                bands = np.stack([data[i] for i in band_idx]).astype(np.float64)
            bands = np.nan_to_num(bands)
            sum_b  += bands.sum(axis=(1, 2))
            sum_sq += (bands**2).sum(axis=(1, 2))
            count_b += bands.shape[1] * bands.shape[2]
        except Exception:
            continue
    mean = sum_b / np.maximum(count_b, 1)
    var  = np.maximum(0.0, (sum_sq / np.maximum(count_b, 1)) - mean**2)
    std  = np.sqrt(var) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def ensure_cog_write(arr, profile, out_path, dtype=None, nodata=None, predictor=2):
    """Write tiled/deflate GTiff + overviews; try COG repack (optional)."""
    profile = profile.copy()
    profile.update({
        "driver": "GTiff",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
        "predictor": predictor,
        "interleave": "band"
    })
    if dtype: profile["dtype"] = dtype
    if nodata is not None: profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
        else:
            dst.write(arr)
        # internal overviews
        h, w = profile["height"], profile["width"]
        levels, scale = [], 2
        while max(h//scale, w//scale) >= 256:
            levels.append(scale); scale *= 2
        if levels:
            is_int = np.issubdtype(np.dtype(profile["dtype"]), np.integer)
            dst.build_overviews(levels, Resampling.nearest if is_int else Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="nearest" if is_int else "average")

    # optional: convert to true COG if driver present
    try:
        from rasterio.shutil import copy as rio_copy
        is_int = np.issubdtype(np.dtype(profile["dtype"]), np.integer)
        tmp = str(Path(out_path).with_suffix(".cog.tmp.tif"))
        rio_copy(out_path, tmp, driver="COG",
                 forward_overviews=True, compress="DEFLATE", blocksize=256,
                 resampling="nearest" if is_int else "average",
                 overview_resampling="nearest" if is_int else "average")
        os.replace(tmp, out_path)
    except Exception:
        pass

@torch.no_grad()
def predict_tile(path, out_dir, model, device, bands, patch, stride, mu, sigma, thr):
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        prof = src.profile

        prob_acc = np.zeros((H, W), dtype=np.float32)
        hits_acc = np.zeros((H, W), dtype=np.uint16)

        # window grid (no partial windows, mirroring training coverage)
        rows = range(0, H - patch + 1, stride)
        cols = range(0, W - patch + 1, stride)
        windows = [(r, c, patch, patch) for r in rows for c in cols]

        for (r, c, h, w) in tqdm(windows, desc=f"Infer {Path(path).name}", leave=False):
            win = Window(c, r, w, h)
            block = src.read([i+1 for i in bands], window=win).astype(np.float32)  # [B,h,w]

            # clip to μ±3σ, then z-score — exactly like dataset __getitem__
            for i in range(block.shape[0]):
                lo = mu[i] - CLIP_SIGMA * sigma[i]
                hi = mu[i] + CLIP_SIGMA * sigma[i]
                np.clip(block[i], lo, hi, out=block[i])
            block = (block - mu[:, None, None]) / (sigma[:, None, None] + 1e-6)
            block = np.nan_to_num(block)

            xb = torch.from_numpy(block[None, ...]).to(device, non_blocking=True)
            logits = model(xb)                       # [1,1,h,w]
            prob = torch.sigmoid(logits)[0,0].cpu().numpy()

            prob_acc[r:r+h, c:c+w] += prob
            hits_acc[r:r+h, c:c+w] += 1

        # average probabilities on covered pixels; others remain 0
        covered = hits_acc > 0
        avg_prob = np.zeros_like(prob_acc, dtype=np.float32)
        avg_prob[covered] = prob_acc[covered] / hits_acc[covered]

        # class map (pixel-wise): 1=non-wetland, 2=wetland, 0=nodata (outside covered grid)
        cls = np.zeros((H, W), dtype=np.uint8)
        cls[covered] = 1
        cls[(avg_prob >= thr) & covered] = 2

        base = Path(path).stem
        prob_path = str(Path(out_dir, f"{base}_wetland_prob.tif"))
        cls_path  = str(Path(out_dir, f"{base}_wetland_cls.tif"))

        prof_prob = prof.copy(); prof_prob.update(count=1, dtype="float32")
        ensure_cog_write(avg_prob, prof_prob, prob_path, dtype="float32", predictor=3)

        prof_cls = prof.copy(); prof_cls.update(count=1, dtype="uint8")
        ensure_cog_write(cls, prof_cls, cls_path, dtype="uint8", nodata=0, predictor=2)

        # simple summary
        wet_pct = (float((cls==2).sum()) / float((covered).sum()))*100.0 if covered.any() else 0.0
        print(f"✓ {Path(path).name} -> {prob_path}, {cls_path} | thr={thr:.3f} | wetland={wet_pct:.2f}%")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Device: {device} | Bands: {BANDS}")

    # model
    model = TransformerUNet(in_channels=len(BANDS), num_classes=1).to(device)
    ckpt = torch.load(WEIGHTS, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # threshold (use training's best_threshold.json if present)
    thr = load_best_threshold(WEIGHTS, fallback=0.50)
    print(f"Using threshold {thr:.3f}")

    # inputs + global stats (same method as train.py)
    tifs = list_tifs(INPUT_DIR)
    print(f"Found {len(tifs)} tif(s) in {INPUT_DIR}")
    if not tifs:
        return

    mu, sigma = compute_global_band_stats(tifs, BANDS)
    print("Global μ:", np.round(mu, 4))
    print("Global σ:", np.round(sigma, 4))

    for tif in tifs:
        base = Path(tif).stem
        out_prob = Path(OUTPUT_DIR, f"{base}_wetland_prob.tif")
        out_cls  = Path(OUTPUT_DIR, f"{base}_wetland_cls.tif")
        if SKIP_IF_DONE and out_prob.exists() and out_cls.exists():
            print(f"• Skip {base} (already exists)")
            continue
        predict_tile(tif, OUTPUT_DIR, model, device, BANDS, PATCH, STRIDE, mu, sigma, thr)

if __name__ == "__main__":
    main()
