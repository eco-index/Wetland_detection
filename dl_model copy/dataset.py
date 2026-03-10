
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from rasterio.windows import Window
import albumentations as A

FEATURE_BANDS = list(range(12))  # Bands 0 to 11 as input
TARGET_BAND = 13  # Multiclass labels assumed in band 13

class WetlandTileDataset(Dataset):
    def __init__(self, samples, band_means, band_stds, patch_size, stride, augment=False):
        self.samples = samples
        self.band_means = np.array(band_means)
        self.band_stds = np.array(band_stds)
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.aug = self._get_aug() if augment else None

    def __len__(self):
        return len(self.samples)

    def _get_aug(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
        ])

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

            y = data[TARGET_BAND].astype(np.int64)

            if self.augment:
                out = self.aug(image=x.transpose(1, 2, 0), mask=y)
                x = out["image"].transpose(2, 0, 1).copy().astype(np.float32)
                y = out["mask"].copy().astype(np.int64)

            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.tensor(bool(_has_pos)),
            )
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
