"""
Reviewer item 5 - SSIM threshold sensitivity for ALL GANs (descriptive, no CNN training)
=====================================================================================
For every generated image this script computes its similarity to the real training
images of the same class, then counts how many generated images would be retained
at SSIM >= 0.60 / 0.70 / 0.80 / 0.90.

Two SSIM definitions are reported:
  * NN-SSIM (main, used for every GAN): the highest SSIM between the generated image
    and ANY real training image of its class ("nearest real neighbour").
    This is the only fair definition for unconditional GANs (WGAN, DCGAN, C-GAN),
    whose outputs are drawn from noise and are not tied to a specific original.
  * Paired SSIM (Pix2Pix / CycleGAN only): SSIM to the original the image was
    translated from - the score used for the Top-5 selection.

SSIM settings match semilarity.py: RGB -> 256x256 LANCZOS -> grayscale, 7x7 window,
data_range = 255 (identical to skimage.metrics.structural_similarity defaults).

Outputs (in ssim_results/):
    ssim_scores_<gan>.csv          one row per generated image
    ssim_threshold_table.csv       the reviewer's table (all GANs side by side)

Usage in Spyder:
    %run ssim_threshold_all.py
    %run ssim_threshold_all.py --gans wgan dcgan
    %run ssim_threshold_all.py --dirs cgan="F:/.../cgan_output/augmented_dataset"
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import uniform_filter
from tqdm import tqdm

BASE = r"F:\jihad\Breast GAN types\breast\breast"
DEFAULT_DIRS = {
    "pix2pix":  os.path.join(BASE, "pix2pix_augmented", "generate_images"),
    "cyclegan": os.path.join(BASE, "cyclegan_augmented", "generate data"),
    "wgan":     os.path.join(BASE, "wgan_output", "augmented_dataset"),
    "dcgan":    os.path.join(BASE, "dcgan_organized_dataset"),
    "cgan":     None,   # give with --dirs cgan="..."
}
TRAIN_DIR = os.path.join(BASE, "split_dataset", "train")
CLASSES = ["benign", "malignant"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
SKIP_PREFIX = ("orig_", "original_", "input_")
THRESHOLDS = [0.60, 0.70, 0.80, 0.90]
SIZE = 256
WIN = 7
C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
COV_NORM = WIN * WIN / (WIN * WIN - 1)     # skimage use_sample_covariance=True
PAD = (WIN - 1) // 2


def load_gray(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    rgb = np.asarray(img, dtype=np.uint8)
    # same conversion as cv2.COLOR_RGB2GRAY (rounded to uint8) used in semilarity.py
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.round(g), 0, 255).astype(np.float64)


def filt(a):
    """7x7 mean filter over the last two axes (reflect border, like skimage)."""
    size = (1,) * (a.ndim - 2) + (WIN, WIN)
    return uniform_filter(a, size=size, mode="reflect")


class RealBank:
    """Pre-computed local statistics of all real images of one class."""
    def __init__(self, paths):
        self.names = [os.path.basename(p) for p in paths]
        self.x = np.stack([load_gray(p) for p in paths])          # (N, H, W)
        self.mu = filt(self.x)
        self.var = (filt(self.x * self.x) - self.mu ** 2) * COV_NORM

    def ssim_all(self, y):
        """SSIM between y (H, W) and every real image -> (N,)"""
        mu_y = filt(y)
        var_y = (filt(y * y) - mu_y ** 2) * COV_NORM
        cov = (filt(self.x * y[None]) - self.mu * mu_y[None]) * COV_NORM
        s = ((2 * self.mu * mu_y + C1) * (2 * cov + C2)) / \
            ((self.mu ** 2 + mu_y ** 2 + C1) * (self.var + var_y + C2))
        return s[:, PAD:-PAD, PAD:-PAD].mean(axis=(1, 2))


def generated_files(class_dir):
    out = []
    for root, _, files in os.walk(class_dir):
        for f in files:
            if f.lower().endswith(IMG_EXT) and not f.lower().startswith(SKIP_PREFIX):
                out.append(os.path.join(root, f))
    return sorted(out)


def source_original(fname, real_names):
    """For Pix2Pix/CycleGAN names like '..._benign (12)_03.png' return 'benign (12).png'."""
    m = re.search(r"((?:benign|malignant) \(\d+\))_\d{2}\.png$", fname)
    if m:
        for ext in (".png", ".jpg", ".jpeg"):
            if m.group(1) + ext in real_names:
                return m.group(1) + ext
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gans", nargs="+", default=["pix2pix", "cyclegan", "wgan", "dcgan", "cgan"])
    ap.add_argument("--dirs", nargs="*", default=[], help='override folders: name="path"')
    ap.add_argument("--train_dir", default=TRAIN_DIR)
    ap.add_argument("--out_dir", default="ssim_results")
    a = ap.parse_args()
    dirs = dict(DEFAULT_DIRS)
    for d in a.dirs:
        k, v = d.split("=", 1)
        dirs[k.strip().lower()] = v.strip().strip('"')
    os.makedirs(a.out_dir, exist_ok=True)

    banks = {}
    for cls in CLASSES:
        cdir = os.path.join(a.train_dir, cls)
        paths = sorted(os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(IMG_EXT))
        print(f"Loading {len(paths)} real {cls} images")
        banks[cls] = RealBank(paths)

    table = {}
    for gan in a.gans:
        gdir = dirs.get(gan)
        if not gdir or not os.path.isdir(gdir):
            print(f"\n!! {gan}: folder not found ({gdir}) - skipped. Use --dirs {gan}=\"...\"")
            continue
        rows = []
        for cls in CLASSES:
            files = generated_files(os.path.join(gdir, cls))
            bank = banks[cls]
            name_idx = {n: i for i, n in enumerate(bank.names)}
            for f in tqdm(files, desc=f"{gan} {cls}"):
                s = bank.ssim_all(load_gray(f))
                src = (source_original(os.path.basename(f), name_idx)
                       if gan in ("pix2pix", "cyclegan") else None)   # unconditional GANs: no source image
                rows.append({"class": cls, "generated": os.path.relpath(f, gdir),
                             "nn_ssim": float(s.max()),
                             "nearest_real": bank.names[int(s.argmax())],
                             "paired_ssim": float(s[name_idx[src]]) if src else np.nan})
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"!! {gan}: no generated images found under {gdir}")
            continue
        df.to_csv(os.path.join(a.out_dir, f"ssim_scores_{gan}.csv"), index=False)
        n = len(df)
        col = {"N generated": n, "mean NN-SSIM": f"{df.nn_ssim.mean():.3f}"}
        for t in THRESHOLDS:
            k = int((df.nn_ssim >= t).sum())
            col[f">= {t:.2f}"] = f"{k} ({100 * k / n:.1f}%)"
        if df.paired_ssim.notna().any():
            col["mean paired SSIM"] = f"{df.paired_ssim.mean():.3f}"
            for t in THRESHOLDS:
                k = int((df.paired_ssim >= t).sum())
                col[f"paired >= {t:.2f}"] = f"{k} ({100 * k / n:.1f}%)"
        table[gan] = col
        print(f"{gan}: {n} images, mean NN-SSIM {df.nn_ssim.mean():.3f}")

    if table:
        t = pd.DataFrame(table)
        t.to_csv(os.path.join(a.out_dir, "ssim_threshold_table.csv"))
        print("\n=== SSIM threshold sensitivity (retained generated images) ===")
        print(t.fillna("-").to_string())


if __name__ == "__main__":
    main()