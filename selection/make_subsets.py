"""
Reviewer items 2-5: build Top-k / Random-k synthetic subsets and SSIM statistics
===============================================================================
For every original training image, the GAN produced 10 synthetic images.
This script:
  1. computes SSIM(original, generated) for ALL generated images, exactly as in
     semilarity.py (RGB -> 256x256 LANCZOS -> grayscale, data_range=255), and
     saves them to  ssim_scores_<gan>.csv
  2. creates subset folders containing ONLY synthetic images (no originals):
         <out_root>/<gan>_top3/{benign,malignant}
         <out_root>/<gan>_top5/{benign,malignant}     (sanity check vs. existing topfive)
         <out_root>/<gan>_top10/{benign,malignant}
         <out_root>/<gan>_random5/{benign,malignant}  (seeded, reproducible)
  3. writes ssim_threshold_counts_<gan>.csv: how many generated images are
     retained at SSIM >= 0.60 / 0.70 / 0.80 / 0.90  (reviewer item 5)

Use the folders with train_conventional_aug.py --extra_train_dirs ...

Usage in Spyder:
    %run make_subsets.py --gan cyclegan
    %run make_subsets.py --gan pix2pix
"""

import os
import re
import random
import shutil
import argparse

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

BASE = r"F:\jihad\Breast GAN types\breast\breast"
DEFAULTS = {
    "train_dir": os.path.join(BASE, "split_dataset", "train"),
    "gen_dir": {
        "cyclegan": os.path.join(BASE, "cyclegan_augmented", "generate data"),
        "pix2pix":  os.path.join(BASE, "pix2pix_augmented"),
    },
    "out_root": os.path.join(BASE, "revision_subsets"),
}
CLASSES = ["benign", "malignant"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
THRESHOLDS = [0.60, 0.70, 0.80, 0.90]


def load(path, size=(256, 256)):
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def generated_for(stem, files, gan):
    """Generated files belonging to one original (stem = original name without extension).
    cyclegan_enhancement_{A2B|B2A|B2A2B}_<stem>_NN.png  /  pix2pix_enhancement_<stem>_NN.png
    'orig_' and 'input_' helper files are excluded."""
    s = re.escape(stem)
    if gan == "cyclegan":
        pat = re.compile(rf"^cyclegan_[a-z_]+?_(A2B|B2A|B2A2B)_{s}_\d{{2}}\.png$")
    else:
        pat = re.compile(rf"^pix2pix_[a-z_]+?_{s}_\d{{2}}\.png$")
    return sorted(f for f in files if pat.match(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", choices=["cyclegan", "pix2pix"], required=True)
    ap.add_argument("--train_dir", default=DEFAULTS["train_dir"])
    ap.add_argument("--gen_dir", default=None, help="Folder with benign/ and malignant/ generated images")
    ap.add_argument("--out_root", default=DEFAULTS["out_root"])
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--random_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    gen_dir = args.gen_dir or DEFAULTS["gen_dir"][args.gan]
    os.makedirs(args.out_root, exist_ok=True)

    # ---------- 1. SSIM for every generated image ----------
    rows = []
    for cls in CLASSES:
        odir, gdir = os.path.join(args.train_dir, cls), os.path.join(gen_dir, cls)
        if not os.path.isdir(gdir):
            raise FileNotFoundError(f"Generated folder not found: {gdir}  (use --gen_dir)")
        gfiles = os.listdir(gdir)
        originals = sorted(f for f in os.listdir(odir) if f.lower().endswith(IMG_EXT))
        missing = 0
        for of in tqdm(originals, desc=f"{args.gan} {cls}"):
            stem = os.path.splitext(of)[0]
            gens = generated_for(stem, gfiles, args.gan)
            if not gens:
                missing += 1
                continue
            o = load(os.path.join(odir, of))
            for g in gens:
                rows.append({"class": cls, "original": of, "generated": g,
                             "path": os.path.join(gdir, g),
                             "ssim": ssim(o, load(os.path.join(gdir, g)), data_range=255)})
        print(f"{cls}: {len(originals)} originals, {missing} without generated images")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No generated images matched. Check --gen_dir and file names.")
        return
    df["rank"] = df.groupby(["class", "original"])["ssim"].rank(ascending=False, method="first").astype(int)
    df.drop(columns="path").to_csv(os.path.join(args.out_root, f"ssim_scores_{args.gan}.csv"), index=False)
    per_orig = df.groupby(["class", "original"]).size()
    print(f"\nGenerated images scored: {len(df)}  "
          f"(per original: min {per_orig.min()}, max {per_orig.max()})")

    # ---------- 2. subsets ----------
    def write_subset(name, sub):
        root = os.path.join(args.out_root, f"{args.gan}_{name}")
        for cls in CLASSES:
            d = os.path.join(root, cls)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        for cls, path, gen in sub[["class", "path", "generated"]].itertuples(index=False):
            shutil.copy2(path, os.path.join(root, cls, gen))
        counts = sub.groupby("class").size().to_dict()
        print(f"  {name:<9} -> {root}   {counts}   mean SSIM {sub['ssim'].mean():.4f}")
        return {"subset": name, **{f"n_{k}": v for k, v in counts.items()},
                "mean_ssim": round(sub["ssim"].mean(), 4)}

    summary = []
    for k in args.ks:
        summary.append(write_subset(f"top{k}", df[df["rank"] <= k]))
    rng = random.Random(args.seed)
    picked = []
    for _, idx in sorted(df.groupby(["class", "original"]).groups.items()):
        idx = sorted(idx)
        picked += rng.sample(idx, min(args.random_k, len(idx)))
    rnd = df.loc[picked]
    summary.append(write_subset(f"random{args.random_k}", rnd))
    pd.DataFrame(summary).to_csv(os.path.join(args.out_root, f"subset_summary_{args.gan}.csv"), index=False)

    # ---------- 3. SSIM threshold counts (item 5) ----------
    thr = [{"threshold": f">= {t:.2f}", "retained": int((df["ssim"] >= t).sum()),
            "total": len(df), "retained_%": round(100 * (df["ssim"] >= t).mean(), 2)}
           for t in THRESHOLDS]
    thr_df = pd.DataFrame(thr)
    thr_df.to_csv(os.path.join(args.out_root, f"ssim_threshold_counts_{args.gan}.csv"), index=False)
    print("\nSSIM threshold sensitivity:\n" + thr_df.to_string(index=False))
    print(f"\nSSIM of all generated: mean {df['ssim'].mean():.4f}, median {df['ssim'].median():.4f}, "
          f"min {df['ssim'].min():.4f}, max {df['ssim'].max():.4f}")


if __name__ == "__main__":
    main()
