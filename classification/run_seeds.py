"""
Multi-seed robustness check - ONE command runs everything
=========================================================
Conditions (all with the same CNN protocol as train_conventional_aug.py):
    Orig          original training images only
    Top5Cycle     original + Top-5 CycleGAN images
    Top5Pix2Pix   original + Top-5 Pix2Pix images
for every model x seed x augmentation setting.

Default: 3 models x 3 conditions x 3 seeds x aug 'original' (= untitled0.py) = 27 runs.

* Resumable: a run whose prediction files already exist is skipped, so you can
  stop Spyder at any time and simply run the same command again.
* At the end (or with --summary_only) it writes, in results_seeds/:
    seeds_all_runs.csv     one row per run
    seeds_mean_sd.csv      mean +/- SD over seeds for every metric
    seeds_gain_vs_orig.csv Top-5 minus Orig per seed, mean +/- SD, and in how
                           many seeds the GAN condition was better

Usage in Spyder (same folder as train_conventional_aug.py):
    %run run_seeds.py
    %run run_seeds.py --models densenet121 resnet50          # fewer models
    %run run_seeds.py --augs original conservative           # both aug settings
    %run run_seeds.py --summary_only                         # just rebuild tables
"""

import os
import sys
import re
import glob
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_conventional_aug as T  # noqa: E402

BASE = r"F:\jihad\Breast GAN types\breast\breast"
GAN_DIRS = {
    "Orig": [],
    "Top5Cycle": [os.path.join(BASE, "cyclegan_top5_generated", "topfive")],
    "Top5Pix2Pix": [os.path.join(BASE, "pix2pix_top5_generated", "topfive")],
}
METRICS = ["ACC", "Balanced_ACC", "F1_macro", "F1_malignant", "TPR", "TNR", "ROC_AUC", "PR_AUC"]


def run_name(cond, aug, seed):
    return f"{cond}_{aug}_s{seed}"


def done(out_dir, model, name):
    return all(os.path.exists(os.path.join(out_dir, f"pred_{model}_{name}_{s}.csv"))
               for s in ("Internal", "External"))


def summarise(out_dir):
    rows = []
    for f in glob.glob(os.path.join(out_dir, "pred_*_s*_*.csv")):
        base = os.path.basename(f)[5:-4]                     # strip 'pred_' and '.csv'
        split = base.rsplit("_", 1)[1]
        rest = base.rsplit("_", 1)[0]
        if not re.search(r"_s\d+$", rest):                   # only multi-seed runs
            continue
        model = next(m for m in ["densenet121", "efficientnet_v2_l", "resnet50"]
                     if rest.startswith(m + "_"))
        cond, aug, seed = rest[len(model) + 1:].rsplit("_", 2)
        df = pd.read_csv(f)
        m = T.compute_metrics(df["true_label"].to_numpy(), df["prob_malignant"].to_numpy())
        rows.append({"Model": model, "Condition": cond, "Aug": aug, "Seed": int(seed[1:]),
                     "Split": split, **{k: m[k] * 100 for k in METRICS}})
    if not rows:
        print("No finished runs yet.")
        return
    runs = pd.DataFrame(rows).sort_values(["Split", "Aug", "Model", "Condition", "Seed"])
    runs.round(2).to_csv(os.path.join(out_dir, "seeds_all_runs.csv"), index=False)

    keys = ["Split", "Aug", "Model", "Condition"]
    g = runs.groupby(keys)
    ms = g[METRICS].mean().round(2)
    sd = g[METRICS].std(ddof=1).round(2)
    n = g.size().rename("n_seeds")
    table = pd.concat([n, *[ms[k].astype(str) + " ± " + sd[k].fillna(0).astype(str)
                            for k in METRICS]], axis=1)
    table.columns = ["n_seeds"] + METRICS
    table.reset_index().to_csv(os.path.join(out_dir, "seeds_mean_sd.csv"), index=False)

    # paired gain vs Orig, seed by seed
    gains = []
    for (split, aug, model), sub in runs.groupby(["Split", "Aug", "Model"]):
        orig = sub[sub.Condition == "Orig"].set_index("Seed")
        for cond in ["Top5Cycle", "Top5Pix2Pix"]:
            gan = sub[sub.Condition == cond].set_index("Seed")
            seeds = orig.index.intersection(gan.index)
            if len(seeds) == 0:
                continue
            row = {"Split": split, "Aug": aug, "Model": model, "Condition": cond,
                   "n_seeds": len(seeds)}
            for k in ["ACC", "Balanced_ACC", "TPR", "ROC_AUC"]:
                d = gan.loc[seeds, k] - orig.loc[seeds, k]
                row[f"gain_{k}"] = f"{d.mean():+.2f} ± {d.std(ddof=1) if len(d) > 1 else 0:.2f}"
                row[f"better_{k}"] = f"{int((d > 0).sum())}/{len(d)}"
            gains.append(row)
    gdf = pd.DataFrame(gains)
    gdf.to_csv(os.path.join(out_dir, "seeds_gain_vs_orig.csv"), index=False)

    ext = table.reset_index()
    ext = ext[ext.Split == "External"][["Aug", "Model", "Condition", "n_seeds",
                                        "ACC", "Balanced_ACC", "TPR", "ROC_AUC"]]
    print("\n=== External: mean ± SD over seeds ===")
    print(ext.to_string(index=False))
    if not gdf.empty:
        print("\n=== External: Top-5 minus Orig (per seed) ===")
        cols = ["Aug", "Model", "Condition", "n_seeds", "gain_Balanced_ACC",
                "better_Balanced_ACC", "gain_ROC_AUC", "better_ROC_AUC"]
        print(gdf[gdf.Split == "External"][cols].to_string(index=False))
    print(f"\nTables saved in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["densenet121", "efficientnet_v2_l", "resnet50"])
    ap.add_argument("--conditions", nargs="+", default=list(GAN_DIRS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--augs", nargs="+", default=["original"],
                    choices=["original", "conservative", "none"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output_dir", default="results_seeds")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    if not a.summary_only:
        for d in sum(GAN_DIRS.values(), []):
            if not os.path.isdir(d):
                raise FileNotFoundError(d)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jobs = [(s, aug, m, c) for s in a.seeds for aug in a.augs
                for m in a.models for c in a.conditions]
        todo = [j for j in jobs if not done(a.output_dir, j[2], run_name(j[3], j[1], j[0]))]
        print(f"Device: {device} | {len(jobs)} runs in total, {len(jobs) - len(todo)} already done, "
              f"{len(todo)} to run")
        for i, (seed, aug, model, cond) in enumerate(todo, 1):
            name = run_name(cond, aug, seed)
            print(f"\n##### Run {i}/{len(todo)}: {model} | {name} #####")
            args = SimpleNamespace(
                seed=seed, aug=aug, no_hflip=False, img_size=224, lr=1e-3,
                epochs=a.epochs, batch_size=a.batch_size, num_workers=0, selection="last",
                train_dir=T.CONFIG["train_dir"], extra_train_dirs=GAN_DIRS[cond])
            try:
                T.run_one(model, name, args, device, a.output_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"!! {model} {name} failed: {e} - continuing")
            # keep disk small: the checkpoint is not needed for the summary
            ck = os.path.join(a.output_dir, f"model_{model}_{name}.pth")
            if os.path.exists(ck):
                os.remove(ck)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summarise(a.output_dir)


if __name__ == "__main__":
    main()