"""
Reviewer request - Random-5 Pix2Pix across the SAME three seeds as the main experiments
=======================================================================================
    DenseNet121 + Random-5 Pix2Pix   seeds 1, 2, 3
    ResNet50    + Random-5 Pix2Pix   seeds 1, 2, 3
= 6 new runs, with exactly the Part-B protocol of run_all_50.py:
    aug 'original' (untitled0.py), 50 epochs, Adam 1e-3, batch 32, last epoch evaluated.

The Random-5 subset is the folder made by make_subsets.py (revision_subsets/pix2pix_random5,
5 randomly chosen synthetic images per original, sampling seed 42). The subset is fixed;
only the CNN seed changes, exactly like the Top-5 runs (fixed Top-5 folder, seeds 1-3).

Results go to results_50ep/ next to the existing Orig / Top5Pix2Pix seed runs, so
they can be compared seed by seed (same seeds, same test images).

Outputs (in results_50ep/):
    pred_<model>_Random5Pix2Pix_original_s<seed>_<split>.csv   (as all other runs)
    random5_seeds_all_runs.csv     every run: Orig, Top5Pix2Pix, Random5Pix2Pix
    random5_seeds_mean_sd.csv      mean +/- SD over the 3 seeds
    random5_seeds_comparisons.csv  paired per-seed differences:
                                       Random5Pix2Pix - Orig
                                       Top5Pix2Pix    - Random5Pix2Pix  (SSIM selection vs random)
The script is resumable: finished runs are skipped.

Usage in Spyder (same folder as train_conventional_aug.py and run_seeds.py):
    %run run_random5_seeds.py
    %run run_random5_seeds.py --summary_only
"""

import os
import sys
import argparse
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_conventional_aug as T  # noqa: E402
import run_seeds as RS               # noqa: E402

BASE = r"F:\jihad\Breast GAN types\breast\breast"
RANDOM5_DIR = os.path.join(BASE, "revision_subsets", "pix2pix_random5")
COND = "Random5Pix2Pix"
AUG = "original"
MODELS = ["densenet121", "resnet50"]
SEEDS = [1, 2, 3]
KEY = ["ACC", "Balanced_ACC", "F1_macro", "F1_malignant", "TPR", "TNR", "ROC_AUC", "PR_AUC"]
CONDS = ["Orig", "Top5Pix2Pix", COND]
COMPARISONS = [(COND, "Orig"), ("Top5Pix2Pix", COND)]   # (B, A) -> B - A


def load_metrics(out_dir, model, cond, seed, split):
    f = os.path.join(out_dir, f"pred_{model}_{RS.run_name(cond, AUG, seed)}_{split}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    m = T.compute_metrics(df["true_label"].to_numpy(), df["prob_malignant"].to_numpy())
    return {k: m[k] * 100 for k in KEY}


def summarise(out_dir, models, seeds):
    rows = []
    for split in ["Internal", "External"]:
        for model in models:
            for cond in CONDS:
                for s in seeds:
                    m = load_metrics(out_dir, model, cond, s, split)
                    if m is None:
                        if split == "External":
                            print(f"  missing: {model} {cond} seed {s}")
                        continue
                    rows.append({"Split": split, "Model": model, "Condition": cond,
                                 "Seed": s, **m})
    if not rows:
        print("No finished runs yet.")
        return
    runs = pd.DataFrame(rows)
    runs.round(2).to_csv(os.path.join(out_dir, "random5_seeds_all_runs.csv"), index=False)

    # mean +/- SD over seeds
    g = runs.groupby(["Split", "Model", "Condition"], sort=False)
    ms, sd = g[KEY].mean(), g[KEY].std(ddof=1)
    table = pd.DataFrame({"n_seeds": g.size()})
    for k in KEY:
        table[k] = [f"{a:.2f} ± {b if pd.notna(b) else 0:.2f}" for a, b in zip(ms[k], sd[k])]
    table = table.reset_index()
    table.to_csv(os.path.join(out_dir, "random5_seeds_mean_sd.csv"), index=False)

    # paired per-seed differences
    comp = []
    for (split, model), sub in runs.groupby(["Split", "Model"], sort=False):
        for b, a in COMPARISONS:
            A = sub[sub.Condition == a].set_index("Seed")
            B = sub[sub.Condition == b].set_index("Seed")
            seeds_ok = A.index.intersection(B.index)
            if len(seeds_ok) == 0:
                continue
            row = {"Split": split, "Model": model, "Comparison": f"{b} - {a}",
                   "n_seeds": len(seeds_ok)}
            for k in ["ACC", "Balanced_ACC", "F1_macro", "TPR", "ROC_AUC"]:
                d = B.loc[seeds_ok, k] - A.loc[seeds_ok, k]
                row[f"d_{k}"] = f"{d.mean():+.2f} ± {d.std(ddof=1) if len(d) > 1 else 0:.2f}"
                row[f"per_seed_{k}"] = " / ".join(f"{v:+.2f}" for v in d)
                row[f"better_{k}"] = f"{int((d > 0).sum())}/{len(d)}"
            comp.append(row)
    cdf = pd.DataFrame(comp)
    cdf.to_csv(os.path.join(out_dir, "random5_seeds_comparisons.csv"), index=False)

    print("\n=== External: mean ± SD over seeds 1-3 ===")
    print(table[table.Split == "External"][["Model", "Condition", "n_seeds", "ACC",
                                            "Balanced_ACC", "TPR", "ROC_AUC"]].to_string(index=False))
    if not cdf.empty:
        print("\n=== External: paired differences (per seed) ===")
        print(cdf[cdf.Split == "External"][["Model", "Comparison", "d_Balanced_ACC",
                                            "per_seed_Balanced_ACC", "better_Balanced_ACC",
                                            "d_ROC_AUC", "better_ROC_AUC"]].to_string(index=False))
    print(f"\nTables saved in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--random5_dir", default=RANDOM5_DIR)
    ap.add_argument("--output_dir", default="results_50ep")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    if not a.summary_only:
        if not os.path.isdir(a.random5_dir):
            raise FileNotFoundError(f"{a.random5_dir}\nRun first: %run make_subsets.py --gan pix2pix")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jobs = [(m, s) for s in a.seeds for m in a.models]
        todo = [(m, s) for m, s in jobs if not RS.done(a.output_dir, m, RS.run_name(COND, AUG, s))]
        print(f"Device: {device} | epochs {a.epochs} | {len(jobs)} runs, "
              f"{len(jobs) - len(todo)} already done, {len(todo)} to run")

        for i, (model, seed) in enumerate(todo, 1):
            name = RS.run_name(COND, AUG, seed)
            print(f"\n##### Run {i}/{len(todo)}: {model} | {name} #####")
            args = SimpleNamespace(
                seed=seed, aug=AUG, no_hflip=False, img_size=224, lr=1e-3,
                epochs=a.epochs, batch_size=32, num_workers=0, selection="last",
                train_dir=T.CONFIG["train_dir"], extra_train_dirs=[a.random5_dir])
            try:
                T.run_one(model, name, args, device, a.output_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"!! {model} {name} failed: {e} - continuing")
            ck = os.path.join(a.output_dir, f"model_{model}_{name}.pth")
            if os.path.exists(ck):
                os.remove(ck)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summarise(a.output_dir, a.models, a.seeds)


if __name__ == "__main__":
    main()
