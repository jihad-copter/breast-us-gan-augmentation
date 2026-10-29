"""
Re-run ALL revision experiments with 50 epochs (as in the original paper) - ONE command
=====================================================================================
Part A - single-seed experiments (seed 42), 18 runs:
    3 models  x  Orig_NoAug (aug none), Orig_ConvAug, Top5Cycle_ConvAug, Top5Pix2Pix_ConvAug
    DenseNet121 x  Top3 / Top10 / Random5  for CycleGAN and Pix2Pix  (ConvAug)
Part B - multi-seed check with the paper's augmentation ('original'), 27 runs:
    3 models x {Orig, Top5Cycle, Top5Pix2Pix} x seeds 1, 2, 3

Everything is written to results_50ep/ (the old 100-epoch results are not touched).
The script is resumable: finished runs are skipped, so after a stop or a restart
just run the same command again.

At the end it writes:
    paper_table_all.csv     Part A, every metric (percent)
    seeds_mean_sd.csv       Part B, mean +/- SD over seeds
    seeds_gain_vs_orig.csv  Part B, Top-5 minus Orig per seed
Then run:  %run bootstrap_ci.py --results_dir results_50ep --reference Orig_ConvAug

Usage in Spyder (same folder as train_conventional_aug.py and run_seeds.py):
    %run run_all_50.py
    %run run_all_50.py --part A          # only the single-seed experiments
    %run run_all_50.py --part B          # only the multi-seed experiments
    %run run_all_50.py --summary_only
"""

import os
import re
import sys
import glob
import argparse
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_conventional_aug as T  # noqa: E402
import run_seeds as RS               # noqa: E402

BASE = r"F:\jihad\Breast GAN types\breast\breast"
SUB = os.path.join(BASE, "revision_subsets")
TOP5 = {
    "Cycle": os.path.join(BASE, "cyclegan_top5_generated", "topfive"),
    "Pix2Pix": os.path.join(BASE, "pix2pix_top5_generated", "topfive"),
}
MODELS = ["densenet121", "efficientnet_v2_l", "resnet50"]
METRICS = ["ACC", "Balanced_ACC", "F1_macro", "F1_malignant", "TPR", "TNR", "ROC_AUC", "PR_AUC"]


def build_jobs():
    """(part, model, condition_name, aug, seed, extra_dirs)"""
    jobs = []
    # ---- Part A: single seed (42), same names as before ----
    for m in MODELS:
        jobs += [("A", m, "Orig_NoAug", "none", 42, []),
                 ("A", m, "Orig_ConvAug", "conservative", 42, []),
                 ("A", m, "Top5Cycle_ConvAug", "conservative", 42, [TOP5["Cycle"]]),
                 ("A", m, "Top5Pix2Pix_ConvAug", "conservative", 42, [TOP5["Pix2Pix"]])]
    for gan, folder in [("Cycle", "cyclegan"), ("Pix2Pix", "pix2pix")]:
        for sub in ["top3", "top10", "random5"]:
            name = {"top3": "Top3", "top10": "Top10", "random5": "Random5"}[sub]
            jobs.append(("A", "densenet121", f"{name}{gan}_ConvAug", "conservative", 42,
                         [os.path.join(SUB, f"{folder}_{sub}")]))
    # ---- Part B: 3 seeds with the paper's augmentation ----
    for s in [1, 2, 3]:
        for m in MODELS:
            for cond, extra in [("Orig", []), ("Top5Cycle", [TOP5["Cycle"]]),
                                ("Top5Pix2Pix", [TOP5["Pix2Pix"]])]:
                jobs.append(("B", m, RS.run_name(cond, "original", s), "original", s, extra))
    return jobs


def summary_part_a(out_dir):
    rows = []
    for f in glob.glob(os.path.join(out_dir, "pred_*.csv")):
        base = os.path.basename(f)[5:-4]
        rest, split = base.rsplit("_", 1)
        if re.search(r"_s\d+$", rest):          # multi-seed runs are summarised separately
            continue
        model = next(m for m in MODELS if rest.startswith(m + "_"))
        df = pd.read_csv(f)
        m = T.compute_metrics(df["true_label"].to_numpy(), df["prob_malignant"].to_numpy())
        rows.append({"Model": model, "Condition": rest[len(model) + 1:], "Split": split,
                     **{k: round(m[k] * 100, 2) for k in METRICS}})
    if rows:
        t = pd.DataFrame(rows).sort_values(["Split", "Model", "Condition"])
        t.to_csv(os.path.join(out_dir, "paper_table_all.csv"), index=False)
        print("\n=== Part A, External ===")
        print(t[t.Split == "External"].drop(columns="Split").to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--output_dir", default="results_50ep")
    ap.add_argument("--summary_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    if not a.summary_only:
        jobs = [j for j in build_jobs() if a.part == "both" or j[0] == a.part]
        missing = {d for j in jobs for d in j[5] if not os.path.isdir(d)}
        if missing:
            raise FileNotFoundError("Folders not found:\n  " + "\n  ".join(sorted(missing)))
        todo = [j for j in jobs if not RS.done(a.output_dir, j[1], j[2])]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device} | epochs {a.epochs} | {len(jobs)} runs, "
              f"{len(jobs) - len(todo)} already done, {len(todo)} to run")

        for i, (part, model, name, aug, seed, extra) in enumerate(todo, 1):
            print(f"\n##### Run {i}/{len(todo)} (part {part}): {model} | {name} | aug={aug} #####")
            args = SimpleNamespace(
                seed=seed, aug=aug, no_hflip=False, img_size=224, lr=1e-3,
                epochs=a.epochs, batch_size=32, num_workers=0, selection="last",
                train_dir=T.CONFIG["train_dir"], extra_train_dirs=extra)
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

    summary_part_a(a.output_dir)
    RS.summarise(a.output_dir)
    print(f"\nNext: %run bootstrap_ci.py --results_dir {a.output_dir} --reference Orig_ConvAug")


if __name__ == "__main__":
    main()