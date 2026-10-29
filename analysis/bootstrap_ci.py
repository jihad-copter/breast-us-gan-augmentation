"""
Bootstrap 95% confidence intervals and paired comparisons - no retraining needed.

Reads the pred_<model>_<condition>_<split>.csv files written by
train_conventional_aug.py and produces:

  1. ci_table.csv        - every metric with 95% CI (stratified bootstrap)
  2. paired_tests.csv    - paired bootstrap difference in Balanced ACC / F1 / AUC
                           between conditions for the same model and split
                           (same test images, so the comparison is paired)

Usage in Spyder:
    %run bootstrap_ci.py
    %run bootstrap_ci.py --results_dir results_conventional_aug --n_boot 5000
    %run bootstrap_ci.py --reference Orig_ConvAug
"""

import os
import re
import glob
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score)

METRICS = ["ACC", "Balanced_ACC", "F1_macro", "F1_malignant", "TPR", "TNR", "ROC_AUC", "PR_AUC"]
PAIRED_METRICS = ["Balanced_ACC", "F1_macro", "ROC_AUC"]
NAME_RE = re.compile(r"pred_(densenet121|efficientnet_v2_l|resnet50)_(.+)_(Internal|External)\.csv$")


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    pos, neg = y == 1, y == 0
    out = {
        "ACC": accuracy_score(y, yhat),
        "Balanced_ACC": balanced_accuracy_score(y, yhat),
        "F1_macro": f1_score(y, yhat, average="macro", zero_division=0),
        "F1_malignant": f1_score(y, yhat, pos_label=1, zero_division=0),
        "TPR": (yhat[pos] == 1).mean() if pos.any() else np.nan,
        "TNR": (yhat[neg] == 0).mean() if neg.any() else np.nan,
    }
    try:
        out["ROC_AUC"] = roc_auc_score(y, p)
        out["PR_AUC"] = average_precision_score(y, p)
    except ValueError:
        out["ROC_AUC"] = out["PR_AUC"] = np.nan
    return out


def stratified_indices(y, rng):
    """Resample positives and negatives separately so class counts stay fixed."""
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    return np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                           rng.choice(idx_neg, len(idx_neg), replace=True)])


def load_preds(results_dir):
    runs = {}
    for f in glob.glob(os.path.join(results_dir, "pred_*.csv")):
        m = NAME_RE.search(os.path.basename(f))
        if not m:
            continue
        model, cond, split = m.groups()
        df = pd.read_csv(f).sort_values("file").reset_index(drop=True)
        runs[(model, cond, split)] = df
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_conventional_aug")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", default=None,
                    help="Only compare every condition against this one (e.g. Orig_ConvAug). "
                         "Default: all pairs.")
    args = ap.parse_args()

    runs = load_preds(args.results_dir)
    if not runs:
        print(f"No pred_*.csv files found in {args.results_dir}")
        return
    print(f"Found {len(runs)} prediction files")

    rng = np.random.default_rng(args.seed)

    # ---------------- 1. CIs per run ----------------
    rows = []
    for (model, cond, split), df in sorted(runs.items()):
        y, p = df["true_label"].to_numpy(), df["prob_malignant"].to_numpy()
        point = metrics(y, p)
        boots = {k: [] for k in METRICS}
        for _ in range(args.n_boot):
            i = stratified_indices(y, rng)
            b = metrics(y[i], p[i])
            for k in METRICS:
                boots[k].append(b[k])
        row = {"Model": model, "Condition": cond, "Split": split, "N": len(y)}
        for k in METRICS:
            lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
            row[k] = round(point[k] * 100, 2)
            row[f"{k}_CI"] = f"[{lo * 100:.1f}, {hi * 100:.1f}]"
        rows.append(row)
    ci = pd.DataFrame(rows)
    ci_path = os.path.join(args.results_dir, "ci_table.csv")
    ci.to_csv(ci_path, index=False)
    print(f"\nSaved {ci_path}")
    show = ["Model", "Condition", "Split", "Balanced_ACC", "Balanced_ACC_CI", "ROC_AUC", "ROC_AUC_CI"]
    print(ci[show].to_string(index=False))

    # ---------------- 2. Paired comparisons ----------------
    groups = {}
    for (model, cond, split) in runs:
        groups.setdefault((model, split), []).append(cond)

    tests = []
    for (model, split), conds in sorted(groups.items()):
        if args.reference:
            if args.reference not in conds:
                continue
            pairs = [(args.reference, c) for c in sorted(conds) if c != args.reference]
        else:
            pairs = list(combinations(sorted(conds), 2))
        for a, b in pairs:
            da, db = runs[(model, a, split)], runs[(model, b, split)]
            if not (da["file"].to_numpy() == db["file"].to_numpy()).all():
                print(f"!! file lists differ for {model} {split}: {a} vs {b} - skipped")
                continue
            y = da["true_label"].to_numpy()
            pa, pb = da["prob_malignant"].to_numpy(), db["prob_malignant"].to_numpy()
            ma, mb = metrics(y, pa), metrics(y, pb)
            diffs = {k: [] for k in PAIRED_METRICS}
            for _ in range(args.n_boot):
                i = stratified_indices(y, rng)          # same images for both runs
                ba, bb = metrics(y[i], pa[i]), metrics(y[i], pb[i])
                for k in PAIRED_METRICS:
                    diffs[k].append(bb[k] - ba[k])
            row = {"Model": model, "Split": split, "A": a, "B": b}
            for k in PAIRED_METRICS:
                d = np.asarray(diffs[k])
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                # two-sided bootstrap p-value for H0: no difference
                pval = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                row[f"d_{k} (B-A)"] = round((mb[k] - ma[k]) * 100, 2)
                row[f"d_{k}_CI"] = f"[{lo * 100:.1f}, {hi * 100:.1f}]"
                row[f"p_{k}"] = round(pval, 3)
            tests.append(row)

    if tests:
        tt = pd.DataFrame(tests)
        t_path = os.path.join(args.results_dir, "paired_tests.csv")
        tt.to_csv(t_path, index=False)
        print(f"\nSaved {t_path}")
        print(tt[["Model", "Split", "A", "B", "d_Balanced_ACC (B-A)",
                  "d_Balanced_ACC_CI", "p_Balanced_ACC"]].to_string(index=False))


if __name__ == "__main__":
    main()
