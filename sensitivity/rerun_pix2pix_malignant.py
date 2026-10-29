"""
Re-run Pix2Pix properly: train a SEPARATE malignant generator, then redo everything downstream
==============================================================================================
Why: in pix2pix.py the line `return models` sits inside the class loop, so only the benign
generator was trained and it was also used for the malignant images.

What this script does (old files are NEVER deleted or overwritten):
  step 1  train  : trains the malignant Pix2Pix generator with the UNCHANGED pix2pix.py code
                   (same U-Net/PatchGAN, Adam 2e-4, batch 4, 1000 epochs, L1 x100) on the
                   malignant training images only.  The benign generator was already trained
                   correctly, so it is reused.
  step 2  generate: 10 new malignant images per original with the malignant generator.
                   Benign images (made by the benign generator) are copied unchanged.
  step 3  subsets: SSIM scores + Top-3 / Top-5 / Top-10 / Random-5 folders (make_subsets.py).
  step 4  cnn    : every CNN run that used Pix2Pix images, same protocol as before:
                     Part A  Top5Pix2Pix_ConvAug (3 models), Top3/Top10/Random5Pix2Pix_ConvAug (DenseNet)
                     Part B  Top5Pix2Pix_original_s1-3 (3 models)
                     Random5 Random5Pix2Pix_original_s1-3 (DenseNet, ResNet50)
                   = 21 runs.  Orig / CycleGAN predictions are copied from results_50ep
                   (they do not depend on Pix2Pix), so all tables can be rebuilt.
  step 5  summary: all summary tables + OLD vs NEW comparison for every Pix2Pix run.

Everything new goes to:
    <BASE>/pix2pix_v2/          models, generated images, subsets
    results_50ep_p2p_v2/        CNN predictions and tables

Resumable: finished steps / runs are skipped, so after a stop just run it again.

Usage in Spyder (same folder as pix2pix.py, train_conventional_aug.py, run_seeds.py,
run_all_50.py, run_random5_seeds.py, make_subsets.py):
    %run rerun_pix2pix_malignant.py                 # everything
    %run rerun_pix2pix_malignant.py --step train    # one step only
    %run rerun_pix2pix_malignant.py --step summary
"""

import os
import sys
import glob
import random
import shutil
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE = r"F:\jihad\Breast GAN types\breast\breast"
TRAIN_DIR = os.path.join(BASE, "split_dataset", "train")
OLD_GEN_CANDIDATES = [os.path.join(BASE, "pix2pix_augmented", "generate_images"),
                      os.path.join(BASE, "pix2pix_augmented")]
V2 = os.path.join(BASE, "pix2pix_v2")
TMP_TRAIN = os.path.join(V2, "train_malignant_only")       # contains only malignant/
MODEL_DIR = os.path.join(V2, "pix2pix_models_malignant_enhancement")
GEN_DIR = os.path.join(V2, "generated")
SUB_DIR = os.path.join(V2, "subsets")
OLD_RESULTS = "results_50ep"
NEW_RESULTS = "results_50ep_p2p_v2"
MODELS = ["densenet121", "efficientnet_v2_l", "resnet50"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def old_gen_dir():
    for d in OLD_GEN_CANDIDATES:
        if glob.glob(os.path.join(d, "benign", "pix2pix_*.png")):
            return d
    raise FileNotFoundError("Old Pix2Pix images not found in:\n  " + "\n  ".join(OLD_GEN_CANDIDATES))


# ---------------------------------------------------------------- step 1
def step_train():
    import pix2pix as P
    gpath = os.path.join(MODEL_DIR, "generator.pth")
    if os.path.exists(gpath):
        print(f"[train] malignant generator already exists -> skipped\n        {gpath}")
        return
    # training folder with ONLY malignant/ -> the benign pass finds no images and is skipped,
    # so the unchanged train_pix2pix() trains exactly one model: malignant
    dst = os.path.join(TMP_TRAIN, "malignant")
    os.makedirs(dst, exist_ok=True)
    src = os.path.join(TRAIN_DIR, "malignant")
    for f in os.listdir(src):
        if f.lower().endswith(IMG_EXT) and not os.path.exists(os.path.join(dst, f)):
            shutil.copy2(os.path.join(src, f), dst)
    print(f"[train] {len(os.listdir(dst))} malignant training images")

    set_seed(SEED)
    cwd = os.getcwd()
    os.chdir(V2)                      # pix2pix.py saves models/curves in the working folder
    try:
        models = P.train_pix2pix(train_directory=TMP_TRAIN, epochs=P.EPOCHS, save_interval=25,
                                 augmentation_type="enhancement", class_specific=True)
    finally:
        os.chdir(cwd)
    if not models or "malignant" not in models:
        raise RuntimeError("Malignant Pix2Pix training failed")
    print(f"[train] done -> {gpath}")


# ---------------------------------------------------------------- step 2
def step_generate():
    import pix2pix as P
    out_m = os.path.join(GEN_DIR, "malignant")
    n_have = len(glob.glob(os.path.join(out_m, "pix2pix_*.png")))
    n_orig = len([f for f in os.listdir(os.path.join(TRAIN_DIR, "malignant")) if f.lower().endswith(IMG_EXT)])
    if n_have >= n_orig * 10:
        print(f"[generate] {n_have} malignant images already exist -> skipped")
    else:
        gen = P.Pix2PixGenerator(P.CHANNELS, P.CHANNELS).to(P.device)
        gen.load_state_dict(torch.load(os.path.join(MODEL_DIR, "generator.pth"), map_location=P.device))
        set_seed(SEED)
        P.generate_pix2pix_dataset({"malignant": {"generator": gen}}, train_directory=TMP_TRAIN,
                                   output_directory=GEN_DIR, augmentation_type="enhancement",
                                   images_per_original=10)
    # benign: made by the (correct) benign generator -> copy unchanged
    src_b = os.path.join(old_gen_dir(), "benign")
    out_b = os.path.join(GEN_DIR, "benign")
    os.makedirs(out_b, exist_ok=True)
    k = 0
    for f in glob.glob(os.path.join(src_b, "pix2pix_*.png")):
        d = os.path.join(out_b, os.path.basename(f))
        if not os.path.exists(d):
            shutil.copy2(f, d)
            k += 1
    print(f"[generate] benign: {len(os.listdir(out_b))} files ({k} copied now) | "
          f"malignant: {len(glob.glob(os.path.join(out_m, 'pix2pix_*.png')))} new images")


# ---------------------------------------------------------------- step 3
def step_subsets():
    if os.path.isdir(os.path.join(SUB_DIR, "pix2pix_random5", "malignant")):
        print("[subsets] already made -> skipped")
    else:
        import make_subsets as MS
        old_argv = sys.argv
        sys.argv = ["make_subsets.py", "--gan", "pix2pix", "--gen_dir", GEN_DIR,
                    "--out_root", SUB_DIR, "--seed", str(SEED)]
        try:
            MS.main()
        finally:
            sys.argv = old_argv
    sc = pd.read_csv(os.path.join(SUB_DIR, "ssim_scores_pix2pix.csv"))
    print("\n[subsets] paired SSIM per class (all 10 candidates):")
    print(sc.groupby("class")["ssim"].describe().round(3).to_string())


# ---------------------------------------------------------------- step 4
def cnn_jobs():
    sub = lambda k: [os.path.join(SUB_DIR, f"pix2pix_{k}")]
    jobs = [(m, "Top5Pix2Pix_ConvAug", "conservative", 42, sub("top5")) for m in MODELS]
    for k, name in [("top3", "Top3"), ("top10", "Top10"), ("random5", "Random5")]:
        jobs.append(("densenet121", f"{name}Pix2Pix_ConvAug", "conservative", 42, sub(k)))
    for s in [1, 2, 3]:
        for m in MODELS:
            jobs.append((m, f"Top5Pix2Pix_original_s{s}", "original", s, sub("top5")))
        for m in ["densenet121", "resnet50"]:
            jobs.append((m, f"Random5Pix2Pix_original_s{s}", "original", s, sub("random5")))
    return jobs


def step_cnn(epochs):
    import train_conventional_aug as T
    import run_seeds as RS
    os.makedirs(NEW_RESULTS, exist_ok=True)
    # Orig / CycleGAN runs do not depend on Pix2Pix -> reuse them
    k = 0
    for f in glob.glob(os.path.join(OLD_RESULTS, "pred_*.csv")):
        if "Pix2Pix" not in os.path.basename(f):
            d = os.path.join(NEW_RESULTS, os.path.basename(f))
            if not os.path.exists(d):
                shutil.copy2(f, d)
                k += 1
    print(f"[cnn] reused {k} Orig/CycleGAN prediction files from {OLD_RESULTS}")

    jobs = cnn_jobs()
    todo = [j for j in jobs if not RS.done(NEW_RESULTS, j[0], j[1])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cnn] device {device} | {len(jobs)} runs, {len(jobs) - len(todo)} done, {len(todo)} to run")
    for i, (model, name, aug, seed, extra) in enumerate(todo, 1):
        print(f"\n##### CNN run {i}/{len(todo)}: {model} | {name} #####")
        args = SimpleNamespace(seed=seed, aug=aug, no_hflip=False, img_size=224, lr=1e-3,
                               epochs=epochs, batch_size=32, num_workers=0, selection="last",
                               train_dir=T.CONFIG["train_dir"], extra_train_dirs=extra)
        try:
            T.run_one(model, name, args, device, NEW_RESULTS)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!! {model} {name} failed: {e} - continuing")
        ck = os.path.join(NEW_RESULTS, f"model_{model}_{name}.pth")
        if os.path.exists(ck):
            os.remove(ck)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- step 5
def step_summary():
    import train_conventional_aug as T
    import run_seeds as RS
    import run_all_50 as RA
    import run_random5_seeds as R5
    RA.summary_part_a(NEW_RESULTS)
    RS.summarise(NEW_RESULTS)
    R5.summarise(NEW_RESULTS, ["densenet121", "resnet50"], [1, 2, 3])

    rows = []
    for f in sorted(glob.glob(os.path.join(NEW_RESULTS, "pred_*Pix2Pix*_External.csv"))):
        name = os.path.basename(f)
        old = os.path.join(OLD_RESULTS, name)
        if not os.path.exists(old):
            continue
        mn = T.compute_metrics(*pd.read_csv(f)[["true_label", "prob_malignant"]].to_numpy().T)
        mo = T.compute_metrics(*pd.read_csv(old)[["true_label", "prob_malignant"]].to_numpy().T)
        row = {"run": name[5:-13]}
        for k in ["Balanced_ACC", "TPR", "ROC_AUC"]:
            row[f"{k}_old"] = round(mo[k] * 100, 2)
            row[f"{k}_new"] = round(mn[k] * 100, 2)
            row[f"{k}_diff"] = round((mn[k] - mo[k]) * 100, 2)
        rows.append(row)
    if rows:
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(os.path.join(NEW_RESULTS, "pix2pix_old_vs_new.csv"), index=False)
        print("\n=== External: OLD (benign generator for both) vs NEW (separate malignant generator) ===")
        print(cmp_df.to_string(index=False))
    print(f"\nNext: %run bootstrap_ci.py --results_dir {NEW_RESULTS} --reference Orig_ConvAug")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["all", "train", "generate", "subsets", "cnn", "summary"])
    ap.add_argument("--epochs", type=int, default=50, help="CNN epochs (50, as in the revision runs)")
    a = ap.parse_args()
    os.makedirs(V2, exist_ok=True)
    steps = ["train", "generate", "subsets", "cnn", "summary"] if a.step == "all" else [a.step]
    for s in steps:
        print("\n" + "=" * 70 + f"\nSTEP: {s}\n" + "=" * 70)
        if s == "train":
            step_train()
        elif s == "generate":
            step_generate()
        elif s == "subsets":
            step_subsets()
        elif s == "cnn":
            step_cnn(a.epochs)
        else:
            step_summary()


if __name__ == "__main__":
    main()
