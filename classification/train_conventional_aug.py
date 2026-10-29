"""
Reviewer item 1 - Conventional augmentation baseline
====================================================
Trains DenseNet121, EfficientNet-V2-L and ResNet50 on the ORIGINAL training
set with conventional (non-GAN) augmentation and evaluates each model on the
INTERNAL test set and the EXTERNAL test set.

The training protocol replicates the author's original training script
(untitled0.py) so the new rows are directly comparable with Tables 2-6:
    ImageNet-pretrained (DEFAULT weights), Adam lr=0.001, batch 32, 50 epochs,
    CrossEntropy, ReduceLROnPlateau(mode='min', patience=3, factor=0.5) stepped
    on the training loss, and the model after the LAST epoch is evaluated
    (no validation-based checkpoint selection).

Augmentation modes (--aug):
    conservative  (default) medically reasonable: rotation +/-10 deg,
                  translation +/-5 %, zoom 0.9-1.1, brightness/contrast +/-15 %,
                  horizontal flip p=0.5, NO vertical flip.
    original      the exact pipeline in untitled0.py: H-flip, V-flip,
                  rotation 15 deg, brightness/contrast 0.2.
    none          resize + normalise only (true no-augmentation baseline).

Metrics (malignant = positive class): ACC, Balanced ACC, F1 (macro),
F1 (malignant), TPR, TNR, ROC-AUC, PR-AUC, TP/TN/FP/FN.

Reusable for later reviewer items (Top-k, Random-5): add synthetic folders with
--extra_train_dirs and name the run with --condition.

Usage (Windows):
    python train_conventional_aug.py
    python train_conventional_aug.py --aug none --condition Orig_NoAug
    python train_conventional_aug.py --models densenet121 --aug original --condition Top3_Cycle ^
        --extra_train_dirs "F:\\...\\cyclegan_top3"
"""

import os
import json
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Configuration  (edit paths if your folder names differ)
# ----------------------------------------------------------------------------
BASE = r"F:\jihad\Breast GAN types\breast\breast"
CONFIG = {
    "train_dir":    os.path.join(BASE, "split_dataset", "train"),
    "val_dir":      os.path.join(BASE, "split_dataset", "val"),    # only used with --selection val
    "test_dir":     os.path.join(BASE, "split_dataset", "test"),   # internal test
    "external_dir": os.path.join(BASE, "test1"),                   # external test
    "output_dir":   "results_conventional_aug",
    "models":       ["densenet121", "efficientnet_v2_l", "resnet50"],
    "epochs":       100,       # matches untitled0.py (manuscript says 100 - see notes)
    "batch_size":   32,
    "lr":           1e-3,
    "img_size":     224,
    "seed":         42,
    "num_workers":  0,        # 0 = Windows-safe, as in untitled0.py
}

# Folder name -> label. Internal uses benign/malignant, external uses b/m.
CLASS_MAP = {"benign": 0, "b": 0, "malignant": 1, "m": 1}
CLASS_NAMES = ["benign", "malignant"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
def build_train_transform(img_size, aug="conservative", use_hflip=True):
    t = [transforms.Resize((img_size, img_size))]
    if aug == "conservative":
        # Medically reasonable for B-mode breast ultrasound. Vertical flip is
        # excluded because the axial (depth) direction carries diagnostic
        # information (posterior shadowing / enhancement).
        t.append(transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                         scale=(0.9, 1.1), fill=0))
        if use_hflip:
            t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
    elif aug == "original":
        # Exact pipeline used in untitled0.py
        t += [transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.RandomRotation(15),
              transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    elif aug != "none":
        raise ValueError(f"Unknown --aug value: {aug}")
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_folder_dataset(path, transform):
    """ImageFolder with labels forced to benign=0, malignant=1
    regardless of folder naming (benign/malignant or b/m)."""
    ds = datasets.ImageFolder(path, transform=transform)
    remap = {}
    for idx, name in enumerate(ds.classes):
        key = name.lower()
        if key not in CLASS_MAP:
            raise ValueError(f"Unknown class folder '{name}' in {path}. "
                             f"Expected one of {list(CLASS_MAP)}")
        remap[idx] = CLASS_MAP[key]
    ds.samples = [(p, remap[t]) for p, t in ds.samples]
    ds.imgs = ds.samples
    ds.targets = [t for _, t in ds.samples]
    return ds


def count_labels(ds):
    if isinstance(ds, ConcatDataset):
        labels = [t for d in ds.datasets for t in d.targets]
    else:
        labels = ds.targets
    return {CLASS_NAMES[k]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}


# ----------------------------------------------------------------------------
# Models (same constructors / DEFAULT weights as untitled0.py)
# ----------------------------------------------------------------------------
def build_model(name, num_classes=2):
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_v2_l":
        m = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return m


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def compute_metrics(y_true, prob_malignant):
    y_pred = (prob_malignant >= 0.5).astype(int)   # identical to argmax for 2 classes
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        roc = roc_auc_score(y_true, prob_malignant)
        pr = average_precision_score(y_true, prob_malignant)
    except ValueError:
        roc, pr = float("nan"), float("nan")
    return {
        "N": int(len(y_true)),
        "ACC": accuracy_score(y_true, y_pred),
        "Balanced_ACC": balanced_accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_malignant": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "TPR": tp / (tp + fn) if (tp + fn) else 0.0,
        "TNR": tn / (tn + fp) if (tn + fp) else 0.0,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ----------------------------------------------------------------------------
# One training run
# ----------------------------------------------------------------------------
def run_one(model_name, condition, args, device, out_dir):
    set_seed(args.seed)
    print(f"\n{'=' * 70}\n{model_name} | {condition} | aug={args.aug}\n{'=' * 70}")

    train_tf = build_train_transform(args.img_size, aug=args.aug, use_hflip=not args.no_hflip)
    eval_tf = build_eval_transform(args.img_size)

    parts = [make_folder_dataset(args.train_dir, train_tf)]
    for d in args.extra_train_dirs:       # synthetic images for later items
        parts.append(make_folder_dataset(d, train_tf))
    train_ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    test_ds = make_folder_dataset(CONFIG["test_dir"], eval_tf)
    ext_ds = make_folder_dataset(CONFIG["external_dir"], eval_tf)
    val_ds = None
    if args.selection == "val":
        val_ds = make_folder_dataset(CONFIG["val_dir"], eval_tf)

    print(f"Train: {count_labels(train_ds)} | Internal test: {count_labels(test_ds)} | "
          f"External: {count_labels(ext_ds)}"
          + (f" | Val: {count_labels(val_ds)}" if val_ds else ""))

    g = torch.Generator().manual_seed(args.seed)
    kw = dict(num_workers=args.num_workers, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=g, **kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kw)
    ext_loader = DataLoader(ext_ds, batch_size=args.batch_size, shuffle=False, **kw)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)
                  if val_ds else None)

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5)

    ckpt_path = os.path.join(out_dir, f"model_{model_name}_{condition}.pth")
    best_f1, best_epoch, history = -1.0, args.epochs, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n_batches, correct, total = 0.0, 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            n_batches += 1
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss = run_loss / n_batches
        scheduler.step(train_loss)            # as in untitled0.py
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": correct / total,
               "lr": optimizer.param_groups[0]["lr"]}

        if val_loader is not None:
            vp, vy = predict(model, val_loader, device)
            vm = compute_metrics(vy, vp)
            row["val_F1_macro"] = vm["F1_macro"]
            if vm["F1_macro"] > best_f1:
                best_f1, best_epoch = vm["F1_macro"], epoch
                torch.save({"model_state_dict": model.state_dict(), "model_name": model_name,
                            "class_names": CLASS_NAMES, "f1_score": best_f1,
                            "epoch": epoch, "condition": condition}, ckpt_path)
        history.append(row)
        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | acc {correct / total:.4f} | "
              f"lr {row['lr']:.2e}" + (f" | val_F1 {row['val_F1_macro']:.4f}" if val_loader else ""))

    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, f"history_{model_name}_{condition}.csv"), index=False)

    if args.selection == "val":
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    else:  # last epoch, as in untitled0.py
        torch.save({"model_state_dict": model.state_dict(), "model_name": model_name,
                    "class_names": CLASS_NAMES, "f1_score": float("nan"),
                    "epoch": args.epochs, "condition": condition}, ckpt_path)

    rows = []
    for split, loader, ds in [("Internal", test_loader, test_ds),
                              ("External", ext_loader, ext_ds)]:
        p, y = predict(model, loader, device)
        m = compute_metrics(y, p)
        rows.append({"Model": model_name, "Condition": condition, "Aug": args.aug,
                     "Split": split, "Epoch_used": best_epoch, **m})
        pd.DataFrame({
            "file": [s[0] for s in ds.samples],
            "true_label": y,
            "prob_malignant": p,
            "pred_label": (p >= 0.5).astype(int),
        }).to_csv(os.path.join(out_dir, f"pred_{model_name}_{condition}_{split}.csv"),
                  index=False)
        print(f"{split:8s} | ACC {m['ACC']:.4f} | BalACC {m['Balanced_ACC']:.4f} | "
              f"F1m {m['F1_macro']:.4f} | TPR {m['TPR']:.4f} | TNR {m['TNR']:.4f} | "
              f"ROC {m['ROC_AUC']:.4f} | PR {m['PR_AUC']:.4f}")
    return rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CONFIG["models"])
    ap.add_argument("--condition", default="Orig_ConvAug",
                    help="Name used in output files / tables")
    ap.add_argument("--aug", choices=["conservative", "original", "none"],
                    default="conservative")
    ap.add_argument("--no_hflip", action="store_true",
                    help="Disable horizontal flip (conservative mode only)")
    ap.add_argument("--selection", choices=["last", "val"], default="last",
                    help="'last' = final epoch (as in untitled0.py); 'val' = best val macro-F1")
    ap.add_argument("--train_dir", default=CONFIG["train_dir"])
    ap.add_argument("--extra_train_dirs", nargs="*", default=[],
                    help="Extra folders (with benign/ and malignant/) added to training")
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--lr", type=float, default=CONFIG["lr"])
    ap.add_argument("--img_size", type=int, default=CONFIG["img_size"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    ap.add_argument("--output_dir", default=CONFIG["output_dir"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"config_{args.condition}.json"), "w") as f:
        json.dump({**CONFIG, **vars(args)}, f, indent=2)

    all_rows = []
    for name in args.models:
        try:
            all_rows += run_one(name, args.condition, args, device, args.output_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!! {name} failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows:
        print("No results.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.output_dir, f"summary_{args.condition}_{ts}.csv"), index=False)

    cols = ["ACC", "Balanced_ACC", "F1_macro", "F1_malignant", "TPR", "TNR", "ROC_AUC", "PR_AUC"]
    paper = df[["Model", "Condition", "Split"] + cols].copy()
    paper[cols] = (paper[cols] * 100).round(2)
    paper.to_csv(os.path.join(args.output_dir, f"paper_table_{args.condition}_{ts}.csv"),
                 index=False)
    print("\n" + paper.to_string(index=False))


if __name__ == "__main__":
    main()