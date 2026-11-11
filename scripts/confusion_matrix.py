"""Image- and pixel-level confusion matrices using only label-free (SynthCal) thresholds.

    python scripts/confusion_matrix.py --data-root data/mvtec --out results/confusion

For every category the detector is fit on normal images, calibrated with SynthCal (no defect labels), and
then applied to the test set:
  * image level: an image is called "Defective" when its anomaly score exceeds the calibrated image threshold
  * pixel level: a pixel is "Defective" when it is inside the predicted SynthCal mask
Counts are appended to `counts.csv` after each category (re-running skips finished ones), then plotted.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from masksight.benchmark import _stack
from masksight.calibrate import apply, synthcal
from masksight.data import MVTEC_CATEGORIES, TEXTURES, ImageSet, mvtec_splits
from masksight.metrics import confusion_matrix, confusion_summary
from masksight.models import build_model
from masksight.viz import plot_confusion, plt

LEVELS = ("image", "pixel")


def run_category(cat: str, a) -> dict:
    train, calib, test = mvtec_splits(a.data_root, cat, 0.1, a.seed)
    extra = {"coreset_ratio": a.coreset_ratio} if a.model == "patchcore" else {}
    model = build_model(a.model, arch=a.arch, layers=("layer2", "layer3"), sigma=4.0, **extra)
    model.fit(ImageSet(train))
    apply(model, synthcal(model, _stack(ImageSet(calib)), seed=a.seed, object_category=cat not in TEXTURES))
    maps, scores, masks, labels, _ = model.predict_dataset(ImageSet(test))
    row = {"category": cat}
    for level, cm in (("image", confusion_matrix(scores > model.image_threshold, labels.bool())),
                      ("pixel", confusion_matrix(model.segment(maps), masks))):
        row.update({f"{level}_{k}": int(v) for k, v in zip(("tn", "fp", "fn", "tp"), cm.ravel())})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def matrix(row, level: str) -> np.ndarray:
    return np.array([[row[f"{level}_tn"], row[f"{level}_fp"]], [row[f"{level}_fn"], row[f"{level}_tp"]]])


def headline(cm: np.ndarray) -> str:
    s = confusion_summary(cm)
    return (f"acc {s['accuracy']:.3f} · precision {s['precision']:.3f} · recall {s['recall']:.3f}\n"
            f"specificity {s['specificity']:.3f} · F1 {s['f1']:.3f}")


def plot(df: pd.DataFrame, out: Path, model: str) -> None:
    total = df.drop(columns="category").sum()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5))
    plot_confusion(ax[0], matrix(total, "image"), f"Image level ({len(df)} categories, {model})\n"
                   + headline(matrix(total, "image")))
    plot_confusion(ax[1], matrix(total, "pixel"), "Pixel level (SynthCal masks), row-normalised\n"
                   + headline(matrix(total, "pixel")), normalize=True)
    fig.tight_layout()
    fig.savefig(out / "confusion_overall.png", dpi=150)
    plt.close(fig)

    n = len(df)
    cols = 5
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.4 * rows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        s = confusion_summary(matrix(r, "image"))
        plot_confusion(ax, matrix(r, "image"), f"{r.category}\nacc {s['accuracy']:.2f} · F1 {s['f1']:.2f}")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Image-level confusion matrix per category ({model}, label-free threshold)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "confusion_per_category.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/mvtec")
    p.add_argument("--out", default="results/confusion")
    p.add_argument("--model", default="patchcore", choices=["patchcore", "padim"])
    p.add_argument("--arch", default="wide_resnet50_2")
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--categories", nargs="*", default=MVTEC_CATEGORIES)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "counts.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    for cat in a.categories:
        if len(df) and cat in set(df.category):
            continue
        t0 = time.perf_counter()
        row = run_category(cat, a)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv, index=False)
        s = confusion_summary(matrix(row, "image"))
        print(f"{cat:11s} image TN={row['image_tn']:3d} FP={row['image_fp']:3d} FN={row['image_fn']:3d} "
              f"TP={row['image_tp']:3d}  acc={s['accuracy']:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = df[df.category.isin(a.categories)]
    plot(df, out, a.model)
    for level in LEVELS:
        cm = matrix(df.drop(columns="category").sum(), level)
        print(f"\n{level.upper()} LEVEL, all categories  [[TN, FP], [FN, TP]] =\n{cm}")
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in confusion_summary(cm).items()))
    print(f"\nsaved {out / 'confusion_overall.png'} and {out / 'confusion_per_category.png'}")


if __name__ == "__main__":
    main()
