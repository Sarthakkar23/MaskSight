"""Evaluation metrics: image/pixel AUROC, AUPRO (per-region overlap), and mask quality (Dice / IoU)."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import roc_auc_score

_trapz = getattr(np, "trapezoid", None) or np.trapz


def image_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float(roc_auc_score(labels.numpy(), scores.numpy()))


def pixel_auroc(maps: torch.Tensor, masks: torch.Tensor) -> float:
    return float(roc_auc_score(masks.flatten().numpy(), maps.flatten().numpy()))


def aupro(maps: torch.Tensor, masks: torch.Tensor, fpr_limit: float = 0.3, steps: int = 200) -> float:
    """Area under the per-region-overlap curve up to `fpr_limit`, normalised to [0, 1].

    Unlike pixel AUROC, every connected defect region counts equally, so small defects matter as much
    as large ones (Bergmann et al., MVTec AD, IJCV 2021).
    """
    masks = masks.bool()
    regions = np.zeros(masks.shape, dtype=np.int64)
    offset = 0
    for i, m in enumerate(masks.numpy()):
        lab, n = ndimage.label(m)
        regions[i] = np.where(lab > 0, lab + offset, 0)
        offset += n
    if offset == 0:
        return float("nan")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    regions = torch.from_numpy(regions).to(dev).flatten()
    maps = maps.to(dev).flatten()
    neg = ~masks.to(dev).flatten()
    region_size = torch.bincount(regions, minlength=offset + 1)[1:].float()
    # Thresholds chosen so the false-positive rate on normal pixels sweeps 0 .. fpr_limit exactly.
    neg_sorted = maps[neg].sort(descending=True).values
    n_neg = neg_sorted.numel()
    fprs = torch.linspace(0, fpr_limit, steps)
    pros = []
    for f in fprs:
        k = min(int(f * n_neg), n_neg - 1)
        thr = neg_sorted[k] if f > 0 else neg_sorted[0] + 1e-6
        hit = torch.bincount(regions[maps > thr], minlength=offset + 1)[1:].float()
        pros.append((hit / region_size).mean().item())
    return float(_trapz(pros, fprs.numpy()) / fpr_limit)


def segmentation_scores(pred: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor) -> dict:
    """Dice / IoU of binary predictions averaged over defective images, plus pooled pixel F1."""
    pred = pred.bool()
    gt = masks.bool()
    bad = labels.bool()
    inter = (pred & gt).flatten(1).sum(1).float()
    union = (pred | gt).flatten(1).sum(1).float()
    dice = 2 * inter / (pred.flatten(1).sum(1) + gt.flatten(1).sum(1) + 1e-8)
    tp = inter.sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return {
        "dice": float(dice[bad].mean()),
        "iou": float((inter / (union + 1e-8))[bad].mean()),
        "pixel_f1": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
        "normal_fp_rate": float(pred[~bad].float().mean()) if (~bad).any() else float("nan"),
    }
