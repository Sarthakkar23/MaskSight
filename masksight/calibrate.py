"""Turning an anomaly heatmap into a binary defect mask, without defect labels.

Strategies compared in the benchmark (least to most information used):
  * quantile     - absolute threshold: a high quantile of scores on held-out normal images
  * synth-abs    - absolute threshold maximising pixel F1 on synthetic defects (naive SynthCal)
  * synthcal     - ours: peak-relative threshold. A pixel is defective if its score exceeds both the
                   normal-image floor and `alpha * (that image's peak score)`; alpha is chosen to maximise
                   Dice on synthetic defects. Because alpha describes the *shape* of a defect's response
                   rather than its strength, it transfers from synthetic to real defects even when the
                   synthetic ones are much more (or less) conspicuous than real ones.
  * oracle       - absolute threshold maximising pixel F1 on the real test labels (upper bound)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sklearn.metrics import roc_curve

from masksight.synth import make_synthetic_set

ALPHAS = torch.linspace(0.2, 0.95, 31)


def segment(maps: torch.Tensor, floor: float, alpha: float = 0.0) -> torch.Tensor:
    """Binary masks: score above the absolute floor and above alpha x the image's own peak."""
    peak = maps.flatten(1).max(1).values.view(-1, *([1] * (maps.dim() - 1)))
    return (maps > floor) & (maps > alpha * peak)


def mean_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Dice averaged over images that contain a defect."""
    has = gt.flatten(1).any(1)
    inter = (pred & gt).flatten(1).sum(1).float()
    dice = 2 * inter / (pred.flatten(1).sum(1) + gt.flatten(1).sum(1) + 1e-8)
    return float(dice[has].mean())


def best_alpha(maps: torch.Tensor, masks: torch.Tensor, floor: float, alphas=ALPHAS) -> tuple[float, float]:
    scores = torch.tensor([mean_dice(segment(maps, floor, float(a)), masks) for a in alphas])
    i = int(scores.argmax())
    return float(alphas[i]), float(scores[i])


def f1_max_threshold(maps: torch.Tensor, masks: torch.Tensor, bins: int = 2000) -> float:
    """Absolute threshold maximising pooled pixel F1, computed in O(N) with score histograms."""
    s = maps.flatten().float()
    y = masks.flatten().bool()
    lo, hi = s.min().item(), s.max().item()
    if hi <= lo:
        return hi
    pos = torch.histc(s[y], bins=bins, min=lo, max=hi)
    neg = torch.histc(s[~y], bins=bins, min=lo, max=hi)
    tp = pos.flip(0).cumsum(0).flip(0)   # predicted positive = score >= left edge of bin
    fp = neg.flip(0).cumsum(0).flip(0)
    fn = pos.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    edges = torch.linspace(lo, hi, bins + 1)
    return float(edges[int(torch.argmax(f1))])


def image_threshold(anomalous_scores: torch.Tensor, normal_scores: torch.Tensor) -> float:
    """Image-level threshold maximising Youden's J (TPR - FPR); insensitive to class imbalance."""
    y = torch.cat([torch.ones(len(anomalous_scores)), torch.zeros(len(normal_scores))]).numpy()
    s = torch.cat([anomalous_scores, normal_scores]).numpy()
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int((tpr - fpr).argmax())])


def quantile_threshold(normal_maps: torch.Tensor, q: float = 0.999) -> float:
    flat = normal_maps.flatten()
    k = min(int(q * flat.numel()), flat.numel() - 1)
    return float(flat.sort().values[k])


@dataclass
class Calibration:
    floor: float            # quantile of normal pixel scores
    alpha: float            # peak-relative factor learned on synthetic defects
    image_threshold: float  # anomaly score above which an image is called defective
    synth_abs_threshold: float
    synthetic_dice: float   # Dice on the synthetic set itself (a label-free sanity signal)
    artifacts: dict = field(default_factory=dict, repr=False)


def _predict_all(model, images: torch.Tensor, batch_size: int = 16):
    out = [model.predict(images[i:i + batch_size]) for i in range(0, len(images), batch_size)]
    return torch.cat([m for m, _ in out]), torch.cat([s for _, s in out])


def synthcal(model, calib_images: torch.Tensor, quantile: float = 0.999, per_image: int = 3, seed: int = 0,
             object_category: bool = True) -> Calibration:
    """Calibrate everything needed to output masks, using held-out normal images only."""
    normal_maps, normal_scores = _predict_all(model, calib_images)
    floor = quantile_threshold(normal_maps, quantile)
    syn_imgs, syn_masks = make_synthetic_set(calib_images, per_image, seed, object_category)
    syn_maps, syn_scores = _predict_all(model, syn_imgs)
    alpha, syn_dice = best_alpha(syn_maps, syn_masks, floor)
    no_defect = torch.zeros(len(calib_images), *syn_masks.shape[1:], dtype=torch.bool)
    abs_thr = f1_max_threshold(torch.cat([syn_maps, normal_maps]), torch.cat([syn_masks, no_defect]))
    return Calibration(floor, alpha, image_threshold(syn_scores, normal_scores), abs_thr, syn_dice,
                       {"synthetic_images": syn_imgs, "synthetic_masks": syn_masks, "synthetic_maps": syn_maps})


def apply(model, cal: Calibration) -> None:
    """Store calibration on the model so `model.segment` / checkpoints carry it."""
    model.pixel_threshold, model.mask_alpha, model.image_threshold = cal.floor, cal.alpha, cal.image_threshold
