import torch

from masksight import metrics
from masksight.calibrate import best_alpha, f1_max_threshold, quantile_threshold, segment


def _toy(n=6, size=32):
    masks = torch.zeros(n, size, size, dtype=torch.bool)
    masks[: n // 2, 8:16, 8:16] = True  # half the images have one square defect
    labels = masks.flatten(1).any(1).long()
    return masks, labels


def test_perfect_map_scores_one():
    masks, labels = _toy()
    maps = masks.float()
    scores = maps.flatten(1).max(1).values
    assert metrics.image_auroc(scores, labels) == 1.0
    assert metrics.pixel_auroc(maps, masks) == 1.0
    assert metrics.aupro(maps, masks) > 0.99


def test_random_map_is_near_chance():
    torch.manual_seed(0)
    masks, _ = _toy(n=20, size=64)
    maps = torch.rand(masks.shape)
    assert 0.45 < metrics.pixel_auroc(maps, masks) < 0.55
    assert metrics.aupro(maps, masks) < 0.25  # random PRO ~= FPR, so area ~= limit/2 normalised


def test_segmentation_scores_at_good_threshold():
    masks, labels = _toy()
    maps = masks.float() + 0.01 * torch.rand(masks.shape)
    s = metrics.segmentation_scores(maps > 0.5, masks, labels)
    assert s["dice"] > 0.99 and s["iou"] > 0.99 and s["normal_fp_rate"] == 0.0


def test_f1_max_threshold_separates_classes():
    masks, _ = _toy()
    maps = masks.float() * 2 + torch.rand(masks.shape) * 0.5
    thr = f1_max_threshold(maps, masks)
    assert 0.5 <= thr <= 2.0


def test_quantile_threshold():
    x = torch.arange(1000).float()
    assert quantile_threshold(x, 0.9) == 900.0


def test_peak_relative_segment_and_best_alpha():
    # a blurry blob: peak 1.0 at the centre, falling off; the true defect is the core
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    blob = torch.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 50.0)
    maps = torch.stack([blob, 3 * blob])  # same shape, different strength
    gt = (blob > 0.5).expand(2, 32, 32)
    assert torch.equal(segment(maps, 0.0, 0.5)[0], segment(maps, 0.0, 0.5)[1])  # strength-invariant
    alpha, dice = best_alpha(maps, gt, floor=0.0)
    assert abs(alpha - 0.5) < 0.05 and dice > 0.95
