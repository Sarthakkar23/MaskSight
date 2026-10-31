"""Figures: prediction panels and synthetic-defect previews."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def overlay(img: torch.Tensor, amap: torch.Tensor, vmax: float, alpha: float = 0.5) -> np.ndarray:
    """RGB uint8 image with a jet heatmap blended on top."""
    rgb = img.permute(1, 2, 0).numpy()
    heat = plt.get_cmap("jet")(np.clip(amap.numpy() / vmax, 0, 1))[..., :3]
    return (255 * ((1 - alpha) * rgb + alpha * heat)).astype(np.uint8)


def prediction_panel(images, masks, maps, preds, titles, path: str | Path, vmax: float | None = None):
    """Rows of: input | ground truth | heatmap | predicted mask outline (red) vs ground truth (green)."""
    n = len(images)
    vmax = vmax or float(np.percentile(torch.stack(list(maps)).numpy(), 99.9))
    fig, ax = plt.subplots(n, 4, figsize=(11, 2.8 * n), squeeze=False)
    for r in range(n):
        img = images[r].permute(1, 2, 0).numpy()
        ax[r, 0].imshow(img); ax[r, 0].set_title(titles[r], fontsize=9)
        ax[r, 1].imshow(masks[r], cmap="gray"); ax[r, 1].set_title("ground truth", fontsize=9)
        ax[r, 2].imshow(overlay(images[r], maps[r], vmax)); ax[r, 2].set_title("anomaly heatmap", fontsize=9)
        ax[r, 3].imshow(img)
        if masks[r].any():
            ax[r, 3].contour(masks[r].numpy().astype(float), levels=[0.5], colors="lime", linewidths=1)
        if preds[r].any():
            ax[r, 3].contour(preds[r].numpy().astype(float), levels=[0.5], colors="red", linewidths=1.2)
        ax[r, 3].set_title("predicted (red) vs true (green)", fontsize=9)
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def synthetic_panel(images, masks, path: str | Path, n: int = 6):
    n = min(n, len(images))
    fig, ax = plt.subplots(2, n, figsize=(2.2 * n, 4.6), squeeze=False)
    for i in range(n):
        ax[0, i].imshow(images[i].permute(1, 2, 0).numpy()); ax[0, i].axis("off")
        ax[1, i].imshow(masks[i], cmap="gray"); ax[1, i].axis("off")
    ax[0, 0].set_title("synthetic defect", fontsize=9, loc="left")
    ax[1, 0].set_title("free label", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
