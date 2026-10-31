"""Shared logic for patch-feature anomaly models: fitting loop, map post-processing, persistence."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from masksight.backbone import PatchEmbedder
from masksight.data import normalize


class AnomalyModel(torch.nn.Module):
    name = "base"

    def __init__(self, arch: str = "wide_resnet50_2", layers=("layer2", "layer3"), pretrained: bool = True,
                 sigma: float = 4.0, device: str | None = None, **_):
        super().__init__()
        self.config = dict(arch=arch, layers=list(layers), pretrained=pretrained, sigma=sigma)
        self.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = PatchEmbedder(arch, tuple(layers), pretrained=pretrained).to(self.device_)
        self.sigma = sigma
        # Filled in by calibration; stored with the model so `predict` can return binary masks.
        self.pixel_threshold: float | None = None
        self.mask_alpha: float = 0.0
        self.image_threshold: float | None = None

    # ---- to implement -------------------------------------------------------------------------
    def _fit_features(self, batches) -> None:  # iterable of B x C x h x w feature tensors
        raise NotImplementedError

    def _patch_scores(self, feats: torch.Tensor) -> torch.Tensor:  # -> B x h x w
        raise NotImplementedError

    def _state(self) -> dict:
        raise NotImplementedError

    def _load_state(self, state: dict) -> None:
        raise NotImplementedError

    # ---- shared -------------------------------------------------------------------------------
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.embedder(normalize(images.to(self.device_)))

    def fit(self, dataset, batch_size: int = 32, num_workers: int = 2):
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self._fit_features(self.embed(x) for x, *_ in loader)
        return self

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images: B x 3 x H x W in [0,1]. Returns (anomaly maps B x H x W, image scores B), on CPU."""
        h, w = images.shape[-2:]
        scores = self._patch_scores(self.embed(images)).unsqueeze(1)
        maps = F.interpolate(scores, size=(h, w), mode="bilinear", align_corners=False)
        if self.sigma > 0:
            k = 2 * int(4 * self.sigma + 0.5) + 1
            maps = TF.gaussian_blur(maps, kernel_size=[k, k], sigma=[self.sigma, self.sigma])
        maps = maps[:, 0]
        return maps.cpu(), maps.flatten(1).max(1).values.cpu()

    def segment(self, maps: torch.Tensor) -> torch.Tensor:
        """Binary defect masks from anomaly maps using the calibrated floor and peak-relative alpha."""
        from masksight.calibrate import segment

        return segment(maps, self.pixel_threshold, self.mask_alpha)

    def predict_dataset(self, dataset, batch_size: int = 16, num_workers: int = 2):
        maps, scores, masks, labels, paths = [], [], [], [], []
        for x, m, y, p in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
            a, s = self.predict(x)
            maps.append(a); scores.append(s); masks.append(m[:, 0]); labels.append(y); paths += list(p)
        return torch.cat(maps), torch.cat(scores), torch.cat(masks).bool(), torch.cat(labels), paths

    def save(self, path: str | Path) -> None:
        ckpt = {"model": self.name, "config": self.config, "state": self._state(),
                "pixel_threshold": self.pixel_threshold, "mask_alpha": self.mask_alpha,
                "image_threshold": self.image_threshold}
        if not self.config["pretrained"]:  # random backbone: weights are part of the model
            ckpt["embedder"] = self.embedder.state_dict()
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str | None = None):
        model = cls(**ckpt["config"], device=device)
        model._load_state(ckpt["state"])
        if "embedder" in ckpt:
            model.embedder.load_state_dict(ckpt["embedder"])
        model.pixel_threshold = ckpt.get("pixel_threshold")
        model.mask_alpha = ckpt.get("mask_alpha", 0.0)
        model.image_threshold = ckpt.get("image_threshold")
        return model
