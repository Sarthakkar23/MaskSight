"""Frozen CNN feature extractor producing locally-aware patch embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PatchEmbedder(nn.Module):
    """Concatenates mid-level feature maps (e.g. layer2 + layer3) at the resolution of the first one.

    Each spatial location becomes a patch embedding. A 3x3 average pool gives every patch a bit of
    neighbourhood context, which makes nearest-neighbour matching robust to small misalignments.
    """

    def __init__(self, arch: str = "wide_resnet50_2", layers: tuple[str, ...] = ("layer2", "layer3"),
                 pretrained: bool = True, pool: int = 3):
        super().__init__()
        net = torchvision.models.get_model(arch, weights="DEFAULT" if pretrained else None)
        self.layers = tuple(layers)
        self.pool = pool
        names = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
        last = max(names.index(n) for n in self.layers)
        self.body = nn.ModuleDict({n: getattr(net, n) for n in names[: last + 1]})
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True):  # always frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = {}
        for name, module in self.body.items():
            x = module(x)
            if name in self.layers:
                feats[name] = x
        maps = [feats[n] for n in self.layers]
        size = maps[0].shape[-2:]
        out = []
        for m in maps:
            if self.pool > 1:
                m = F.avg_pool2d(m, self.pool, stride=1, padding=self.pool // 2)
            if m.shape[-2:] != size:
                m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
            out.append(m)
        return torch.cat(out, dim=1)  # B x C x h x w
