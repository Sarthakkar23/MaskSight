"""PatchCore-style model: a coreset memory bank of normal patches + nearest-neighbour distance.

Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022. Reimplemented here.
"""

from __future__ import annotations

import torch

from masksight.models.base import AnomalyModel


def greedy_coreset(bank: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center selection: repeatedly add the patch farthest from everything chosen so far.

    Distances are computed in a random low-dimensional projection (Johnson-Lindenstrauss) for speed.
    Returns the indices of the selected rows.
    """
    n = bank.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return torch.arange(n, device=bank.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(bank.shape[1], proj_dim, generator=g).to(bank.device, bank.dtype)
    z = bank @ proj
    idx = torch.empty(k, dtype=torch.long, device=bank.device)
    idx[0] = 0
    min_d = ((z - z[0]) ** 2).sum(1)
    for i in range(1, k):
        j = torch.argmax(min_d)
        idx[i] = j
        min_d = torch.minimum(min_d, ((z - z[j]) ** 2).sum(1))
    return idx


class PatchCore(AnomalyModel):
    name = "patchcore"

    def __init__(self, coreset_ratio: float = 0.1, chunk: int = 4096, half_bank: bool = False, **kw):
        super().__init__(**kw)
        self.config.update(coreset_ratio=coreset_ratio, half_bank=half_bank)
        self.coreset_ratio = coreset_ratio
        self.chunk = chunk
        self.half_bank = half_bank
        self.bank: torch.Tensor | None = None

    def _fit_features(self, batches) -> None:
        patches = [f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]) for f in batches]
        bank = torch.cat(patches)
        bank = bank[greedy_coreset(bank, self.coreset_ratio)]
        self.bank = bank.half() if self.half_bank else bank

    @torch.no_grad()
    def _patch_scores(self, feats: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feats.shape
        q = feats.permute(0, 2, 3, 1).reshape(-1, c)
        bank = self.bank.float()
        out = torch.empty(q.shape[0], device=q.device)
        for i in range(0, q.shape[0], self.chunk):
            out[i:i + self.chunk] = torch.cdist(q[i:i + self.chunk], bank).min(1).values
        return out.view(b, h, w)

    def _state(self) -> dict:
        return {"bank": self.bank.cpu()}

    def _load_state(self, state: dict) -> None:
        self.bank = state["bank"].to(self.device_)
