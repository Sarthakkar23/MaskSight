"""PaDiM-style model: one multivariate Gaussian per patch position, scored by Mahalanobis distance.

Defard et al., "PaDiM: a Patch Distribution Modeling Framework", ICPR 2021. Reimplemented here.
"""

from __future__ import annotations

import torch

from masksight.models.base import AnomalyModel


class PaDiM(AnomalyModel):
    name = "padim"

    def __init__(self, dims: int = 550, eps: float = 0.01, **kw):
        super().__init__(**kw)
        self.config.update(dims=dims, eps=eps)
        self.dims = dims
        self.eps = eps
        self.channel_idx: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None      # P x d
        self.inv_cov: torch.Tensor | None = None   # P x d x d

    def _select(self, f: torch.Tensor) -> torch.Tensor:
        return f[:, self.channel_idx].flatten(2).permute(0, 2, 1)  # B x P x d

    def _fit_features(self, batches) -> None:
        # Sums are accumulated in float32 on the device around a shift (the first batch's mean) to keep
        # them well conditioned; the covariance and its inverse are then formed in float64 on the CPU,
        # which avoids relying on float64 GPU kernels that some CUDA builds do not ship.
        n, shift, s1, s2 = 0, None, None, None
        for f in batches:
            if self.channel_idx is None:
                g = torch.Generator().manual_seed(0)
                d = min(self.dims, f.shape[1])
                self.channel_idx = torch.randperm(f.shape[1], generator=g)[:d].sort().values.to(f.device)
            x = self._select(f).float()
            if shift is None:
                shift = x.mean(0)
            x = x - shift
            s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
            xx = torch.einsum("bpi,bpj->pij", x, x)
            s2 = xx if s2 is None else s2 + xx
            n += x.shape[0]
        s1, s2 = s1.cpu().double(), s2.cpu().double()
        mean_c = s1 / n
        cov = (s2 - n * torch.einsum("pi,pj->pij", mean_c, mean_c)) / max(n - 1, 1)
        cov += self.eps * torch.eye(cov.shape[-1], dtype=cov.dtype)
        self.mean = (mean_c.float().to(shift.device) + shift)
        self.inv_cov = torch.linalg.inv(cov).float().to(shift.device)

    @torch.no_grad()
    def _patch_scores(self, feats: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feats.shape
        diff = self._select(feats) - self.mean  # B x P x d
        m = torch.einsum("bpi,pij,bpj->bp", diff, self.inv_cov, diff)
        return m.clamp_min(0).sqrt().view(b, h, w)

    def _state(self) -> dict:
        return {"channel_idx": self.channel_idx.cpu(), "mean": self.mean.cpu(), "inv_cov": self.inv_cov.cpu()}

    def _load_state(self, state: dict) -> None:
        self.channel_idx = state["channel_idx"].to(self.device_)
        self.mean = state["mean"].to(self.device_)
        self.inv_cov = state["inv_cov"].to(self.device_)
