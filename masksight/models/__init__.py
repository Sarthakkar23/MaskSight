from __future__ import annotations

import torch

from masksight.models.base import AnomalyModel
from masksight.models.padim import PaDiM
from masksight.models.patchcore import PatchCore

MODELS: dict[str, type[AnomalyModel]] = {"patchcore": PatchCore, "padim": PaDiM}


def build_model(name: str, **kwargs) -> AnomalyModel:
    return MODELS[name](**kwargs)


def load_model(path: str, device: str | None = None) -> AnomalyModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return MODELS[ckpt["model"]].from_checkpoint(ckpt, device=device)


__all__ = ["AnomalyModel", "PatchCore", "PaDiM", "MODELS", "build_model", "load_model"]
