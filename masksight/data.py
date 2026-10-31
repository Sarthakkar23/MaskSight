"""MVTec AD loading. Images are resized + center-cropped; masks follow the same geometry."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_transform(resize: int, crop: int, normalize: bool = True) -> T.Compose:
    steps = [T.Resize(resize), T.CenterCrop(crop), T.ToTensor()]
    if normalize:
        steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(steps)


def mask_transform(resize: int, crop: int) -> T.Compose:
    return T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(crop), T.ToTensor()])


def normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize a [0,1] image tensor (C,H,W) or (B,C,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x - mean) / std


@dataclass
class Sample:
    image: Path
    mask: Path | None
    label: int  # 0 = good, 1 = defective
    defect: str


class ImageSet(Dataset):
    """Yields (image[0,1] tensor, mask tensor, label, path). Normalization happens in the model."""

    def __init__(self, samples: list[Sample], resize: int = 256, crop: int = 224):
        self.samples = samples
        self.crop = crop
        self.img_tf = image_transform(resize, crop, normalize=False)
        self.mask_tf = mask_transform(resize, crop)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = self.img_tf(Image.open(s.image).convert("RGB"))
        if s.mask is not None:
            mask = (self.mask_tf(Image.open(s.mask).convert("L")) > 0.5).float()
        else:
            mask = torch.zeros(1, self.crop, self.crop)
        return img, mask, s.label, str(s.image)


def mvtec_splits(root: str | Path, category: str, calib_fraction: float = 0.1, seed: int = 0):
    """Return (train, calib, test) sample lists. `calib` is a held-out slice of *normal* train images."""
    root = Path(root) / category
    good = sorted((root / "train" / "good").glob("*.png"))
    random.Random(seed).shuffle(good)
    n_cal = max(1, int(len(good) * calib_fraction)) if calib_fraction > 0 else 0
    calib = [Sample(p, None, 0, "good") for p in good[:n_cal]]
    train = [Sample(p, None, 0, "good") for p in good[n_cal:]]
    test = []
    for d in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        for p in sorted(d.glob("*.png")):
            if d.name == "good":
                test.append(Sample(p, None, 0, "good"))
            else:
                test.append(Sample(p, root / "ground_truth" / d.name / f"{p.stem}_mask.png", 1, d.name))
    return train, calib, test


def download_mvtec(root: str | Path, categories: list[str]) -> Path:
    """Download selected MVTec AD categories (CC BY-NC-SA 4.0) from a Hugging Face mirror."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="foersben/mvtec-ad", repo_type="dataset",
                      allow_patterns=[f"{c}/*" for c in categories], local_dir=str(root))
    return Path(root)
