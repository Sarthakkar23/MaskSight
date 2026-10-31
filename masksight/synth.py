"""Synthetic defect generator used for label-free threshold calibration (SynthCal).

We paint realistic-looking blobs onto held-out *normal* images: the blob shape comes from thresholded
Perlin noise (irregular, multi-scale outlines like stains, cracks and scratches), and its content is
either a colour-jittered patch of another normal image, a shifted copy of the image itself, or a
flat stain. Because we generated the defect, we know its exact mask, which gives us free pixel labels.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import ndimage


def _fade(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def perlin_noise(shape: tuple[int, int], res: tuple[int, int], g: torch.Generator) -> torch.Tensor:
    """2D gradient (Perlin) noise. `shape` must be divisible by `res`. Values roughly in [-1, 1]."""
    h, w = shape
    d = (h // res[0], w // res[1])
    ys = torch.arange(h, dtype=torch.float32) * res[0] / h
    xs = torch.arange(w, dtype=torch.float32) * res[1] / w
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, generator=g)
    grads = torch.stack((torch.cos(angles), torch.sin(angles)), -1)

    def tile(r0, r1, c0, c1):
        return grads[r0:r1, c0:c1].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

    def dot(grad, sy, sx):
        return ((grid[..., 0] + sy) * grad[..., 0]) + ((grid[..., 1] + sx) * grad[..., 1])

    n00 = dot(tile(0, -1, 0, -1), 0, 0)
    n10 = dot(tile(1, None, 0, -1), -1, 0)
    n01 = dot(tile(0, -1, 1, None), 0, -1)
    n11 = dot(tile(1, None, 1, None), -1, -1)
    t = _fade(grid)
    top = torch.lerp(n00, n10, t[..., 0])
    bottom = torch.lerp(n01, n11, t[..., 0])
    return math.sqrt(2) * torch.lerp(top, bottom, t[..., 1])


def _divisors(n: int, options=(1, 2, 4, 7)) -> list[int]:
    return [r for r in options if n % r == 0] or [1]


def _pick(options: list, g: torch.Generator):
    return options[int(torch.randint(len(options), (1,), generator=g))]


def foreground(img: torch.Tensor, tol: float = 0.12) -> torch.Tensor:
    """Rough object mask: pixels whose colour differs from the median border colour."""
    border = torch.cat([img[:, 0], img[:, -1], img[:, :, 0], img[:, :, -1]], dim=1)
    bg = border.median(dim=1).values.view(3, 1, 1)
    fg = (img - bg).abs().amax(0) > tol
    fg = F.max_pool2d(fg[None, None].float(), 9, 1, 4)[0, 0] > 0  # close small holes
    return fg if fg.float().mean() > 0.05 else torch.ones_like(fg)


def _keep_largest(m: torch.Tensor, k: int) -> torch.Tensor:
    """Real defects are one or a few connected regions, not a spray of specks."""
    lab, n = ndimage.label(m.numpy())
    if n <= k:
        return m
    sizes = ndimage.sum(m.numpy(), lab, range(1, n + 1))
    keep = 1 + sizes.argsort()[::-1][:k]
    return torch.from_numpy(np.isin(lab, keep))


def defect_mask(shape, g: torch.Generator, region: torch.Tensor | None = None,
                area=(0.005, 0.10), tries: int = 30) -> torch.Tensor:
    h, w = shape
    for _ in range(tries):
        ry, rx = (_pick(_divisors(n), g) for n in (h, w))
        noise = perlin_noise((h, w), (ry, rx), g)
        thr = 0.3 + 0.3 * torch.rand(1, generator=g).item()
        m = noise > thr
        if region is not None:
            m &= region
        m = _keep_largest(m, k=int(torch.randint(1, 3, (1,), generator=g)))
        frac = m.float().mean().item()
        if area[0] <= frac <= area[1]:
            return m
    # Fallback: a random ellipse inside the region
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = (torch.rand(2, generator=g) * torch.tensor([h, w]) * 0.6 + torch.tensor([h, w]) * 0.2).tolist()
    ry, rx = (torch.rand(2, generator=g) * 0.08 + 0.03).tolist()
    m = ((yy - cy) / (ry * h)) ** 2 + ((xx - cx) / (rx * w)) ** 2 < 1
    return m & region if region is not None and (m & region).any() else m


def _jitter(img: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    r = torch.rand(4, generator=g).tolist()
    img = TF.adjust_brightness(img, 0.5 + r[0])
    img = TF.adjust_contrast(img, 0.5 + r[1])
    img = TF.adjust_saturation(img, 2 * r[2])
    img = TF.adjust_hue(img, r[3] - 0.5)
    return img.clamp(0, 1)


def synthesize(img: torch.Tensor, donors: list[torch.Tensor], g: torch.Generator,
               object_category: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (defective image, bool mask) for a [0,1] image tensor of shape 3 x H x W."""
    _, h, w = img.shape
    region = foreground(img) if object_category else None
    mask = defect_mask((h, w), g, region)
    kind = int(torch.randint(3, (1,), generator=g))
    if kind == 0:   # foreign texture from another normal image
        src = donors[int(torch.randint(len(donors), (1,), generator=g))]
        src = torch.rot90(src, int(torch.randint(4, (1,), generator=g)), dims=(1, 2))
        src = _jitter(src, g)
    elif kind == 1:  # displaced copy of itself (misplaced structure)
        dy, dx = torch.randint(h // 8, h // 2, (2,), generator=g).tolist()
        src = torch.roll(img.flip(-1), shifts=(dy, dx), dims=(1, 2))
    else:            # flat stain
        src = torch.rand(3, 1, 1, generator=g).expand_as(img)
    beta = 0.4 + 0.6 * torch.rand(1, generator=g).item()
    m = mask[None].float()
    out = img * (1 - beta * m) + src * (beta * m)
    return out.clamp(0, 1), mask


def make_synthetic_set(images: torch.Tensor, per_image: int = 2, seed: int = 0, object_category: bool = True):
    """images: N x 3 x H x W normal images in [0,1]. Returns (N*per_image images, masks)."""
    g = torch.Generator().manual_seed(seed)
    donors = list(images)
    outs, masks = [], []
    for img in images:
        for _ in range(per_image):
            o, m = synthesize(img, donors, g, object_category)
            outs.append(o); masks.append(m)
    return torch.stack(outs), torch.stack(masks)
