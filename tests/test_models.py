"""End-to-end checks on a toy dataset with a randomly initialised ResNet-18 (no downloads, CPU only)."""

import pytest
import torch
from torch.utils.data import Dataset

from masksight import build_model, load_model
from masksight.models.patchcore import greedy_coreset

SIZE = 64


class Toy(Dataset):
    """Smooth grey textures; defective images get a bright square."""

    def __init__(self, n, defect=False, seed=0):
        g = torch.Generator().manual_seed(seed)
        base = 0.5 + 0.05 * torch.randn(n, 1, SIZE // 8, SIZE // 8, generator=g)
        self.x = torch.nn.functional.interpolate(base, size=SIZE, mode="bilinear").repeat(1, 3, 1, 1)
        self.m = torch.zeros(n, 1, SIZE, SIZE)
        if defect:
            self.x[:, :, 20:36, 20:36] = 1.0
            self.m[:, :, 20:36, 20:36] = 1.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.m[i], int(self.m[i].any()), f"toy{i}"


KW = dict(arch="resnet18", layers=("layer2", "layer3"), pretrained=False, sigma=1.0, device="cpu")


@pytest.mark.parametrize("name,extra", [("patchcore", {"coreset_ratio": 0.5}), ("padim", {"dims": 32})])
def test_model_ranks_defects_higher(name, extra, tmp_path):
    torch.manual_seed(0)
    model = build_model(name, **KW, **extra).fit(Toy(8), batch_size=4, num_workers=0)
    maps_good, s_good = model.predict(Toy(2, seed=1).x)
    maps_bad, s_bad = model.predict(Toy(2, defect=True, seed=2).x)
    assert maps_bad.shape == (2, SIZE, SIZE)
    assert s_bad.min() > s_good.max()
    # the defect region should be hotter than the rest of the defective image
    assert maps_bad[:, 20:36, 20:36].mean() > maps_bad[:, :10, :10].mean()

    model.pixel_threshold = 1.23
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = load_model(path, device="cpu")
    assert loaded.pixel_threshold == 1.23
    torch.testing.assert_close(loaded.predict(Toy(2, seed=1).x)[0], maps_good)


def test_greedy_coreset_picks_spread_out_points():
    pts = torch.cat([torch.zeros(50, 4), torch.ones(1, 4) * 10])
    idx = greedy_coreset(pts, ratio=2 / 51)
    assert 50 in idx.tolist()  # the outlier is always kept
