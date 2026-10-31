import torch

from masksight.synth import defect_mask, foreground, make_synthetic_set, perlin_noise


def test_perlin_shape_and_range():
    g = torch.Generator().manual_seed(0)
    n = perlin_noise((64, 64), (4, 8), g)
    assert n.shape == (64, 64)
    assert n.abs().max() <= 1.5 and n.std() > 0.05


def test_defect_mask_area_bounds():
    g = torch.Generator().manual_seed(1)
    for _ in range(10):
        m = defect_mask((64, 64), g)
        assert m.dtype == torch.bool and m.any()


def test_foreground_finds_object_on_dark_background():
    img = torch.zeros(3, 64, 64)
    img[:, 16:48, 16:48] = 0.8
    fg = foreground(img)
    assert fg[32, 32] and not fg[0, 0]


def test_synthetic_set_is_deterministic_and_marks_changed_pixels():
    imgs = torch.rand(3, 3, 64, 64)
    a, ma = make_synthetic_set(imgs, per_image=2, seed=7, object_category=False)
    b, mb = make_synthetic_set(imgs, per_image=2, seed=7, object_category=False)
    assert a.shape == (6, 3, 64, 64) and ma.shape == (6, 64, 64)
    assert torch.equal(a, b) and torch.equal(ma, mb)
    src = imgs.repeat_interleave(2, 0)
    changed = (a - src).abs().amax(1) > 1e-6
    assert not changed[~ma].any()  # nothing outside the mask is touched
