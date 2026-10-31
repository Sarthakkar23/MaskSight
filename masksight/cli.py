"""Command line interface: `masksight {download,benchmark,fit,predict,report}`."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _download(a):
    from masksight.data import MVTEC_CATEGORIES, download_mvtec

    download_mvtec(a.root, a.categories or MVTEC_CATEGORIES)


def _benchmark(a):
    from masksight.benchmark import load_config, run, summarize

    cfg = load_config(a.config)
    if a.data_root:
        cfg["data_root"] = a.data_root
    if a.categories:
        cfg["categories"] = a.categories
    df = run(cfg)
    print(summarize(df).round(3).to_string())


def _fit(a):
    """Fit one category and save a checkpoint with label-free (SynthCal) thresholds baked in."""
    from masksight.benchmark import _stack
    from masksight.calibrate import apply, synthcal
    from masksight.data import TEXTURES, ImageSet, mvtec_splits
    from masksight.models import build_model

    train, calib, _ = mvtec_splits(a.data_root, a.category, 0.1)
    extra = {"coreset_ratio": a.coreset_ratio, "half_bank": True} if a.model == "patchcore" else {}
    model = build_model(a.model, arch=a.arch, **extra)
    model.fit(ImageSet(train))
    cal = synthcal(model, _stack(ImageSet(calib)), object_category=a.category not in TEXTURES)
    apply(model, cal)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out)
    print(f"saved {a.out}  floor={cal.floor:.3f} alpha={cal.alpha:.2f} image_thr={cal.image_threshold:.3f}")


def _predict(a):
    from PIL import Image

    from masksight.data import image_transform
    from masksight.models import load_model
    from masksight.viz import prediction_panel

    model = load_model(a.checkpoint)
    tf = image_transform(256, 224, normalize=False)
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in a.images])
    maps, scores = model.predict(imgs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p, img, m, s in zip(a.images, imgs, maps, scores):
        verdict = "DEFECT" if model.image_threshold is not None and s > model.image_threshold else "ok"
        print(f"{p}: score={s:.3f} -> {verdict}")
        prediction_panel([img], torch.zeros(1, *m.shape), [m], model.segment(m[None]),
                         [f"{Path(p).name}: {verdict}"], out / f"{Path(p).stem}_pred.png")


def _report(a):
    import pandas as pd

    from masksight.benchmark import summarize

    df = pd.read_csv(a.csv)
    print(summarize(df).round(3).to_markdown())
    if a.per_category:
        cols = ["image_auroc", "aupro", "dice_quantile", "dice_synthcal", "dice_oracle"]
        print(df.pivot_table(index="category", columns="run", values=cols).round(3).to_markdown())


def main(argv=None):
    p = argparse.ArgumentParser(prog="masksight", description=__doc__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("download", help="download MVTec AD categories")
    s.add_argument("--root", default="data/mvtec")
    s.add_argument("--categories", nargs="*")
    s.set_defaults(fn=_download)

    s = sub.add_parser("benchmark", help="run a YAML benchmark config")
    s.add_argument("config")
    s.add_argument("--data-root")
    s.add_argument("--categories", nargs="*")
    s.set_defaults(fn=_benchmark)

    s = sub.add_parser("fit", help="fit one category and save a calibrated checkpoint")
    s.add_argument("--category", required=True)
    s.add_argument("--data-root", default="data/mvtec")
    s.add_argument("--model", default="patchcore", choices=["patchcore", "padim"])
    s.add_argument("--arch", default="wide_resnet50_2")
    s.add_argument("--coreset-ratio", type=float, default=0.01)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_fit)

    s = sub.add_parser("predict", help="predict masks for images with a saved checkpoint")
    s.add_argument("checkpoint")
    s.add_argument("images", nargs="+")
    s.add_argument("--out", default="predictions")
    s.set_defaults(fn=_predict)

    s = sub.add_parser("report", help="print markdown tables from a results CSV")
    s.add_argument("csv")
    s.add_argument("--per-category", action="store_true")
    s.set_defaults(fn=_report)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
