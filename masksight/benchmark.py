"""Benchmark runner: fit -> calibrate -> evaluate for every (category, run) pair.

Rows are appended to a CSV as soon as they finish and finished rows are skipped on restart, so a
Colab disconnect costs at most one category.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from masksight import metrics
from masksight.calibrate import apply, best_alpha, f1_max_threshold, segment, synthcal
from masksight.data import MVTEC_CATEGORIES, TEXTURES, ImageSet, mvtec_splits
from masksight.models import build_model
from masksight.viz import prediction_panel, synthetic_panel


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("categories") in (None, "all"):
        cfg["categories"] = MVTEC_CATEGORIES
    return cfg


def _stack(ds) -> torch.Tensor:
    return torch.stack([ds[i][0] for i in range(len(ds))])


def _latency_ms(model, images: torch.Tensor, reps: int = 20) -> float:
    x = images[:1]
    for _ in range(3):
        model.predict(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(reps):
        model.predict(images[i % len(images)][None])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps


def _image_f1(scores, labels, thr) -> float:
    pred = scores > thr
    tp = (pred & labels.bool()).sum().item()
    return 2 * tp / (pred.sum().item() + labels.sum().item() + 1e-8)


def run_one(cfg: dict, category: str, run: dict, out_dir: Path) -> dict:
    run = dict(run)
    run_name, model_name = run.pop("name"), run.pop("model")
    model_kw = {**cfg.get("model_defaults", {}), **run}
    seed = cfg.get("seed", 0)
    train, calib, test = mvtec_splits(cfg["data_root"], category, cfg.get("calib_fraction", 0.1), seed)
    size = dict(resize=cfg.get("resize", 256), crop=cfg.get("crop", 224))
    train_ds, calib_ds, test_ds = ImageSet(train, **size), ImageSet(calib, **size), ImageSet(test, **size)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = build_model(model_name, **model_kw).fit(train_ds)
    fit_s = time.perf_counter() - t0

    # --- calibration: everything here uses held-out normal images only ------------------------
    calib_imgs = _stack(calib_ds)
    cal = synthcal(model, calib_imgs, cfg.get("quantile", 0.999), cfg.get("synth_per_image", 3), seed,
                   object_category=category not in TEXTURES)
    apply(model, cal)

    maps, scores, masks, labels, paths = model.predict_dataset(test_ds)
    oracle_thr = f1_max_threshold(maps, masks)
    oracle_alpha, _ = best_alpha(maps, masks, cal.floor)
    predictions = {
        "quantile": segment(maps, cal.floor),
        "synthabs": segment(maps, cal.synth_abs_threshold),
        "synthcal": model.segment(maps),
        "oracle": segment(maps, oracle_thr),
        "oraclerel": segment(maps, cal.floor, oracle_alpha),
    }

    row = {"category": category, "run": run_name, "model": model_name,
           "image_auroc": metrics.image_auroc(scores, labels),
           "pixel_auroc": metrics.pixel_auroc(maps, masks),
           "aupro": metrics.aupro(maps, masks),
           "image_f1_synthcal": _image_f1(scores, labels, cal.image_threshold),
           "alpha_synthcal": cal.alpha, "alpha_oracle": oracle_alpha, "synthetic_dice": cal.synthetic_dice}
    for tag, pred in predictions.items():
        for k, v in metrics.segmentation_scores(pred, masks, labels).items():
            row[f"{k}_{tag}"] = v
    row.update(fit_seconds=fit_s, latency_ms=_latency_ms(model, calib_imgs),
               peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0,
               n_train=len(train_ds), n_test=len(test_ds))

    if cfg.get("save_figures", True):
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        bad = torch.where(labels == 1)[0]
        pick = bad[torch.linspace(0, len(bad) - 1, min(4, len(bad))).long()]
        test_imgs = torch.stack([test_ds[int(i)][0] for i in pick])
        titles = [f"{category} / {Path(paths[int(i)]).parent.name}" for i in pick]
        prediction_panel(test_imgs, masks[pick], maps[pick], predictions["synthcal"][pick], titles,
                         fig_dir / f"{category}_{run_name}.png")
        synthetic_panel(cal.artifacts["synthetic_images"], cal.artifacts["synthetic_masks"],
                        fig_dir / f"{category}_synthetic.png")

    if cfg.get("save_models", False):
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        model.save(out_dir / "models" / f"{category}_{run_name}.pt")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def run(cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg.get("output_dir", "results/main"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    csv = out_dir / "results.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    done = set(zip(df["category"], df["run"])) if len(df) else set()
    for run_cfg in cfg["runs"]:
        for cat in cfg["categories"]:
            if (cat, run_cfg["name"]) in done:
                continue
            t0 = time.perf_counter()
            row = run_one(cfg, cat, run_cfg, out_dir)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(csv, index=False)
            print(f"[{run_cfg['name']:>14s}] {cat:11s} img-AUROC={row['image_auroc']:.3f} "
                  f"AUPRO={row['aupro']:.3f} Dice quantile/synthcal/oracle="
                  f"{row['dice_quantile']:.3f}/{row['dice_synthcal']:.3f}/{row['dice_oracle']:.3f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["image_auroc", "pixel_auroc", "aupro", "image_f1_synthcal", "dice_quantile", "dice_synthabs",
            "dice_synthcal", "dice_oracle", "dice_oraclerel", "iou_synthcal", "latency_ms", "fit_seconds",
            "peak_gpu_mb"]
    return df.groupby("run")[[c for c in cols if c in df]].mean()
