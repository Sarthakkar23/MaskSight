"""Local Gradio app for trying the calibrated checkpoints.

Loads every `models/<category>.pt` produced by `masksight fit` and shows heatmap + predicted mask.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion

from masksight.calibrate import segment
from masksight.data import image_transform
from masksight.models import load_model
from masksight.viz import overlay

HERE = Path(__file__).parent
MODEL_DIR = HERE.parent / "models"
MODELS = {p.stem: load_model(str(p), device="cpu") for p in sorted(MODEL_DIR.glob("*.pt"))}
TF = image_transform(256, 224, normalize=False)
torch.set_num_threads(max(1, torch.get_num_threads()))


def run(image: Image.Image, category: str, sensitivity: float):
    if image is None:
        return None, None, "Upload an image first."
    model = MODELS[category]
    x = TF(image.convert("RGB"))
    t0 = time.perf_counter()
    maps, scores = model.predict(x[None])
    ms = 1000 * (time.perf_counter() - t0)
    amap, score = maps[0], float(scores[0])
    # sensitivity 1.0 = calibrated mask; >1 grows the mask, <1 shrinks it
    thr = model.pixel_threshold / sensitivity
    mask = segment(amap[None], thr, min(model.mask_alpha / sensitivity, 0.99))[0].numpy()
    rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    heat = overlay(x, amap, vmax=max(thr * 1.5, 1e-6))
    outlined = rgb.copy()
    outlined[mask & ~binary_erosion(mask, iterations=2)] = (255, 0, 0)
    verdict = "DEFECT" if score > model.image_threshold else "normal"
    info = (f"**{verdict}** - anomaly score {score:.2f} (threshold {model.image_threshold:.2f}) · "
            f"defect area {100 * mask.mean():.1f}% · {ms:.0f} ms on CPU")
    return heat, outlined, info


examples = sorted((HERE / "examples").glob("*.png"))
with gr.Blocks(title="MaskSight") as demo:
    gr.Markdown(
        "# MaskSight - label-free defect segmentation\n"
        "Trained on **normal images only**. The mask is calibrated on synthetic defects (SynthCal), "
        "so no defect labels were used anywhere. "
        "[Code](https://github.com/Sarthakkar23/MaskSight) · Data: MVTec AD (CC BY-NC-SA 4.0)")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="input")
            cat = gr.Dropdown(sorted(MODELS), value=sorted(MODELS)[0] if MODELS else None, label="product type")
            sens = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="sensitivity")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            heat = gr.Image(label="anomaly heatmap")
            out = gr.Image(label="predicted defect mask")
            info = gr.Markdown()
    btn.click(run, [inp, cat, sens], [heat, out, info])
    if examples:
        gr.Examples([[str(p), p.stem.split("__")[0], 1.0] for p in examples], [inp, cat, sens],
                    [heat, out, info], fn=run, cache_examples=False)

if __name__ == "__main__":
    demo.launch()
