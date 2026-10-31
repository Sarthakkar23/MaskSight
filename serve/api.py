"""REST inference service.

    MASKSIGHT_MODELS=models uvicorn serve.api:app --port 8000
    curl -F image=@part.png "localhost:8000/predict?category=bottle"

Every `<category>.pt` checkpoint (from `masksight fit`) in MASKSIGHT_MODELS is loaded once at startup.
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from masksight.data import image_transform
from masksight.models import load_model

MODEL_DIR = Path(os.environ.get("MASKSIGHT_MODELS", "models"))
MODELS = {p.stem: load_model(str(p)) for p in sorted(MODEL_DIR.glob("*.pt"))}
TF = image_transform(256, 224, normalize=False)

app = FastAPI(title="MaskSight", description="Label-free defect segmentation API")


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "categories": sorted(MODELS)}


@app.post("/predict")
async def predict(category: str, image: UploadFile = File(...)):
    model = MODELS.get(category)
    if model is None:
        raise HTTPException(404, f"no model for '{category}', available: {sorted(MODELS)}")
    img = TF(Image.open(io.BytesIO(await image.read())).convert("RGB"))
    t0 = time.perf_counter()
    maps, scores = model.predict(img[None])
    ms = 1000 * (time.perf_counter() - t0)
    mask = model.segment(maps)[0].numpy()
    score = float(scores[0])
    return {
        "category": category,
        "anomaly_score": score,
        "is_defective": bool(score > model.image_threshold),
        "defect_area_fraction": float(mask.mean()),
        "mask_png_base64": _png_b64((mask * 255).astype(np.uint8)),
        "latency_ms": round(ms, 1),
    }
