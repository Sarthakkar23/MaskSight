"""REST inference service with inspection history.

    MASKSIGHT_MODELS=models uvicorn serve.api:app --port 8000
    curl -F image=@part.png "localhost:8000/predict?category=bottle"

Environment:
  MASKSIGHT_MODELS      directory of `<category>.pt` checkpoints (from `masksight fit`), loaded at startup
  MASKSIGHT_CATEGORIES  optional comma list to load only some of them, e.g. "bottle,hazelnut"
  MASKSIGHT_STORAGE     where uploaded images and masks are kept (default: storage)
  DATABASE_URL          SQLAlchemy URL (default: SQLite inside MASKSIGHT_STORAGE); PostgreSQL works too
  CORS_ORIGINS          comma list of allowed web origins (default: *)
  MAX_UPLOAD_MB         largest accepted upload (default: 10)

Endpoints: GET /health, GET /categories, POST /predict, GET /inspections, GET /inspections/{id}.
"""

from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from serve.db import Inspection, list_inspections, make_engine, make_sessionmaker

OVERLAY_RGB = (239, 68, 68)  # mask colour sent to clients (red)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _mask_layer(mask_png: bytes) -> bytes:
    """Transparent RGBA layer: coloured where the mask is set, see-through elsewhere."""
    mask = np.asarray(Image.open(io.BytesIO(mask_png)).convert("L")) > 127
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*OVERLAY_RGB, 255)
    return _png_bytes(Image.fromarray(rgba, "RGBA"))


def load_models(model_dir: Path, only: list[str] | None = None) -> dict:
    from masksight.models import load_model

    paths = sorted(model_dir.glob("*.pt"))
    if only:
        paths = [p for p in paths if p.stem in only]
    return {p.stem: load_model(str(p), device="cpu") for p in paths}


def create_app(models: dict | None = None, storage: str | Path | None = None, database_url: str | None = None,
               cors_origins: list[str] | None = None, max_upload_mb: float | None = None) -> FastAPI:
    """Build the API. Arguments left as None are read from the environment (see module docstring)."""
    storage = Path(storage or os.environ.get("MASKSIGHT_STORAGE", "storage"))
    database_url = database_url or os.environ.get("DATABASE_URL") or f"sqlite:///{storage / 'masksight.db'}"
    cors_origins = cors_origins or [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    max_bytes = int((max_upload_mb or float(os.environ.get("MAX_UPLOAD_MB", 10))) * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if models is None:
            only = [c.strip() for c in os.environ.get("MASKSIGHT_CATEGORIES", "").split(",") if c.strip()]
            app.state.models = load_models(Path(os.environ.get("MASKSIGHT_MODELS", "models")), only or None)
        else:
            app.state.models = models
        (storage / "inspections").mkdir(parents=True, exist_ok=True)
        engine = make_engine(database_url)
        app.state.sessions = make_sessionmaker(engine)
        yield
        engine.dispose()

    app = FastAPI(title="MaskSight", description="Label-free defect segmentation API", version="0.2.0",
                  lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    from masksight.data import image_transform

    resize_crop = image_transform(256, 224, normalize=False)

    def detail(row: Inspection) -> dict:
        image_png = (storage / row.image_path).read_bytes()
        mask_png = (storage / row.mask_path).read_bytes()
        return {**row.summary(), "image_png_base64": _b64(image_png), "mask_png_base64": _b64(mask_png),
                "mask_layer_png_base64": _b64(_mask_layer(mask_png))}

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "categories": sorted(request.app.state.models)}

    @app.get("/categories")
    def categories(request: Request):
        return [{"name": name, "model": m.name, "image_threshold": float(m.image_threshold)}
                for name, m in sorted(request.app.state.models.items())]

    @app.post("/predict")
    async def predict(request: Request, category: str, image: UploadFile = File(...)):
        model = request.app.state.models.get(category)
        if model is None:
            raise HTTPException(404, f"no model for '{category}', available: {sorted(request.app.state.models)}")
        raw = await image.read()
        if len(raw) > max_bytes:
            raise HTTPException(413, f"image is larger than {max_bytes // (1024 * 1024)} MB")
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError) as err:
            raise HTTPException(400, "could not read the upload as an image") from err

        x = resize_crop(pil)  # 3 x 224 x 224 in [0, 1]: exactly what the model sees
        t0 = time.perf_counter()
        maps, scores = model.predict(x[None])
        ms = 1000 * (time.perf_counter() - t0)
        mask = model.segment(maps)[0].numpy()
        score = float(scores[0])

        image_png = _png_bytes(Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)))
        mask_png = _png_bytes(Image.fromarray((mask * 255).astype(np.uint8)))
        with request.app.state.sessions() as session:
            row = Inspection(category=category, anomaly_score=score, threshold=float(model.image_threshold),
                             is_defective=bool(score > model.image_threshold),
                             defect_area_fraction=float(mask.mean()), latency_ms=round(ms, 1),
                             filename=(image.filename or None) and image.filename[:255])
            session.add(row)
            session.flush()  # assigns row.id
            row.image_path = f"inspections/{row.id}_image.png"
            row.mask_path = f"inspections/{row.id}_mask.png"
            (storage / row.image_path).write_bytes(image_png)
            (storage / row.mask_path).write_bytes(mask_png)
            session.commit()
            return detail(row)

    @app.get("/inspections")
    def inspections(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                    category: str | None = None):
        with request.app.state.sessions() as session:
            return [r.summary() for r in list_inspections(session, limit, offset, category)]

    @app.get("/inspections/{inspection_id}")
    def inspection(request: Request, inspection_id: int):
        with request.app.state.sessions() as session:
            row = session.get(Inspection, inspection_id)
            if row is None:
                raise HTTPException(404, f"no inspection with id {inspection_id}")
            return detail(row)

    return app


app = create_app()
