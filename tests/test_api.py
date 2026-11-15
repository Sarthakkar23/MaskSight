"""API tests with a stand-in model, so they run in CI without checkpoints or backbone weights."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient  # noqa: E402

from serve.api import create_app  # noqa: E402


class FakeModel:
    """Flags the bright top-left quadrant of any image as a defect."""

    name = "patchcore"
    image_threshold = 0.5

    def predict(self, images):
        maps = images.mean(1)
        return maps, maps.flatten(1).max(1).values

    def segment(self, maps):
        return maps > 0.5


def png(bright: bool) -> bytes:
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    if bright:
        arr[:150, :150] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    app = create_app(models={"bottle": FakeModel()}, storage=tmp_path, database_url=f"sqlite:///{tmp_path}/t.db")
    with TestClient(app) as c:
        yield c


def test_categories_and_cors(client):
    r = client.get("/categories", headers={"Origin": "http://localhost:8081"})
    assert r.status_code == 200
    assert r.json() == [{"name": "bottle", "model": "patchcore", "image_threshold": 0.5}]
    assert r.headers["access-control-allow-origin"] == "*"


def test_predict_saves_history(client):
    r = client.post("/predict?category=bottle", files={"image": ("a.png", png(True), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_defective"] and body["id"] == 1 and body["filename"] == "a.png"
    layer = Image.open(io.BytesIO(base64.b64decode(body["mask_layer_png_base64"])))
    image = Image.open(io.BytesIO(base64.b64decode(body["image_png_base64"])))
    assert layer.mode == "RGBA" and layer.size == image.size == (224, 224)
    assert 0.1 < body["defect_area_fraction"] < 0.5

    client.post("/predict?category=bottle", files={"image": ("b.png", png(False), "image/png")})
    rows = client.get("/inspections").json()
    assert [row["id"] for row in rows] == [2, 1]
    assert rows[0]["is_defective"] is False and "image_png_base64" not in rows[0]
    again = client.get("/inspections/1").json()
    assert again["mask_png_base64"] == body["mask_png_base64"]


def test_errors(client):
    assert client.post("/predict?category=nope", files={"image": ("a.png", png(True), "image/png")}).status_code == 404
    assert client.post("/predict?category=bottle", files={"image": ("a.txt", b"hi", "text/plain")}).status_code == 400
    assert client.get("/inspections/99").status_code == 404
    assert client.get("/inspections").json() == []

