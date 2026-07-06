"""Unit + endpoint tests for the tamper-detection API (app.py).

Runs without the real 1.2 GB checkpoints: the model layer is mocked, so these
are fast and CI-friendly. Uses CPU only.
"""
import asyncio

import cv2
import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app as appmod


# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #
class _DummyModel:
    """Stands in for MVSS-Net: returns a fixed segmentation map."""

    def __init__(self, fill=0.0):
        self.fill = fill

    def to(self, _device):
        return self

    def __call__(self, x):
        b = x.shape[0]
        seg = torch.full((b, 1, appmod.RESIZE, appmod.RESIZE), self.fill)
        return None, seg  # (edge, seg) — app only uses seg


class _FakeUpload:
    """Minimal UploadFile stand-in supporting chunked .read(n)."""

    def __init__(self, data: bytes):
        self._b = data
        self._i = 0

    async def read(self, n: int = -1) -> bytes:
        chunk = self._b[self._i : self._i + n] if n and n > 0 else self._b[self._i :]
        self._i += len(chunk)
        return chunk


def _png_bytes(h=8, w=8) -> bytes:
    """A small, genuinely-decodable PNG."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(monkeypatch):
    """TestClient with the model load mocked out (no checkpoints needed)."""
    monkeypatch.setattr(appmod, "API_KEY", "testsecret")
    monkeypatch.setattr(appmod, "build_model", lambda _path: _DummyModel(fill=0.0))
    # startup checks os.path.exists on the checkpoint paths — always pass
    monkeypatch.setattr(appmod.os.path, "exists", lambda _p: True)
    with TestClient(appmod.app) as c:
        yield c


# --------------------------------------------------------------------------- #
# require_api_key
# --------------------------------------------------------------------------- #
class TestRequireApiKey:
    def test_valid_key_passes(self, monkeypatch):
        monkeypatch.setattr(appmod, "API_KEY", "secret")
        appmod.require_api_key("secret")  # no raise

    @pytest.mark.parametrize("supplied", ["wrong", "", None])
    def test_bad_or_missing_key_rejected(self, monkeypatch, supplied):
        monkeypatch.setattr(appmod, "API_KEY", "secret")
        with pytest.raises(HTTPException) as exc:
            appmod.require_api_key(supplied)
        assert exc.value.status_code == 401

    def test_auth_disabled_when_key_unset(self, monkeypatch):
        monkeypatch.setattr(appmod, "API_KEY", None)
        appmod.require_api_key(None)  # no-op, no raise
        appmod.require_api_key("anything")


# --------------------------------------------------------------------------- #
# _read_capped
# --------------------------------------------------------------------------- #
class TestReadCapped:
    def test_under_limit_returned_whole(self, monkeypatch):
        monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 4 * 1024 * 1024)
        data = b"\x00" * (1024 * 1024)
        out = asyncio.run(appmod._read_capped(_FakeUpload(data)))
        assert out == data

    def test_exact_limit_ok(self, monkeypatch):
        monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 1000)
        out = asyncio.run(appmod._read_capped(_FakeUpload(b"x" * 1000)))
        assert len(out) == 1000

    def test_over_limit_rejected_413(self, monkeypatch):
        monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 1000)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(appmod._read_capped(_FakeUpload(b"x" * 1001)))
        assert exc.value.status_code == 413


# --------------------------------------------------------------------------- #
# _score
# --------------------------------------------------------------------------- #
class TestScore:
    def test_sigmoid_max_of_zero_seg(self, monkeypatch):
        # seg all zeros -> sigmoid 0.5 -> max 0.5
        monkeypatch.setattr(appmod, "_models", {"casia": _DummyModel(fill=0.0)})
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        assert appmod._score(img) == {"casia": 0.5}

    def test_high_logit_scores_near_one(self, monkeypatch):
        monkeypatch.setattr(appmod, "_models", {"casia": _DummyModel(fill=20.0)})
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        assert appmod._score(img)["casia"] == pytest.approx(1.0, abs=1e-3)

    def test_nan_guarded_to_zero(self, monkeypatch):
        monkeypatch.setattr(appmod, "_models", {"casia": _DummyModel(fill=float("nan"))})
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        assert appmod._score(img) == {"casia": 0.0}

    def test_multiple_models_all_scored(self, monkeypatch):
        monkeypatch.setattr(
            appmod,
            "_models",
            {"casia": _DummyModel(fill=0.0), "defacto": _DummyModel(fill=20.0)},
        )
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        out = appmod._score(img)
        assert set(out) == {"casia", "defacto"}
        assert out["casia"] == 0.5
        assert out["defacto"] == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
class TestEndpoints:
    def test_health_open_and_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_detect_requires_key(self, client):
        r = client.post("/detect", files={"file": ("x.png", _png_bytes(), "image/png")})
        assert r.status_code == 401

    def test_detect_rejects_wrong_key(self, client):
        r = client.post(
            "/detect",
            headers={"X-API-Key": "nope"},
            files={"file": ("x.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 401

    def test_detect_valid_key_and_image(self, client):
        r = client.post(
            "/detect",
            headers={"X-API-Key": "testsecret"},
            files={"file": ("x.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        # DummyModel(fill=0) -> score 0.5 == default threshold -> manipulated True
        assert body["score"] == 0.5
        assert body["threshold"] == 0.5
        assert body["manipulated"] is True
        assert set(body["models"]) == {"casia", "defacto"}

    def test_detect_threshold_query_respected(self, client):
        r = client.post(
            "/detect?threshold=0.9",
            headers={"X-API-Key": "testsecret"},
            files={"file": ("x.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["manipulated"] is False  # 0.5 < 0.9

    def test_detect_bad_image_400(self, client):
        r = client.post(
            "/detect",
            headers={"X-API-Key": "testsecret"},
            files={"file": ("x.png", b"not-an-image", "image/png")},
        )
        assert r.status_code == 400

    def test_detect_oversized_413(self, client, monkeypatch):
        monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 1024)
        r = client.post(
            "/detect",
            headers={"X-API-Key": "testsecret"},
            files={"file": ("big.png", b"\x00" * 2048, "image/png")},
        )
        assert r.status_code == 413
