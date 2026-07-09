#!/usr/bin/env python
"""
HTTP endpoint for image tamper detection with MVSS-Net.

Loads the checkpoints once at startup, then answers per-image requests.

Run:
    MVSS_API_KEY=your-secret .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000

Env:
    MVSS_API_KEY        require this value in the X-API-Key header (unset = open, dev only)
    MVSS_MAX_UPLOAD_MB  reject uploads larger than this (default 15)
    MVSS_THRESHOLD      default decision threshold (default 0.5)
    MVSS_DEVICE         auto | cpu | mps (default auto)

Use:
    curl -s -H "X-API-Key: your-secret" -F "file=@/path/to/photo.jpg" \
        http://localhost:8000/detect | jq
    curl -s -H "X-API-Key: your-secret" -F "file=@photo.jpg" \
        "http://localhost:8000/detect?threshold=0.4" | jq

Response:
    {
      "manipulated": true,
      "score": 0.81,
      "threshold": 0.5,
      "degraded": false,        # true => a model output NaN/Inf; result unreliable
      "models": {"casia": 0.81, "defacto": 0.97}
    }
"""
import hmac
import io
import logging
import os

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile

from detect import MEAN, STD, build_model, pick_device

# --- config (override via env) ---
DEVICE = pick_device(os.getenv("MVSS_DEVICE", "auto"))
RESIZE = int(os.getenv("MVSS_RESIZE", "512"))
DEFAULT_THRESH = float(os.getenv("MVSS_THRESHOLD", "0.5"))
# Shared-secret guard. If unset, auth is DISABLED (dev only — warned at startup).
API_KEY = os.getenv("MVSS_API_KEY")
MAX_UPLOAD_MB = float(os.getenv("MVSS_MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
CKPTS = {
    "casia": os.getenv("MVSS_CASIA", "ckpt/mvssnet_model/mvssnet_casia.pt"),
    "defacto": os.getenv("MVSS_DEFACTO", "ckpt/mvssnet_model/mvssnet_defacto.pt"),
}

app = FastAPI(title="MVSS-Net tamper detection")
_models = {}
logger = logging.getLogger("mvss")


def require_api_key(x_api_key: str = Header(None)):
    """Reject requests without a valid X-API-Key. No-op if MVSS_API_KEY unset."""
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def _read_capped(file: UploadFile) -> bytes:
    """Read the upload, aborting if it exceeds MAX_UPLOAD_BYTES (avoids OOM)."""
    chunks, total = [], 0
    while True:
        chunk = await file.read(1 << 20)  # 1 MB at a time
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File too large (max %.0f MB)" % MAX_UPLOAD_MB,
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.on_event("startup")
def _load():
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            raise RuntimeError("Checkpoint missing: %s" % path)
        m = build_model(path)
        m.to(DEVICE)
        _models[name] = m
    print("Loaded models %s on %s" % (list(_models), DEVICE.type))
    if not API_KEY:
        print("WARNING: MVSS_API_KEY is not set — /detect is UNAUTHENTICATED. "
              "Set it before exposing this service publicly.")


def _score(img_bgr):
    """Return (scores_by_model, degraded_models).

    `degraded_models` lists any model whose output was numerically corrupt
    (NaN/Inf); its score is forced to 0.0 and must NOT be trusted.
    """
    img = cv2.resize(img_bgr, (RESIZE, RESIZE))
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    out, degraded = {}, []
    with torch.no_grad():
        for name, model in _models.items():
            _, seg = model(x)
            seg = torch.sigmoid(seg)
            if torch.isnan(seg).any() or torch.isinf(seg).any():
                # Numerical corruption: don't silently report "authentic".
                logger.warning("NaN/Inf in %s model output — detector failed; "
                               "score forced to 0.0 and is NOT reliable", name)
                out[name] = 0.0
                degraded.append(name)
            else:
                out[name] = round(float(seg.max()), 4)
    return out, degraded


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE.type, "models": list(_models)}


@app.post("/detect", dependencies=[Depends(require_api_key)])
async def detect(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESH, ge=0.0, le=1.0),
):
    raw = await _read_capped(file)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    scores, degraded = _score(img)
    decisive = max(scores.values())  # flag if EITHER model crosses the line
    return {
        "manipulated": bool(decisive >= threshold),
        "score": decisive,
        "threshold": threshold,
        "degraded": bool(degraded),  # true => a model failed; result unreliable
        "models": scores,
    }
