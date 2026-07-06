# CPU-only image for the MVSS-Net tamper-detection API.
# No GPU/CUDA required — a single ResNet-50 forward pass runs fine on CPU.
FROM python:3.12-slim

# opencv-python-headless still needs libglib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# application code + model definitions
COPY detect.py app.py ./
COPY models/ ./models/

# pretrained weights, baked in (~1.2 GB). To keep the image slim instead,
# drop these two COPY lines and mount the dir at runtime:
#   -v /opt/mvss/ckpt:/app/ckpt/mvssnet_model
COPY ckpt/mvssnet_model/mvssnet_casia.pt   ckpt/mvssnet_model/mvssnet_casia.pt
COPY ckpt/mvssnet_model/mvssnet_defacto.pt ckpt/mvssnet_model/mvssnet_defacto.pt

ENV MVSS_DEVICE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
