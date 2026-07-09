"""Modal deployment of the tamper-detection API on a GPU.

Reuses the existing FastAPI app (app.py) unchanged — Modal just runs it on a
T4 with the checkpoints mounted from a Volume. Auth (X-API-Key) is enforced by
app.py exactly as on the box.

Setup (one-time):
    modal volume create mvss-weights
    modal volume put mvss-weights ckpt/mvssnet_model/mvssnet_casia.pt   /mvssnet_casia.pt
    modal volume put mvss-weights ckpt/mvssnet_model/mvssnet_defacto.pt /mvssnet_defacto.pt
    modal secret create mvss-secrets MVSS_API_KEY=<key>

Deploy:
    modal deploy modal_app.py    # prints the public https URL
"""
import modal

app = modal.App("id-tamper")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0")
    .pip_install(
        "torch==2.12.1",
        "torchvision==0.27.1",
        "numpy==2.4.6",
        "opencv-python-headless>=4.13,<4.14",
        "fastapi==0.137.2",
        "uvicorn[standard]==0.49.0",
        "python-multipart>=0.0.9",
    )
    # our source (mounted into the container at /app)
    .add_local_file("detect.py", "/app/detect.py")
    .add_local_file("app.py", "/app/app.py")
    .add_local_dir("models", "/app/models")
)

weights = modal.Volume.from_name("mvss-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",                       # plenty for ResNet-50; ~$0.60/hr while warm
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("mvss-secrets")],  # -> MVSS_API_KEY
    min_containers=0,               # scale to zero (trial); set 1 to keep warm
    scaledown_window=120,
)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    # point app.py at the mounted checkpoints + GPU
    os.environ["MVSS_CASIA"] = "/weights/mvssnet_casia.pt"
    os.environ["MVSS_DEFACTO"] = "/weights/mvssnet_defacto.pt"
    os.environ.setdefault("MVSS_DEVICE", "auto")  # -> cuda on the GPU

    sys.path.insert(0, "/app")
    from app import app as fastapi  # loads models on startup (onto CUDA)
    return fastapi
