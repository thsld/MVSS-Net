# Running MVSS-Net locally (macOS / Apple Silicon)

This is a local, CPU/MPS-only setup for **detecting image tampering** (splicing,
copy-move, removal/inpainting). No CUDA and no NVIDIA `apex` are required — those
are only needed for *training* in the original repo.

## What's installed

- **`uv`** (via Homebrew) manages an isolated Python 3.12 environment in `.venv/`
  (your system Python 3.14 is too new for PyTorch wheels).
- Modern PyTorch 2.12 + torchvision + opencv + numpy in `.venv/`.
- Pretrained checkpoints downloaded to `ckpt/mvssnet_model/`:
  | file | trained on | use |
  |------|-----------|-----|
  | `mvssnet_casia.pt` | CASIA | **best general-purpose detector (default)** |
  | `mvssnet_defacto.pt` | DEFACTO | inpainting/removal-heavy forgeries |
  | `resfcn_casia.pt` / `resfcn_defacto.pt` | — | lighter FCN baseline |

## Run it

```bash
# one photo
.venv/bin/python detect.py /path/to/photo.jpg

# several, or a whole folder (recurses)
.venv/bin/python detect.py photo1.jpg photo2.png ./album/

# options
.venv/bin/python detect.py photo.jpg \
    --model ckpt/mvssnet_model/mvssnet_defacto.pt \
    --thresh 0.5 \
    --out out \
    --device mps        # auto | cpu | mps
```

## Output

For each image it prints a line like:

```
  MANIPULATED  score=0.945  photo.jpg
  authentic    score=0.112  other.jpg
```

- **score** = the model's highest manipulation confidence over the image
  (max of the sigmoid of the segmentation map), in `[0, 1]`.
- **verdict** = `MANIPULATED` if `score >= --thresh` (default 0.5), else `authentic`.

And in `out/` per image:
- `<name>_mask.png` — grayscale heatmap; bright = predicted tampered pixels.
- `<name>_overlay.png` — that heatmap (red/JET) blended over the original, so you
  can see *where* the suspected manipulation is.

## How to read results (important caveats)

- MVSS-Net is a **research model**, not a courtroom tool. Treat the score as a
  signal, not proof. It can false-positive on heavy compression, screenshots,
  collages, strong filters, or synthetic/AI-generated images (it was trained to
  find *spliced/edited* regions, not to detect GAN/diffusion images).
- The **mask localization** is often more informative than the score: a coherent
  bright blob over a suspicious object is stronger evidence than the number alone.
- Try both `mvssnet_casia` and `mvssnet_defacto` — they disagree usefully.
- Best on reasonably high-resolution JPEGs. Re-saved/recompressed images are harder.

## Notes

- First run downloads ResNet-50 ImageNet weights (~98 MB) to
  `~/.cache/torch/`; cached afterward. Those base weights are overwritten by the
  checkpoint, so this is just an initialization quirk.
- `detect.py` is a from-scratch, device-agnostic runner. The repo's original
  `inference.py` is CUDA+apex only and is left untouched for reference.
