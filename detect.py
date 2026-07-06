#!/usr/bin/env python
"""
Single-image (or folder) tamper detection with MVSS-Net.

Runs on CPU or Apple-Silicon MPS — no CUDA, no apex required.
Faithfully reproduces the original repo's preprocessing (BGR cv2 image,
resize 512, ImageNet normalization) and scoring (max of sigmoid over the
segmentation map).

Usage:
    python detect.py path/to/photo.jpg
    python detect.py img1.jpg img2.png --out out --thresh 0.5
    python detect.py some_folder/ --model ckpt/mvssnet_model/mvssnet_defacto.pt
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

from models.mvssnet import get_mvss
from models.resfcn import ResFCN

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_path):
    name = os.path.basename(model_path).lower()
    if "mvssnet" in name:
        # pretrained_base=False: we load the full checkpoint with strict=True
        # right after, so there's no need to fetch ImageNet ResNet weights.
        model = get_mvss(backbone="resnet50", pretrained_base=False,
                         nclass=1, sobel=True, constrain=True, n_input=3)
    elif "fcn" in name:
        model = ResFCN()
    else:
        sys.exit("Model type not recognised from filename: %s "
                 "(expected 'mvssnet' or 'fcn')" % model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def preprocess(img_bgr, size):
    img = cv2.resize(img_bgr, (size, size))
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return x


def overlay(img_bgr, mask_gray):
    """Red heatmap of the predicted tampered region on top of the photo."""
    heat = cv2.applyColorMap(mask_gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0)


def gather_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in IMG_EXTS:
                out += glob.glob(os.path.join(p, "**", "*" + ext), recursive=True)
        else:
            out.append(p)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="MVSS-Net tamper detection")
    ap.add_argument("images", nargs="+", help="image file(s) or folder(s)")
    ap.add_argument("--model", default="ckpt/mvssnet_model/mvssnet_casia.pt")
    ap.add_argument("--out", default="out", help="dir for masks + overlays")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="score above this => flagged as manipulated")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit("Checkpoint not found: %s" % args.model)

    device = pick_device(args.device)
    model = build_model(args.model)
    try:
        model.to(device)
    except Exception as e:
        print("Could not use %s (%s); falling back to CPU." % (device, e))
        device = torch.device("cpu")
        model.to(device)

    files = gather_images(args.images)
    if not files:
        sys.exit("No images found.")

    os.makedirs(args.out, exist_ok=True)
    print("model:  %s" % os.path.basename(args.model))
    print("device: %s" % device.type)
    print("thresh: %.2f\n" % args.thresh)

    results = []
    with torch.no_grad():
        for path in files:
            img = cv2.imread(path)
            if img is None:
                print("  [skip] cannot read %s" % path)
                continue
            h, w = img.shape[:2]
            x = preprocess(img, args.resize).to(device)
            _, seg = model(x)
            seg = torch.sigmoid(seg).squeeze().float().cpu().numpy()  # HxW in [0,1]
            score = 0.0 if (np.isnan(seg).any() or np.isinf(seg).any()) else float(seg.max())

            mask = (seg * 255).astype(np.uint8)
            mask = cv2.resize(mask, (w, h))
            stem = os.path.splitext(os.path.basename(path))[0]
            mask_path = os.path.join(args.out, stem + "_mask.png")
            over_path = os.path.join(args.out, stem + "_overlay.png")
            cv2.imwrite(mask_path, mask)
            cv2.imwrite(over_path, overlay(img, mask))

            verdict = "MANIPULATED" if score >= args.thresh else "authentic"
            results.append((path, score, verdict))
            print("  %-12s score=%.3f  %s" % (verdict, score, path))

    print("\nMasks + overlays written to: %s/" % args.out)
    flagged = [r for r in results if r[1] >= args.thresh]
    print("Flagged %d / %d image(s) as manipulated." % (len(flagged), len(results)))


if __name__ == "__main__":
    main()
