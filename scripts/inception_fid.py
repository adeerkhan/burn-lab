#!/usr/bin/env python3
"""Inception FID, for calibration against the literature.

Our headline numbers are pixel-space Frechet distances at 8x8 pooling; they are
not comparable with Inception FID in papers. This computes the real thing for
one model, so the relation between the two scales is measured rather than
assumed.

The reference distribution is the **training** split, not held out: Inception
features are 2048-dimensional, so both sides need more than 2048 samples for a
full-rank covariance, and the held-out split has 367.

    /tmp/glm-venv/bin/python scripts/inception_fid.py \
        --samples /tmp/flow/ddim20.raw --reference /tmp/flowers/train64.bin \
        --size 64 --json /tmp/flow/inception.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3


def load_raw(path: Path) -> np.ndarray:
    meta = json.loads(Path(f"{path}.json").read_text())
    _, channels, height, width = meta["shape"]
    flat = np.fromfile(path, dtype="<f4")
    # A chunked dump can lose a chunk to a teardown crash; trust the file size.
    count = len(flat) // (channels * height * width)
    return flat[: count * channels * height * width].reshape(count, channels, height, width)


def load_u8(path: Path, size: int) -> np.ndarray:
    flat = np.fromfile(path, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
    return flat.reshape(-1, size, size, 3).transpose(0, 3, 1, 2)


def features(images: np.ndarray) -> np.ndarray:
    """2048-d pool3 features. `images` is NCHW in [-1, 1]."""
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    tensors = torch.from_numpy(images)
    # FID resizes to 299 with bicubic filtering and ImageNet normalisation.
    tensors = torch.nn.functional.interpolate(
        tensors, size=(299, 299), mode="bicubic", align_corners=False, antialias=True
    )
    tensors = (tensors + 1.0) * 0.5
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensors = (tensors - mean) / std

    out = []
    with torch.no_grad():
        for start in range(0, len(tensors), 32):
            out.append(model(tensors[start : start + 32]).numpy())
    return np.concatenate(out)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, mu_b = a.mean(0), b.mean(0)
    sa, sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    roots = np.linalg.eigvals(sa @ sb).real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(sa) + np.trace(sb) - 2 * np.sqrt(np.clip(roots, 0, None)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--project", type=int, default=0, help="PCA dimensions, 0 = full 2048")
    parser.add_argument("--floor-held", type=Path, help="also score this real split as the floor")
    args = parser.parse_args()

    samples = load_raw(args.samples)
    reference = load_u8(args.reference, args.size)
    # The covariance needs more samples than dimensions; with a projection the
    # requirement drops to the projected width.
    dimensions = args.project or 2048
    if len(samples) <= dimensions:
        raise SystemExit(
            f"need more than {dimensions} samples for full rank, got {len(samples)}"
        )
    print(f"samples {samples.shape}, reference {reference.shape}")

    fa = features(samples)
    fb = features(reference)

    if args.project:
        # Inception features are 2048-d; with only 3,303 real images no floor is
        # computable at that width. Fit PCA on the reference and project both
        # sides, so a real-vs-real floor exists at the same width.
        mean = fb.mean(0)
        centred = fb - mean
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[: args.project]
        fa = (fa - mean) @ components.T
        fb = centred @ components.T

    value = frechet(fa, fb)
    result = {
        "inception_fid": value,
        "samples": len(samples),
        "reference": len(reference),
        "dimensions": fa.shape[1],
    }
    if args.floor_held:
        held = features(load_u8(args.floor_held, args.size))
        if args.project:
            held = (held - mean) @ components.T
        result["floor_real_vs_real"] = frechet(held, fb)
        result["floor_samples"] = int(len(held))
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
