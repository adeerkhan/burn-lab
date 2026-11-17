#!/usr/bin/env python3
"""LPIPS with pretrained VGG16, for judging perceptual quality.

    /home/adeerkhan/tmp/venv/bin/python scripts/lpips_score.py \
        --samples model.raw --reference /path/train64.bin --size 64

Uses torchvision's VGG16 up to relu4_3 (the standard LPIPS feature layer), with
ImageNet normalisation and the model's own weights. No learned LPIPS calibration
network: the straight feature-space L1 divided by channel std is the diagnostic,
not a paper-comparable LPIPS number.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.models.vgg import VGG16_Weights

WEIGHTS = Path("/home/adeerkhan/tmp/vgg16.pth")


def load_raw(path: Path) -> np.ndarray:
    meta = json.loads(Path(f"{path}.json").read_text())
    _, channels, height, width = meta["shape"]
    flat = np.fromfile(path, dtype="<f4")
    return flat[: len(flat) // (channels * height * width) * channels * height * width].reshape(
        -1, channels, height, width
    )


def load_u8(path: Path, size: int) -> np.ndarray:
    flat = np.fromfile(path, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
    return flat.reshape(-1, size, size, 3).transpose(0, 3, 1, 2)


def features(images: torch.Tensor, model: nn.Module) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            out.append(model(images[start : start + 16]))
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    slices = list(backbone.features)[:23]  # through relu4_3
    model = nn.Sequential(*slices).eval()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

    samples = torch.from_numpy(load_raw(args.samples).astype(np.float32))
    reference = torch.from_numpy(load_u8(args.reference, args.size).astype(np.float32))
    samples = ((samples + 1.0) * 0.5 - mean) / std
    reference = ((reference + 1.0) * 0.5 - mean) / std

    fs = features(samples, model)
    fr = features(reference, model)
    # Per-feature standard deviation over the reference, as in LPIPS.
    scale = fr.flatten(1).std(dim=0).reshape(1, -1, 1, 1).clamp_min(1e-4)
    a = fs.flatten(1) / scale.flatten(1)
    b = fr.flatten(1) / scale.flatten(1)
    # Mean nearest-reference distance, as LPIPS does; comparing corresponding
    # rows of two different-size sets would be meaningless.
    distances = []
    for start in range(0, len(a), 32):
        block = a[start : start + 32]
        distances.append(
            (block[:, None, :] - b[None, :, :]).abs().mean(dim=2).min(dim=1)[0]
        )
    score = torch.cat(distances).mean().item()

    result = {"lpips_vgg_feature_l1": score, "samples": len(fs), "reference": len(fr)}
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
