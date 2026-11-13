#!/usr/bin/env python3
"""Score raw DDPM samples against a held-out split.

    /tmp/glm-venv/bin/python scripts/ddpm_fidelity.py \
        --samples /tmp/opencode/run64/ddim50.raw \
        --held /tmp/opencode/flowers/held64.bin \
        --train /tmp/opencode/flowers/train64.bin \
        --size 64 --json /tmp/opencode/run64/metrics-ddim50.json

Two numbers:

- `fid`: pixel-space Frechet distance between the sample and held-out
  distributions. Full 64x64x3 covariance is 12,288 dims and needs more samples
  than we have, so both sides are average-pooled to 8x8x3 = 192 dims first.
  Lower is better; 0 means identical first and second moments.
- `nearest_l1`: mean L1 distance from each sample to its closest training image,
  in [-1, 1] pixel units. Low means memorised or blurry, high means the sample
  is unlike the data. It is a proxy, not a verdict.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_u8(path: Path, ceiling: int, legacy_dump: bool = False) -> np.ndarray:
    """Load an interleaved RGB file as NCHW, or in the pre-fix permuted order.

    `legacy_dump` reproduces the ordering of the 2026-11-13 64x64 run, whose
    loader copied interleaved bytes straight into a `[3, H, W]` tensor and whose
    sample writer undid that permutation on display. Its raw dumps are therefore
    in tensor space, and must be compared against references read the same way.
    """
    flat = np.fromfile(path, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
    stride = 3 * ceiling * ceiling
    if flat.size % stride:
        raise SystemExit(f"{path} is not a whole number of {ceiling}x{ceiling} images")
    if legacy_dump:
        return flat.reshape(-1, 3, ceiling, ceiling)
    return flat.reshape(-1, ceiling, ceiling, 3).transpose(0, 3, 1, 2)


def load_f32(path: Path) -> np.ndarray:
    shape = json.loads(Path(f"{path}.json").read_text())["shape"]
    flat = np.fromfile(path, dtype="<f4")
    return flat.reshape(shape).astype(np.float32)


def pool(images: np.ndarray, grid: int = 8) -> np.ndarray:
    """Average-pool NCHW to N,3,grid,grid and flatten to N,(3*grid*grid)."""
    batch, channels, height, width = images.shape
    if height % grid or width % grid:
        raise SystemExit(f"cannot pool {height}x{width} to {grid}x{grid}")
    folded = images.reshape(batch, channels, grid, height // grid, grid, width // grid)
    return folded.mean(axis=(3, 5)).reshape(batch, -1)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between two Gaussian fits: |mu_a-mu_b|^2 + Tr(Sa+Sb-2 sqrt(Sa Sb))."""
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    sa = np.cov(a, rowvar=False)
    sb = np.cov(b, rowvar=False)
    # Symmetric PSD matrix square root via eigendecomposition.
    product = sa @ sb
    roots = np.linalg.eigvals(product)
    trace_root = np.sqrt(np.clip(roots.real, 0.0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(sa) + np.trace(sb) - 2 * trace_root)


def nearest_l1(samples: np.ndarray, references: np.ndarray, keep: int = 32) -> float:
    """Mean over samples of the L1 distance to the closest reference image.

    Comparing every sample against all 3,303 training images at 64x64 would be
    5e9 element operations and a 400 MB intermediate per chunk. Instead the
    search runs in two stages: rank the references by L1 on the 8x8 pooled
    image (192 dims, 64x cheaper), then recompute exact full-resolution L1 for
    the best `keep` candidates. `check_prefilter` measures the error this
    introduces.
    """
    flat_samples = samples.reshape(len(samples), -1)
    flat_refs = references.reshape(len(references), -1)
    pooled_samples = pool(samples, 8)
    pooled_refs = pool(references, 8)

    best = np.empty(len(flat_samples), dtype=np.float32)
    for start in range(0, len(flat_samples), 16):
        block = pooled_samples[start : start + 16]
        coarse = np.abs(block[:, None, :] - pooled_refs[None, :, :]).sum(axis=2)
        candidates = np.argpartition(coarse, min(keep, len(pooled_refs) - 1), axis=1)[:, :keep]
        for row, refs in enumerate(candidates):
            exact = np.abs(flat_samples[start + row] - flat_refs[refs]).mean(axis=1)
            best[start + row] = exact.min()
    return float(best.mean())


def check_prefilter(samples: np.ndarray, references: np.ndarray, count: int = 8) -> float:
    """Largest nearest-distance error from the two-stage search, on `count` samples."""
    flat_samples = samples[:count].reshape(count, -1)
    flat_refs = references.reshape(len(references), -1)
    worst = 0.0
    for row in range(count):
        best = np.inf
        for start in range(0, len(flat_refs), 64):
            chunk = flat_refs[start : start + 64]
            best = min(best, np.abs(flat_samples[row] - chunk).mean(axis=1).min())
        approximate = nearest_l1(samples[row : row + 1], references)
        worst = max(worst, abs(best - approximate))
    return float(worst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--held", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--legacy-dump", action="store_true")
    args = parser.parse_args()

    samples = load_f32(args.samples)
    held = load_u8(args.held, args.size, args.legacy_dump)
    train = load_u8(args.train, args.size, args.legacy_dump)
    print(f"samples {samples.shape}, held {held.shape}, train {train.shape}")

    result = {
        "samples": int(len(samples)),
        "fid": frechet(pool(samples, args.grid), pool(held, args.grid)),
        "nearest_l1": nearest_l1(samples, train),
    }
    # A held-out image's own nearest-training distance, as the floor a real image
    # would score.
    result["heldout_nearest_l1"] = nearest_l1(held, train)
    result["prefilter_error"] = check_prefilter(samples, train)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
