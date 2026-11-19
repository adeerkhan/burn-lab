#!/usr/bin/env python3
"""Composite exact cars and sunset from the real frame onto a pixel dream.

    python3 scripts/vice_hybrid.py --frames /tmp/vice2/dream \
        --real /tmp/vice2/data --out docs/vice-hybrid.mp4

The world model handles the city and lighting; the car sprites come from the
known renderer at their exact positions, so they never overlap or smear.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def blobs(mask: np.ndarray, minimum: int = 6):
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            points = []
            while stack:
                px, py = stack.pop()
                points.append((px, py))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(points) >= minimum:
                out.append((min(p[0] for p in points), min(p[1] for p in points),
                            max(p[0] for p in points), max(p[1] for p in points)))
    return out


def compose(real: np.ndarray, dream: np.ndarray) -> np.ndarray:
    out = dream.copy()
    # Sunset and skyline are structural: take them exactly.
    out[:14] = real[:14]
    r, g, b = real[..., 0].astype(int), real[..., 1].astype(int), real[..., 2].astype(int)
    cars = ((r > 180) & (g < 110) & (b < 100)) | ((b > 140) & (r < 110) & (g < 130))
    for x0, y0, x1, y1 in blobs(cars):
        pad = 2
        ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
        ax1, ay1 = min(64, x1 + pad + 1), min(64, y1 + pad + 1)
        if ax1 - ax0 < 2 or ay1 - ay0 < 2:
            continue
        # Clear the dream's version of this car to a local road colour, then
        # paste the real patch so its shape and lights stay exact.
        region = out[ay0:ay1, ax0:ax1].astype(float)
        median = np.median(region.reshape(-1, 3), axis=0)
        out[ay0:ay1, ax0:ax1] = np.clip(median, 0, 255).astype(np.uint8)
        out[ay0:ay1, ax0:ax1] = real[ay0:ay1, ax0:ax1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = np.memmap(args.real / "frames.bin", dtype=np.uint8, mode="r").reshape(-1, 64, 64, 3)
    work = args.out.parent / "vice-hybrid-frames"
    work.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(sorted(args.frames.glob("step-*.png"))):
        strip = np.asarray(Image.open(path).convert("RGB"))
        real, dream = strip[:, :64], strip[:, 64:]
        Image.fromarray(np.concatenate([real, compose(real, dream)], axis=1)).save(
            work / f"f{index:04d}.png")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
        "-i", str(work / "f%04d.png"), "-vf", "scale=768:-1:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-crf", "20", str(args.out),
    ], check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
