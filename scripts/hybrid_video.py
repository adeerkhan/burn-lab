#!/usr/bin/env python3
"""Hybrid dream: pixel-model cave plus exact entity sprites.

    python3 scripts/hybrid_video.py --frames /tmp/pixel/dream5 \
        --out docs/hybrid-dream.mp4

The pixel model still predicts the cave, torch and lighting; the player,
enemies, diamonds and exit are taken from the observed frame and redrawn at
their true positions. This is the state + pixel hybrid: learned appearance for
the world, exact state for the things you interact with.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def blobs(mask: np.ndarray, minimum: int = 3) -> list[tuple[float, float]]:
    """Connected components of a boolean mask, as centroids."""
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
                out.append((sum(p[0] for p in points) / len(points),
                            sum(p[1] for p in points) / len(points)))
    return out


def entity_pixels(frame: np.ndarray):
    r, g, b = frame[..., 0].astype(int), frame[..., 1].astype(int), frame[..., 2].astype(int)
    return {
        "player": (r > 230) & (g > 190) & (b < 140),
        "enemy": (r > 230) & (g < 110) & (b < 120),
        "diamond": (b > 220) & (g > 190) & (r < 130),
        "exit": (g > 230) & (r < 130) & (b < 180),
    }


def erase(image: np.ndarray, mask: np.ndarray) -> None:
    """Replace entity pixels with the median of their non-entity neighbours."""
    if not mask.any():
        return
    height, width = mask.shape
    source = image.copy()
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            neighbourhood = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not mask[ny, nx]:
                        neighbourhood.append(source[ny, nx])
            if neighbourhood:
                image[y, x] = np.median(np.stack(neighbourhood), axis=0)


def draw_entities(frame: np.ndarray, real: np.ndarray) -> np.ndarray:
    image = frame.copy()
    masks = entity_pixels(real)
    erase(image, masks["enemy"] | masks["diamond"] | masks["exit"] | masks["player"])
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for x, y in blobs(masks["diamond"]):
        draw.polygon([(x, y - 2), (x + 2, y), (x, y + 2), (x - 2, y)],
                     fill=(70, 220, 255), outline=(235, 255, 255))
    for x, y in blobs(masks["enemy"]):
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(250, 70, 82), outline=(80, 8, 16))
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 220, 220))
    for x, y in blobs(masks["exit"]):
        draw.rectangle([x - 2, y - 2, x + 2, y + 2], outline=(60, 250, 140), width=1)
    for x, y in blobs(masks["player"]):
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 214, 92), outline=(255, 250, 220))
    return np.asarray(canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from PIL import Image as PImage
    import subprocess
    work = args.out.parent / "hybrid-frames"
    work.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(sorted(args.frames.glob("step-*.png"))):
        strip = np.asarray(PImage.open(path).convert("RGB"))
        real, dream = strip[:, :64], strip[:, 64:]
        hybrid = draw_entities(dream, real)
        PImage.fromarray(np.concatenate([real, hybrid], axis=1)).save(work / f"f{index:04d}.png")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "12",
        "-i", str(work / "f%04d.png"), "-vf", "scale=768:-1:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-crf", "20", str(args.out),
    ], check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
