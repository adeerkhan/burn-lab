#!/usr/bin/env python3
"""Figures for the 64x64 flowers run: loss curve, sampler grids, metric bars.

    /tmp/glm-venv/bin/python scripts/plot_ddpm_flowers.py \
        --log /tmp/opencode/run64.log \
        --metrics /tmp/opencode/run64/metrics.json \
        --grid "real=/tmp/opencode/run64/real.png" \
        --grid "DDIM 20=/tmp/opencode/run64/sample-ddim-20.png" \
        --out /tmp/opencode/run64
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

STEP = re.compile(r"^step\s+(\d+)\s+loss\s+([0-9.]+)")


def loss_curve(log: Path, path: Path) -> None:
    steps, losses = [], []
    for line in log.read_text().splitlines():
        match = STEP.match(line.strip())
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))

    figure, axis = plt.subplots(figsize=(9, 3.6))
    axis.plot(steps, losses, color="#3b82f6", linewidth=0.9, alpha=0.55, label="reported loss")
    if len(losses) >= 20:
        window = max(5, len(losses) // 40)
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        axis.plot(
            steps[window - 1 :],
            smooth,
            color="#1d4ed8",
            linewidth=1.8,
            label=f"moving average ({window})",
        )
    axis.set_xlabel("step")
    axis.set_ylabel("v-prediction loss")
    axis.set_title("Training loss, 64x64 flowers")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path} ({len(steps)} points)")


def metric_bars(metrics: dict, path: Path) -> None:
    labels = list(metrics)
    fid = [metrics[label]["fid"] for label in labels]
    l1 = [metrics[label]["nearest_l1"] for label in labels]

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for axis, values, title, colour in [
        (axes[0], fid, "Frechet distance, 8x8x3 pool (lower is better)", "#f97316"),
        (axes[1], l1, "nearest-training L1 (lower is better)", "#3b82f6"),
    ]:
        bars = axis.bar(labels, values, color=colour)
        axis.set_title(title, fontsize=9)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")


def raw_grid(source: Path, path: Path, count: int = 16, columns: int = 4, scale: int = 3) -> None:
    """Tile the first `count` samples from a raw f32 NCHW dump into a PNG."""
    meta = json.loads(Path(f"{source}.json").read_text())
    _, channels, height, width = meta["shape"]
    flat = np.fromfile(source, dtype="<f4").reshape(-1, channels, height, width)
    flat = flat[:count]
    rows = (len(flat) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(flat):
        pixels = np.clip((image.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels), ((index % columns) * width, (index // columns) * height))
    canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
    canvas.save(path)
    print(f"wrote {path} {canvas.size}")


def comparison_grid(grids: list[tuple[str, Path]], path: Path, scale: int = 2) -> None:
    loaded = [(label, Image.open(source).convert("RGB")) for label, source in grids]
    width = max(image.width for _, image in loaded)
    label_width = max(90, max(len(label) for label, _ in loaded) * 8)
    height = sum(image.height for _, image in loaded)
    canvas = Image.new("RGB", (width + label_width, height), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, image in loaded:
        canvas.paste(image, (label_width, y))
        draw.text((10, y + image.height // 2 - 4), label, fill=(235, 235, 235))
        y += image.height
    if scale != 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
    canvas.save(path)
    print(f"wrote {path} {canvas.size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--grid", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--render-raw", action="append", default=[], metavar="PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.log:
        loss_curve(args.log, args.out / "loss.png")
    if args.metrics:
        metric_bars(json.loads(args.metrics.read_text()), args.out / "metrics.png")
    for source in args.render_raw:
        raw_grid(Path(source), args.out / f"grid-{Path(source).stem}.png")
    if args.grid:
        grids = []
        for item in args.grid:
            label, _, source = item.partition("=")
            grids.append((label, Path(source)))
        comparison_grid(grids, args.out / "comparison.png", args.scale)


if __name__ == "__main__":
    main()
