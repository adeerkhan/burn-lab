#!/usr/bin/env python3
"""Render weight and gradient heatmaps from `glm53`-style diffsion dumps.

    /tmp/glm-venv/bin/python scripts/plot_ddpm_inspection.py \
        --dump /tmp/opencode/flowers-run/dump.json --out /tmp/opencode/flowers-run
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def select(layers, limit=24):
    """Keep the layers worth drawing: large enough to have structure."""
    usable = [layer for layer in layers if len(layer["grid"]) == 32 * 32 and np.ptp(layer["grid"]) > 0]
    if len(usable) <= limit:
        return usable
    step = len(usable) / limit
    return [usable[int(index * step)] for index in range(limit)]


def draw(layers, title, path, cmap="viridis"):
    columns = 6
    rows = (len(layers) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.35))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, layer in zip(axes, layers):
        grid = np.array(layer["grid"], dtype=np.float32).reshape(32, 32)
        axis.imshow(grid, cmap=cmap, interpolation="nearest")
        axis.set_title(
            f"#{layer['index']} {tuple(layer['dims'])}\n"
            f"mean {layer['mean']:+.4f}  sd {layer['std']:.4f}",
            fontsize=7,
        )
        axis.axis("on")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"wrote {path} ({len(layers)} layers)")


def draw_magnitudes(dump, path):
    weights = dump["weights"]
    gradients = dump["gradients"]
    if not gradients:
        return
    by_index = {layer["index"]: layer for layer in gradients}
    indices = [layer["index"] for layer in weights if layer["index"] in by_index and layer["std"] > 0]
    weight_sd = [weights[i]["std"] for i in indices if i < len(weights)]
    grad_sd = [by_index[i]["std"] for i in indices]

    figure, axes = plt.subplots(2, 1, figsize=(max(8, len(indices) * 0.28), 6), sharex=True)
    axes[0].bar(range(len(weight_sd)), weight_sd, color="#3b82f6")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("weight sd")
    axes[1].bar(range(len(grad_sd)), grad_sd, color="#f97316")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("gradient sd")
    axes[1].set_xlabel("layer index (visit order)")
    figure.suptitle("Per-layer scale: weights above, gradients below (log)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dump = json.loads(args.dump.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    draw(select(dump["weights"]), "UNet weights (32x32 block means)", args.out / "weights.png")
    if dump["gradients"]:
        draw(select(dump["gradients"]), "Gradients of those layers", args.out / "gradients.png", cmap="magma")
    draw_magnitudes(dump, args.out / "layer-scales.png")


if __name__ == "__main__":
    main()
