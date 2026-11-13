#!/usr/bin/env python3
"""Train and sample the *reference* DDPM on the same synthetic shapes the Rust
port uses, so the two can be compared directly.

    /tmp/glm-venv/bin/python scripts/ddpm_reference_run.py --steps 900
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

import sys

sys.path.insert(0, "/tmp/ddpm-ref")
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (  # noqa: E402
    GaussianDiffusion,
    Unet,
)


def template(index: int, size: int) -> np.ndarray:
    """The same eight shapes as `diffusion/src/data.rs`."""
    pixels = np.zeros((size, size, 3), dtype=np.float32)
    palette = [
        (1.0, 0.2, 0.2), (0.2, 0.9, 0.3), (0.2, 0.4, 1.0), (1.0, 0.9, 0.1),
        (0.9, 0.2, 0.9), (0.1, 0.9, 0.9), (1.0, 0.5, 0.1), (0.9, 0.9, 0.9),
    ]
    colour = palette[index % 8]
    margin = size // 8
    start, end, mid = margin, size - margin, size // 2
    kind = index % 8

    def put(x, y):
        if 0 <= x < size and 0 <= y < size:
            pixels[y, x] = colour

    if kind == 0:
        for y in range(mid - 2, mid + 2):
            for x in range(start, end):
                put(x, y)
    elif kind == 1:
        for x in range(mid - 2, mid + 2):
            for y in range(start, end):
                put(x, y)
    elif kind == 2:
        for y in range(start, end):
            for x in range(start, end):
                put(x, y)
    elif kind == 3:
        for y in range(start, end):
            for x in range(mid - 1, mid + 1):
                put(x, y)
        for y in range(mid - 1, mid + 1):
            for x in range(start, end):
                put(x, y)
    elif kind == 4:
        for i in range(start, end):
            put(i, i)
            put(size - 1 - i, i)
    elif kind == 5:
        for y in range(start, end):
            for x in range(start, end):
                if (x // 4 + y // 4) % 2 == 0:
                    put(x, y)
    elif kind == 6:
        radius = size * 0.3
        for y in range(size):
            for x in range(size):
                d = math.hypot(x - mid, y - mid)
                if abs(d - radius) < 2.0:
                    put(x, y)
    else:
        for y in range(start, end):
            for x in range(start, end):
                if x < start + 2 or x >= end - 2 or y < start + 2 or y >= end - 2:
                    put(x, y)
    return pixels


def batch(batch_size: int, size: int, generator: torch.Generator) -> torch.Tensor:
    indices = torch.randint(0, 8, (batch_size,), generator=generator).tolist()
    data = [template(index, size) for index in indices]
    pixels = np.stack(data)                      # [b, h, w, c] in [0, 1]
    tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2) * 2.0 - 1.0
    return tensor


def nearest_distance(samples: torch.Tensor, size: int) -> float:
    templates = [torch.from_numpy(template(i, size)).permute(2, 0, 1) * 2.0 - 1.0 for i in range(8)]
    flat = samples.clamp(-1, 1)
    total = 0.0
    for index in range(flat.shape[0]):
        best = min((flat[index] - t).abs().mean().item() for t in templates)
        total += best
    return total / flat.shape[0]


def grid(samples: torch.Tensor, path: Path) -> None:
    samples = ((samples.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    count, _, height, width = samples.shape
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    canvas = Image.new("RGB", (columns * width, rows * height))
    for index in range(count):
        image = Image.fromarray(samples[index].permute(1, 2, 0).numpy())
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("/tmp/opencode/reference"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    model = Unet(dim=args.dim, dim_mults=(1, 2, 4, 8), channels=3)
    diffusion = GaussianDiffusion(
        model,
        image_size=args.size,
        timesteps=args.timesteps,
        sampling_timesteps=args.ddim_steps,
        objective="pred_v",
        beta_schedule="sigmoid",
        auto_normalize=False,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"reference Unet dim={args.dim} params={parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x0 = batch(args.batch, args.size, generator)
        times = torch.randint(0, args.timesteps, (args.batch,), generator=generator)
        loss = diffusion.p_losses(x0, times)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(args.steps // 10, 1) == 0:
            print(f"step {step:>5}  loss {loss.item():.5f}")
    training_seconds = time.perf_counter() - started
    print(f"trained in {training_seconds:.1f}s")

    model.eval()
    results = {"params": parameter_count, "training_seconds": training_seconds, "steps": args.steps}

    # DDIM: the same module, fewer sampling steps.
    started = time.perf_counter()
    with torch.no_grad():
        ddim = diffusion.sample(batch_size=args.samples)
    ddim_seconds = time.perf_counter() - started
    results["ddim"] = {
        "steps": args.ddim_steps,
        "seconds": ddim_seconds,
        "distance": nearest_distance(ddim, args.size),
    }
    grid(ddim, args.out / "ddim.png")

    # DDPM: full chain, one step per timestep.
    torch.manual_seed(args.seed + 1)
    full = GaussianDiffusion(
        model,
        image_size=args.size,
        timesteps=args.timesteps,
        sampling_timesteps=args.timesteps,
        objective="pred_v",
        beta_schedule="sigmoid",
        auto_normalize=False,
    )
    started = time.perf_counter()
    with torch.no_grad():
        ddpm_samples = full.sample(batch_size=args.samples)
    ddpm_seconds = time.perf_counter() - started
    results["ddpm"] = {
        "steps": args.timesteps,
        "seconds": ddpm_seconds,
        "distance": nearest_distance(ddpm_samples, args.size),
    }
    grid(ddpm_samples, args.out / "ddpm.png")

    torch.manual_seed(args.seed)
    real = batch(args.samples, args.size, generator)
    grid(real, args.out / "real.png")
    results["real_distance"] = nearest_distance(real, args.size)

    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
