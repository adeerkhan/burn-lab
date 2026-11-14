#!/usr/bin/env python3
"""Train the reference (PyTorch) DDPM on the same packed images as the port.

    /tmp/glm-venv/bin/python scripts/ddpm_reference_flowers.py \
        --data /tmp/opencode/flowers/train32.bin --size 32 --steps 4000 \
        --raw-prefix /tmp/opencode/reference32/draw

Matches the Rust run's settings: `dim 16`, `dim_mults (1,2,4,8)`, batch 16,
sigmoid schedule, 200 timesteps, v-prediction, AdamW 2e-4, EMA 0.999 updated
every step. Samples come from the EMA weights with DDIM, one 512-image draw per
file, so `scripts/ddpm_fidelity.py` can score both implementations identically.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/ddpm-ref")
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (  # noqa: E402
    GaussianDiffusion,
    Unet,
)
from ema_pytorch import EMA  # noqa: E402


class FlatImages:
    """A flat u8 RGB file read the way the Rust loader reads it."""

    def __init__(self, path: Path, size: int):
        data = np.fromfile(path, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        stride = 3 * size * size
        if data.size % stride:
            raise SystemExit(f"{path} is not a whole number of {size}x{size} images")
        # Interleaved RGB -> channel planes, exactly like `RawImages::batch`.
        images = data.reshape(-1, size, size, 3).transpose(0, 3, 1, 2).copy()
        self.images = torch.from_numpy(images)
        print(f"dataset {path} ({self.images.shape[0]} images)")

    def batch(self, count: int, generator: torch.Generator) -> torch.Tensor:
        index = torch.randint(0, self.images.shape[0], (count,), generator=generator)
        return self.images[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--raw-prefix", type=Path, required=True)
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
    print(f"reference Unet dim={args.dim} params={sum(p.numel() for p in model.parameters()):,}")

    ema = EMA(model, beta=args.ema_decay, update_every=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data = FlatImages(args.data, args.size)

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x0 = data.batch(args.batch, generator)
        times = torch.randint(0, args.timesteps, (args.batch,), generator=generator)
        loss = diffusion.p_losses(x0, times)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        if step == 1 or step % 100 == 0:
            print(f"step {step:>5}  loss {loss.item():.5f}")
    seconds = time.perf_counter() - started
    print(f"trained in {seconds:.1f}s")

    torch.save(ema.ema_model.state_dict(), f"{args.raw_prefix}-ema.pt")

    sampler = GaussianDiffusion(
        ema.ema_model,
        image_size=args.size,
        timesteps=args.timesteps,
        sampling_timesteps=args.ddim_steps,
        objective="pred_v",
        beta_schedule="sigmoid",
        auto_normalize=False,
    )

    started = time.perf_counter()
    samples = []
    for draw in range(1, args.draws + 1):
        torch.manual_seed(args.seed + 100 * draw)
        with torch.no_grad():
            out = sampler.sample(batch_size=args.samples)
        raw = out.numpy().astype("<f4")
        raw.tofile(f"{args.raw_prefix}-{draw}.raw")
        Path(f"{args.raw_prefix}-{draw}.raw.json").write_text(
            json.dumps({"shape": list(out.shape), "sampling_steps": args.ddim_steps, "sampler": "ddim"})
        )
        print(f"draw {draw}: {tuple(out.shape)}")
        samples.append(out)
    sampling_seconds = time.perf_counter() - started

    result = {
        "params": sum(p.numel() for p in model.parameters()),
        "steps": args.steps,
        "training_seconds": seconds,
        "sampling_seconds": sampling_seconds,
        "ema_decay": args.ema_decay,
        "sampler": f"ddim {args.ddim_steps}",
    }
    Path(f"{args.raw_prefix}-run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
