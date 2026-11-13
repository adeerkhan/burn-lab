#!/usr/bin/env python3
"""The floor of the pixel-Frechet estimator: real images against real images.

No model can score below the distance between the training distribution and the
held-out distribution, because a perfect sampler reproduces the former and is
scored against the latter. If that floor is above the target, the target is
unreachable with this estimator and must be restated.

    /tmp/glm-venv/bin/python scripts/ddpm_floor.py \
        --train /tmp/opencode/flowers/train64.bin \
        --held /tmp/opencode/flowers/held64.bin --size 64
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ddpm_fidelity import colour_distance, frechet, load_u8, pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--held", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--subset", type=int, default=367)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    train = pool(load_u8(args.train, args.size), 8)
    held = pool(load_u8(args.held, args.size), 8)
    print(f"train {train.shape}, held {held.shape}")

    # A model trained on all 3,303 images is scored against the held-out set, so
    # the full-training-set distance is the floor that matters.
    full = frechet(train, held)

    # Matched sample counts, to show how much of that is finite-sample noise.
    rng = np.random.default_rng(11)
    draws = []
    for _ in range(args.draws):
        subset = train[rng.choice(len(train), args.subset, replace=False)]
        draws.append(frechet(subset, held))

    result = {
        "floor_full_train_vs_held": full,
        "floor_matched_subsets": draws,
        "floor_matched_mean": float(np.mean(draws)),
        "floor_matched_sd": float(np.std(draws, ddof=1)),
        "floor_colour_distance": colour_distance(train, held),
        "samples": {"train": int(len(train)), "held": int(len(held)), "subset": args.subset},
        "dims": int(train.shape[1]),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
