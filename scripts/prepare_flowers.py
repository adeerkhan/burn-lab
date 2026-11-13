#!/usr/bin/env python3
"""Pack the TensorFlow flowers photos into a flat raw RGB file the Rust loader
can sample without decoding 3,670 JPEGs per epoch.

    /tmp/glm-venv/bin/python scripts/prepare_flowers.py --size 32
"""

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/opencode/flowers/flower_photos"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("/tmp/opencode/flowers/flowers32.bin"))
    parser.add_argument("--holdout", type=float, default=0.0)
    parser.add_argument("--holdout-out", type=Path, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    args = parser.parse_args()

    photos = sorted(path for path in args.root.rglob("*.jpg"))
    if args.holdout > 0:
        import random

        random.Random(args.shuffle_seed).shuffle(photos)
        cut = int(len(photos) * args.holdout)
        held_out = photos[:cut]
        photos = photos[cut:]
    else:
        held_out = []

    data = bytearray()
    for path in photos:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((args.size, args.size), Image.BICUBIC)
            data.extend(image.tobytes())

    args.out.write_bytes(bytes(data))
    count = len(data) // (args.size * args.size * 3)
    print(f"{count} train images of {args.size}x{args.size} -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if held_out and args.holdout_out is not None:
        held = bytearray()
        for path in held_out:
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                side = min(width, height)
                left = (width - side) // 2
                top = (height - side) // 2
                image = image.crop((left, top, left + side, top + side))
                image = image.resize((args.size, args.size), Image.BICUBIC)
                held.extend(image.tobytes())
        args.holdout_out.write_bytes(bytes(held))
        print(f"{len(held_out)} held-out images -> {args.holdout_out}")


if __name__ == "__main__":
    main()
