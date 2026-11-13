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
    args = parser.parse_args()

    photos = sorted(path for path in args.root.rglob("*.jpg"))
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
    print(f"{count} images of {args.size}x{args.size} -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
