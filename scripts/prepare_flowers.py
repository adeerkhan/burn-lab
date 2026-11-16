#!/usr/bin/env python3
"""Pack flower photographs into a flat raw RGB file the Rust loader can sample.

    /tmp/glm-venv/bin/python scripts/prepare_flowers.py --size 64 --holdout 0.1 \
        --out /tmp/flowers/train64.bin --holdout-out /tmp/flowers/held64.bin

Several photo sets can be combined, and each training image can contribute
random-resized-crop variants, which is the cheapest regulariser available when
the dataset is small:

    --root /tmp/flowers/flower_photos --root /tmp/oxford/jpg --augment 2 --dedupe

`--dedupe` drops any training image whose packed bytes match a held-out image,
so the two sources cannot leak into each other. The held-out split always comes
from the first root, which keeps the metric's reference set fixed across
experiments.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def random_resized_crop(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """The usual random crop: random area fraction, random aspect ratio."""
    width, height = image.size
    area = width * height
    for _ in range(10):
        target_area = area * rng.uniform(0.35, 1.0)
        aspect = float(np.exp(rng.uniform(-0.7, 0.7)))
        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / aspect) ** 0.5))
        if crop_w <= width and crop_h <= height:
            left = int(rng.integers(0, width - crop_w + 1))
            top = int(rng.integers(0, height - crop_h + 1))
            return image.crop((left, top, left + crop_w, top + crop_h))
    return square_crop(image)


def pack(path: Path, size: int, rng: np.random.Generator, augment: int) -> list[bytes]:
    """One centre crop plus `augment` random crops, as packed byte strings."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        variants = [square_crop(image)]
        for _ in range(augment):
            crop = random_resized_crop(image, rng)
            if rng.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            variants.append(crop)
    out = []
    for variant in variants:
        out.append(variant.resize((size, size), Image.BICUBIC).tobytes())
    return out


def digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="photo folder; repeat to combine sets. Default: the TensorFlow set.",
    )
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("/tmp/opencode/flowers/flowers32.bin"))
    parser.add_argument("--holdout", type=float, default=0.0)
    parser.add_argument("--holdout-out", type=Path, default=None)
    parser.add_argument("--augment", type=int, default=0, help="random crops per training image")
    parser.add_argument("--dedupe", action="store_true", help="drop training copies of held-out images")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    roots = args.root or [Path("/tmp/opencode/flowers/flower_photos")]
    rng = np.random.default_rng(args.seed)

    # The held-out split comes from the first root only, so the metric's
    # reference set does not move when more data is added.
    first = sorted(path for path in roots[0].rglob("*.jpg"))
    if args.holdout > 0:
        order = rng.permutation(len(first))
        cut = int(len(first) * args.holdout)
        held_paths = [first[index] for index in order[:cut]]
        first = [first[index] for index in order[cut:]]
    else:
        held_paths = []

    held_digests = set()
    held_bytes = bytearray()
    for path in held_paths:
        data = pack(path, args.size, rng, 0)[0]
        held_digests.add(digest(data))
        held_bytes.extend(data)

    # The held-out photographs are excluded from every root, by path. The first
    # version dropped them from one list and then re-read the whole folder, so
    # the held-out images came back into the training set - and into the
    # reference set - through their augmented variants.
    held_paths = set(held_paths)

    train = bytearray()
    count = 0
    duplicates = 0
    for root in roots:
        for path in sorted(root.rglob("*.jpg")):
            if path in held_paths:
                continue
            for data in pack(path, args.size, rng, args.augment):
                if args.dedupe and digest(data) in held_digests:
                    duplicates += 1
                    continue
                train.extend(data)
                count += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(bytes(train))
    print(
        f"{count} training images of {args.size}x{args.size} -> {args.out} "
        f"({args.out.stat().st_size / 1e6:.1f} MB, {duplicates} duplicates dropped)"
    )
    if held_paths and args.holdout_out:
        args.holdout_out.parent.mkdir(parents=True, exist_ok=True)
        args.holdout_out.write_bytes(bytes(held_bytes))
        print(f"{len(held_paths)} held-out images -> {args.holdout_out}")


if __name__ == "__main__":
    main()
