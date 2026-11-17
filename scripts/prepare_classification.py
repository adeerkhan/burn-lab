#!/usr/bin/env python3
"""Pack the five-class TensorFlow flower set for classification.

Writes one flat u8 file of 64x64 RGB images and one u8 file of class ids, plus a
holdout with the same format. The Rust side streams random rows from these
files with `seek`, so the whole dataset never has to sit in memory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/adeerkhan/tmp/flowers/flower_photos"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/flowers/cls"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    classes = sorted(
        path.name for path in args.root.iterdir() if path.is_dir() and path.name != "LICENSE.txt"
    )
    rng = np.random.default_rng(args.seed)
    train_images, train_labels = bytearray(), bytearray()
    held_images, held_labels = bytearray(), bytearray()

    for class_id, name in enumerate(classes):
        paths = sorted((args.root / name).glob("*.jpg"))
        order = rng.permutation(len(paths))
        cut = int(len(paths) * args.holdout)
        for split, indices in (("train", order[cut:]), ("held", order[:cut])):
            for index in indices:
                with Image.open(paths[index]) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    side = min(width, height)
                    image = image.crop(((width - side) // 2, (height - side) // 2,
                                        (width + side) // 2, (height + side) // 2))
                    image = image.resize((args.size, args.size), Image.BICUBIC)
                    data = image.tobytes()
                if split == "train":
                    train_images.extend(data)
                    train_labels.append(class_id)
                else:
                    held_images.extend(data)
                    held_labels.append(class_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for split, images, labels in (
        ("train", train_images, train_labels),
        ("held", held_images, held_labels),
    ):
        image_path = args.out.with_name(f"{args.out.name}-{split}-images.bin")
        label_path = args.out.with_name(f"{args.out.name}-{split}-labels.bin")
        image_path.write_bytes(bytes(images))
        label_path.write_bytes(bytes(labels))
        print(f"{split}: {len(labels)} images -> {image_path} ({len(images) / 1e6:.1f} MB)")

    (args.out.with_name(f"{args.out.name}-classes.json")).write_text(
        json.dumps({"classes": classes, "size": args.size})
    )
    print("classes:", classes)


if __name__ == "__main__":
    main()
