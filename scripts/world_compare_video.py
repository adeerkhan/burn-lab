#!/usr/bin/env python3
"""Build a three-way comparison video: real room / world-model dream / copy baseline.

    python3 scripts/world_compare_video.py \
        --frames /tmp/world/dream-p4 --real /tmp/world/data/frames.npy \
        --actions /tmp/world/data/actions.bin --out docs/world-compare.mp4

The copy panel is the last real frame of the warm-up context, held fixed — the
only trivial predictor available once the dream stops seeing reality. Each
frame also prints the pixel MAE of the dream and of the copy against the real
room, so the comparison is auditable rather than just visual.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACTION_LABELS = ["stay", "forward", "backward", "turn left", "turn right"]


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--actions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--context", type=int, default=4)
    a = ap.parse_args()

    strips = sorted(a.frames.glob("step-*.png"))
    if not strips:
        raise SystemExit(f"no step-*.png in {a.frames}")
    real_frames = np.load(a.real, mmap_mode="r")
    actions = np.fromfile(a.actions, dtype=np.uint8)
    copy = np.asarray(real_frames[0, a.context - 1], dtype=np.float32)

    small, big = font(14), font(15)
    work = a.out.parent / "world-compare-frames"
    work.mkdir(parents=True, exist_ok=True)
    panel = 64 * a.scale
    width = panel * 3
    height = panel + 56

    for index, path in enumerate(strips):
        step = index
        real = np.asarray(real_frames[0, a.context + step], dtype=np.float32)
        strip = Image.open(path).convert("RGB")
        dream = strip.crop((strip.width // 2, 0, strip.width, strip.height))
        dream_np = np.asarray(dream, dtype=np.float32)

        dream_mae = np.abs(dream_np - real).mean()
        copy_mae = np.abs(copy - real).mean()
        action = int(actions[a.context + step]) if a.context + step < len(actions) else 0
        label = ACTION_LABELS[action] if action < len(ACTION_LABELS) else "?"

        frame = Image.new("RGB", (width, height), (12, 13, 17))
        frame.paste(dream.resize((panel, panel), Image.NEAREST), (panel, 0))
        frame.paste(
            Image.fromarray(copy.astype(np.uint8)).resize((panel, panel), Image.NEAREST),
            (panel * 2, 0),
        )
        frame.paste(
            Image.fromarray(real.astype(np.uint8)).resize((panel, panel), Image.NEAREST),
            (0, 0),
        )
        draw = ImageDraw.Draw(frame)
        for i, title in enumerate(("real room", "world model dream", "copy last real frame")):
            draw.text((i * panel + 8, 4), title, fill=(225, 225, 230), font=small)
        draw.text(
            (8, panel + 10),
            f"step {step:03d}   action: {label}",
            fill=(200, 200, 200),
            font=big,
        )
        draw.text(
            (panel + 8, panel + 10),
            f"dream MAE {dream_mae:5.1f}",
            fill=(242, 184, 75),
            font=big,
        )
        draw.text(
            (panel * 2 + 8, panel + 10),
            f"copy MAE {copy_mae:5.1f}",
            fill=(150, 180, 210),
            font=big,
        )
        frame.save(work / f"f{index:04d}.png")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
            "-i", str(work / "f%04d.png"), "-vf", "format=yuv420p",
            "-c:v", "libx264", "-crf", "22", "-movflags", "+faststart", str(a.out),
        ],
        check=True,
    )
    gif = a.out.with_suffix(".gif")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
            "-i", str(work / "f%04d.png"),
            "-vf", f"scale={width // 2}:-1:flags=neighbor,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop", "0", str(gif),
        ],
        check=True,
    )
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) and {gif}")


if __name__ == "__main__":
    main()
