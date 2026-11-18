#!/usr/bin/env python3
"""Turn a world-dream rollout into a small labelled video.

    python3 scripts/world_video.py --frames /tmp/world/video \
        --actions /tmp/world/data/actions.bin --out docs/world-dream.mp4

`world-dream` writes one [real | dream] strip per step; this scales them with
nearest-neighbour (the pixels are the point), adds labels and the action, and
encodes with ffmpeg. Also writes a looping GIF for inline viewing.
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
    ap.add_argument("--actions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--filter", choices=["nearest", "bicubic", "lanczos"], default="nearest")
    ap.add_argument("--sharpen", type=float, default=0.0)
    a = ap.parse_args()

    frames = sorted(a.frames.glob("step-*.png"))
    if not frames:
        raise SystemExit(f"no step-*.png in {a.frames}")
    actions = np.fromfile(a.actions, dtype=np.uint8)
    raw = Image.open(frames[0]).convert("RGB")
    panel_w, panel_h = raw.width // 2, raw.height
    out_w, out_h = raw.width * a.scale, raw.height * a.scale + 62
    title_font, small_font = font(22 * a.scale // 6), font(15 * a.scale // 6)

    work = a.out.parent / "world-video-frames"
    work.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(frames):
        resample = {"nearest": Image.NEAREST, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}[a.filter]
        strip = Image.open(path).convert("RGB").resize(
            (out_w, panel_h * a.scale), resample
        )
        frame = Image.new("RGB", (out_w, out_h), (12, 13, 17))
        frame.paste(strip, (0, 0))
        draw = ImageDraw.Draw(frame)
        draw.text((12 * a.scale // 6, 4), "real room", fill=(200, 200, 200), font=title_font)
        draw.text(
            ((panel_w + 12) * a.scale // 6, 4),
            "world model dream",
            fill=(242, 184, 75),
            font=title_font,
        )
        action = int(actions[a.context + index]) if a.context + index < len(actions) else 0
        label = ACTION_LABELS[action] if action < len(ACTION_LABELS) else "?"
        draw.text(
            (12 * a.scale // 6, panel_h * a.scale + 8),
            f"step {index:03d}   action: {label}",
            fill=(180, 186, 198),
            font=small_font,
        )
        frame.save(work / f"f{index:04d}.png")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(a.fps),
            "-i", str(work / "f%04d.png"),
            "-vf", ("unsharp=5:5:" + str(a.sharpen) + ",format=yuv420p") if a.sharpen > 0 else "format=yuv420p",
            "-c:v", "libx264", "-crf", "20",
            "-movflags", "+faststart", str(a.out),
        ],
        check=True,
    )
    gif = a.out.with_suffix(".gif")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(a.fps), "-i", str(work / "f%04d.png"),
            "-vf", f"scale={out_w // 2}:-1:flags=neighbor,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop", "0", str(gif),
        ],
        check=True,
    )
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) and {gif}")


if __name__ == "__main__":
    main()
