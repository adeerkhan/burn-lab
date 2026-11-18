#!/usr/bin/env python3
"""Build the gameplay video and mix the event sounds into it.

    python3 scripts/game_video.py --frames /tmp/game --events /tmp/game/events.json \
        --audio docs/audio --out docs/dungeon-game.mp4
"""

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 22050
FPS = 12
SOUND = {"pickup": "pickup", "unlock": "pickup", "level": "level", "hit": "hit", "gameover": "hit"}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    count = len(list(args.frames.glob("step-*.png")))
    duration = count / args.fps + 0.4
    track = np.zeros(int(RATE * duration), dtype=np.float32)
    events = json.loads(args.events.read_text())
    for time, name in events:
        sound = SOUND.get(name)
        if sound is None:
            continue
        data = read_wav(args.audio / f"{sound}.wav")
        start = int((time / args.fps) * RATE)
        end = min(len(track), start + len(data))
        track[start:end] += data[: end - start] * 0.5
    track = np.clip(track, -1, 1)
    audio = args.out.with_suffix(".wav")
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((track * 32767).astype(np.int16).tobytes())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(args.fps), "-i", str(args.frames / "step-%04d.png"),
        "-i", str(audio),
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-shortest", str(args.out),
    ], check=True)
    gif = args.out.with_suffix(".gif")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(args.out),
        "-vf", "fps=6,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
        "-loop", "0", str(gif),
    ], check=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) and {gif}")


if __name__ == "__main__":
    main()
