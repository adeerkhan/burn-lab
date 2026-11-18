#!/usr/bin/env python3
"""Generate the tiny sound effects used by the dungeon game.

    python3 scripts/make_sounds.py --out docs/audio
"""

import argparse
import wave
from pathlib import Path

import numpy as np

RATE = 22050


def write(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(samples, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())


def tone(freq: float, duration: float, decay: float = 8.0) -> np.ndarray:
    t = np.linspace(0, duration, int(RATE * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)


def mix(*parts: np.ndarray) -> np.ndarray:
    length = max(len(part) for part in parts)
    out = np.zeros(length)
    for part in parts:
        out[: len(part)] += part
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/audio"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pickup = mix(tone(880, 0.08), 0.7 * tone(1320, 0.12, 6.0))
    write(args.out / "pickup.wav", pickup * 0.6)

    t = np.linspace(0, 0.22, int(RATE * 0.22), endpoint=False)
    hit = 0.5 * np.sign(np.sin(2 * np.pi * 110 * t)) * np.exp(-9 * t)
    hit += 0.25 * np.random.default_rng(1).normal(0, 1, len(t)) * np.exp(-18 * t)
    write(args.out / "hit.wav", hit * 0.6)

    level = np.concatenate([tone(f, 0.09, 5.0) for f in (523, 659, 784, 1047)])
    write(args.out / "level.wav", level * 0.5)

    print(f"wrote {args.out}/pickup.wav hit.wav level.wav")


if __name__ == "__main__":
    main()
