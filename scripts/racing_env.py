"""First-person city chase: road, tall buildings, bends, police in the mirror.

    python3 scripts/racing_env.py --preview
    python3 scripts/racing_env.py --episodes 300 --steps 128 --out /tmp/fpcity/data

The camera sits in the driver's seat. Steering at an intersection bends the
road left or right; police close in behind and are visible in the rear-view
mirror at the top of the frame. Two 64x64 views are recorded: the pixel frame
and a wireframe proxy of the same structure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
HORIZON = 18
SKY = (96, 150, 205)
ROAD = (68, 70, 76)
SHOULDER = (92, 94, 98)
LANE = (225, 225, 210)
BUILDINGS = [
    (150, 110, 95), (110, 130, 150), (170, 145, 110), (120, 120, 140),
    (140, 100, 110), (100, 140, 125), (165, 160, 150), (110, 105, 135),
]
RED = (235, 70, 60)
BLUE = (50, 80, 200)
FLASH_RED = (255, 60, 60)
FLASH_BLUE = (70, 150, 255)
ACTIONS = ["stay", "left", "right", "accelerate", "brake"]


class ChaseWorld:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed + 1)
        self.seed = seed
        self.distance = 0.0
        self.speed = 1.3
        self.curve = 0.0
        self.turn = 0
        self.turn_left = False
        self.intersection = False
        self.police = [3.5, 6.5]
        self.time = 0

    def step(self, action: int) -> None:
        action = int(action)
        if self.intersection and action in (1, 2) and self.turn == 0:
            self.turn = 10
            self.turn_left = action == 1
        if action == 3:
            self.speed = min(2.6, self.speed + 0.35)
        elif action == 4:
            self.speed = max(0.4, self.speed - 0.35)
        if self.turn > 0:
            self.curve = float(np.clip(self.curve + (-0.12 if self.turn_left else 0.12), -0.9, 0.9))
            self.turn -= 1
        else:
            self.curve *= 0.93
        self.distance += self.speed
        self.time += 1
        self.intersection = (self.distance % 24) < 1.2
        for index, gap in enumerate(self.police):
            target = 1.0 + (self.speed - 1.0) * 3.0 + abs(self.curve) * 1.5 + index * 1.6
            target = float(np.clip(target + self.rng.normal(0, 0.1), 0.6, 14.0))
            self.police[index] = gap * 0.7 + target * 0.3

    def _centre(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        return 32.0 + self.curve * (1.0 - t) * 16.0

    def _half(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        return 2.0 + t * 29.0

    def _road(self, draw, proxy: bool) -> None:
        if proxy:
            draw.rectangle((0, 0, SIZE - 1, SIZE - 1), fill=(0, 0, 0))
        else:
            draw.rectangle((0, 0, SIZE - 1, HORIZON - 1), fill=SKY)
        ct, cb = self._centre(HORIZON), self._centre(SIZE)
        ht, hb = self._half(HORIZON), self._half(SIZE)
        if proxy:
            draw.line([(ct - ht, HORIZON), (cb - hb, SIZE)], fill=(220, 220, 220))
            draw.line([(ct + ht, HORIZON), (cb + hb, SIZE)], fill=(220, 220, 220))
        else:
            draw.polygon([(ct - ht - 2, HORIZON), (ct + ht + 2, HORIZON),
                          (cb + hb + 3, SIZE), (cb - hb - 3, SIZE)], fill=SHOULDER)
            draw.polygon([(ct - ht, HORIZON), (ct + ht, HORIZON),
                          (cb + hb, SIZE), (cb - hb, SIZE)], fill=ROAD)
        for i in range(8):
            phase = (i + (self.distance * 0.11) % 1.0) / 8.0
            y = int(HORIZON + 1 + (phase ** 1.7) * (SIZE - HORIZON))
            c = self._centre(y)
            mark = max(1, int(1 + phase * 2))
            draw.line([(c - 1, y - mark), (c + 1, y + mark)],
                      fill=(200, 200, 200) if proxy else LANE, width=1)

    def _buildings(self, draw, proxy: bool) -> None:
        for side in (-1, 1):
            for i in range(9):
                phase = (i + (self.distance * 0.11) % 1.0) / 9.0
                depth = phase ** 1.7
                y0 = int(HORIZON + depth * (SIZE - HORIZON))
                c = self._centre(y0)
                half = self._half(y0)
                width = int(2 + depth * 13)
                height = int(4 + depth * 27)
                x_edge = c + side * (half + 1)
                box = (x_edge if side > 0 else x_edge - width,
                       max(HORIZON, y0 - height),
                       x_edge + width if side > 0 else x_edge,
                       min(SIZE, y0 + int(2 + depth * 5)))
                colour = (185, 185, 185) if proxy else BUILDINGS[(i + self.seed) % len(BUILDINGS)]
                draw.rectangle(box, fill=colour, outline=(25, 25, 30) if not proxy else None)
                if not proxy and depth > 0.14:
                    for cx in range(2):
                        for cy in range(3):
                            wx = box[0] + int(1 + cx * max(2, width / 3))
                            wy = box[1] + int(2 + cy * 5 * depth)
                            if wx + 1 < box[2] and wy + 1 < box[3]:
                                draw.rectangle((wx, wy, wx + 1, wy + 1), fill=(240, 220, 140))

    def _mirror(self, draw, proxy: bool) -> None:
        box = (18, 1, 46, 12)
        draw.rectangle(box, fill=(25, 28, 34), outline=(210, 210, 215) if not proxy else (255, 255, 255))
        for index, gap in enumerate(self.police):
            depth = float(np.clip(1.0 - gap / 14.0, 0.05, 1.0))
            width = max(1, int(1 + depth * 5))
            x = 24 + index * 9
            y = 4 + int((1.0 - depth) * 3)
            colour = (210, 210, 210) if proxy else BLUE
            draw.rectangle((x - width, y, x + width, y + 4), fill=colour, outline=(10, 10, 15))
            if not proxy:
                flash = FLASH_RED if (self.time + index) % 6 < 3 else FLASH_BLUE
                draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=flash)

    def _hood(self, draw, proxy: bool) -> None:
        if proxy:
            draw.line([(0, SIZE - 2), (SIZE, SIZE - 2)], fill=(255, 255, 255))
        else:
            draw.polygon([(6, SIZE), (20, SIZE - 5), (44, SIZE - 5), (58, SIZE)], fill=RED)
            draw.rectangle((28, SIZE - 4, 36, SIZE - 2), fill=(60, 60, 65))

    def render(self, proxy: bool = False) -> np.ndarray:
        image = Image.new("RGB", (SIZE, SIZE))
        draw = ImageDraw.Draw(image)
        self._road(draw, proxy)
        self._buildings(draw, proxy)
        self._mirror(draw, proxy)
        self._hood(draw, proxy)
        return np.asarray(image)


def policy(world: ChaseWorld, rng):
    while True:
        if world.turn > 0:
            yield 0
        elif world.intersection and rng.random() < 0.55:
            yield 1 if rng.random() < 0.5 else 2
        elif rng.random() < 0.25:
            yield int(rng.integers(5))
        elif rng.random() < 0.5:
            yield 3 if world.speed < 2.0 else 0
        else:
            yield 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/tmp/fpcity/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.preview:
        world = ChaseWorld(7)
        walk = policy(world, np.random.default_rng(8))
        strip = Image.new("RGB", (8 * SIZE * 3, SIZE * 3))
        for t in range(8):
            strip.paste(Image.fromarray(world.render()).resize((SIZE * 3, SIZE * 3), Image.NEAREST),
                        (t * SIZE * 3, 0))
            for _ in range(16):
                world.step(next(walk))
        strip.save("/tmp/opencode/fpcity-preview.png")
        print("wrote preview")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.empty((args.episodes, args.steps + 1, SIZE, SIZE, 3), dtype=np.uint8)
    proxies = np.empty_like(frames)
    actions = np.empty((args.episodes, args.steps), dtype=np.uint8)
    for episode in range(args.episodes):
        world = ChaseWorld(args.seed * 100003 + episode)
        walk = policy(world, np.random.default_rng(episode + 1))
        frames[episode, 0] = world.render()
        proxies[episode, 0] = world.render(proxy=True)
        for t in range(args.steps):
            action = next(walk)
            actions[episode, t] = action
            world.step(action)
            frames[episode, t + 1] = world.render()
            proxies[episode, t + 1] = world.render(proxy=True)
        if episode % 50 == 0:
            print(f"episode {episode}", flush=True)
    (args.out / "frames.bin").write_bytes(np.ascontiguousarray(frames).reshape(-1).tobytes())
    (args.out / "proxy.bin").write_bytes(np.ascontiguousarray(proxies).reshape(-1).tobytes())
    (args.out / "actions.bin").write_bytes(np.ascontiguousarray(actions).tobytes())
    (args.out / "meta.json").write_text(json.dumps({
        "episodes": args.episodes, "steps": args.steps, "size": SIZE, "actions": ACTIONS,
    }))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
