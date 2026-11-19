"""Third-person GTA-style city chase with a detailed car and Manhattan blocks.

    python3 scripts/racing_env.py --preview
    python3 scripts/racing_env.py --episodes 300 --steps 128 --out /tmp/chase/data

Camera follows behind the player's car. Tall varied buildings line the street,
a distant skyline sits at the vanishing point, and police cars close in behind
with flashing light bars. Two views are recorded: the pixel frame and a proxy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
HORIZON = 14
ROAD = (66, 68, 74)
SIDEWALK = (105, 107, 112)
LANE = (230, 228, 205)
BUILDINGS = [
    (120, 118, 126), (142, 130, 112), (105, 124, 145), (130, 108, 105),
    (150, 145, 135), (110, 132, 120), (138, 122, 140), (125, 140, 150),
]
RED = (220, 55, 45)
RED_DARK = (140, 30, 28)
POLICE = (45, 70, 170)
FLASH_RED = (255, 60, 60)
FLASH_BLUE = (70, 150, 255)
ACTIONS = ["stay", "left", "right", "accelerate", "brake"]


class ChaseWorld:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed + 1)
        self.seed = seed
        self.distance = 0.0
        self.speed = 1.4
        self.curve = 0.0
        self.turn = 0
        self.turn_left = False
        self.intersection = False
        self.police = [4.0, 7.0]
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
            self.curve = float(np.clip(self.curve + (-0.10 if self.turn_left else 0.10), -0.85, 0.85))
            self.turn -= 1
        else:
            self.curve *= 0.94
        self.distance += self.speed
        self.time += 1
        self.intersection = (self.distance % 26) < 1.4
        for index, gap in enumerate(self.police):
            target = 1.0 + (self.speed - 1.0) * 3.2 + abs(self.curve) * 1.2 + index * 1.8
            target = float(np.clip(target + self.rng.normal(0, 0.1), 0.7, 15.0))
            self.police[index] = gap * 0.7 + target * 0.3

    def _centre(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        return 32.0 + self.curve * (1.0 - t) * 18.0

    def _half(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        return 2.0 + t * 30.0

    def _sky(self, draw, proxy: bool) -> None:
        if proxy:
            draw.rectangle((0, 0, SIZE - 1, SIZE - 1), fill=(0, 0, 0))
            return
        for y in range(HORIZON):
            t = y / max(1, HORIZON)
            colour = (int(255 - 70 * t), int(190 - 60 * t), int(130 + 50 * t))
            draw.line([(0, y), (SIZE, y)], fill=colour)
        # Distant skyline at the vanishing point.
        for i in range(14):
            x = i * 5
            height = 2 + (i * 7 + self.seed) % 7
            draw.rectangle((x, HORIZON - height, x + 4, HORIZON),
                           fill=(110, 105, 115), outline=(70, 70, 78))

    def _road(self, draw, proxy: bool) -> None:
        ct, cb = self._centre(HORIZON), self._centre(SIZE)
        ht, hb = self._half(HORIZON), self._half(SIZE)
        if proxy:
            draw.line([(ct - ht, HORIZON), (cb - hb, SIZE)], fill=(220, 220, 220))
            draw.line([(ct + ht, HORIZON), (cb + hb, SIZE)], fill=(220, 220, 220))
        else:
            draw.polygon([(ct - ht - 3, HORIZON), (ct + ht + 3, HORIZON),
                          (cb + hb + 4, SIZE), (cb - hb - 4, SIZE)], fill=SIDEWALK)
            draw.polygon([(ct - ht, HORIZON), (ct + ht, HORIZON),
                          (cb + hb, SIZE), (cb - hb, SIZE)], fill=ROAD)
        for lane in (-1, 1):
            for i in range(9):
                phase = (i + (self.distance * 0.1) % 1.0) / 9.0
                y = int(HORIZON + 1 + (phase ** 1.7) * (SIZE - HORIZON))
                c = self._centre(y)
                offset = int(lane * (2 + phase * 9))
                mark = max(1, int(1 + phase * 2))
                draw.line([(c + offset, y - mark), (c + offset, y + mark)],
                          fill=(205, 205, 195) if proxy else LANE, width=1)

    def _buildings(self, draw, proxy: bool) -> None:
        for side in (-1, 1):
            for i in range(11):
                phase = (i + (self.distance * 0.1) % 1.0) / 11.0
                depth = phase ** 1.7
                y0 = int(HORIZON + depth * (SIZE - HORIZON))
                c = self._centre(y0)
                half = self._half(y0)
                width = int(3 + depth * 13)
                height = int(6 + depth * 30)
                x_edge = c + side * (half + 2)
                box = (x_edge if side > 0 else x_edge - width,
                       max(2, y0 - height),
                       x_edge + width if side > 0 else x_edge,
                       min(SIZE, y0 + int(2 + depth * 4)))
                colour = (185, 185, 185) if proxy else BUILDINGS[(i + self.seed + (side + 1)) % len(BUILDINGS)]
                if not proxy and depth < 0.4:
                    colour = tuple(int(v * 0.8) for v in colour)
                draw.rectangle(box, fill=colour, outline=(25, 25, 30) if not proxy else None)
                if not proxy and depth > 0.1:
                    rows = max(1, int(height / 5))
                    cols = max(1, int(width / 3))
                    for col in range(cols):
                        for row in range(rows):
                            wx = box[0] + 1 + col * max(2, width // max(1, cols))
                            wy = box[1] + 2 + row * 4
                            if wx + 1 < box[2] and wy + 1 < box[3]:
                                lit = (col * 3 + row * 5 + i + self.seed) % 7 < 3
                                draw.rectangle((wx, wy, wx + 1, wy + 1),
                                               fill=(245, 225, 150) if lit else (60, 70, 85))

    def _player(self, draw, proxy: bool) -> None:
        y = 49
        c = self._centre(y)
        if proxy:
            draw.rectangle((c - 7, y - 6, c + 7, y + 6), outline=(255, 255, 255))
            return
        draw.ellipse((c - 9, y + 3, c + 9, y + 8), fill=(20, 20, 24))
        draw.rectangle((c - 7, y - 5, c + 7, y + 5), fill=RED, outline=(20, 20, 24))
        draw.polygon([(c - 5, y - 5), (c - 3, y - 9), (c + 3, y - 9), (c + 5, y - 5)],
                     fill=RED_DARK)
        draw.rectangle((c - 3, y - 8, c + 3, y - 5), fill=(70, 120, 160))
        draw.rectangle((c - 6, y + 1, c - 3, y + 4), fill=(255, 70, 60))
        draw.rectangle((c + 3, y + 1, c + 6, y + 4), fill=(255, 70, 60))
        draw.rectangle((c - 2, y + 3, c + 2, y + 4), fill=(40, 40, 45))
        draw.rectangle((c - 8, y - 3, c - 7, y + 3), fill=(15, 15, 18))
        draw.rectangle((c + 7, y - 3, c + 8, y + 3), fill=(15, 15, 18))

    def _police(self, draw, proxy: bool) -> None:
        for index, gap in enumerate(self.police):
            y = int(np.clip(60 - gap * 0.7, 50, 62))
            c = self._centre(y) + (index - 0.5) * 12
            if proxy:
                draw.rectangle((c - 5, y - 5, c + 5, y + 5), outline=(255, 255, 255))
                continue
            draw.ellipse((c - 6, y + 3, c + 6, y + 7), fill=(20, 20, 24))
            draw.rectangle((c - 5, y - 5, c + 5, y + 5), fill=POLICE, outline=(15, 15, 22))
            draw.polygon([(c - 4, y - 5), (c - 2, y - 8), (c + 2, y - 8), (c + 4, y - 5)],
                         fill=(30, 50, 130))
            draw.rectangle((c - 2, y - 7, c + 2, y - 5), fill=(90, 140, 200))
            flash = FLASH_RED if (self.time + index) % 6 < 3 else FLASH_BLUE
            draw.rectangle((c - 3, y - 9, c + 3, y - 7), fill=flash)
            draw.rectangle((c - 4, y + 2, c - 2, y + 4), fill=(255, 90, 70))
            draw.rectangle((c + 2, y + 2, c + 4, y + 4), fill=(255, 90, 70))

    def render(self, proxy: bool = False) -> np.ndarray:
        image = Image.new("RGB", (SIZE, SIZE))
        draw = ImageDraw.Draw(image)
        self._sky(draw, proxy)
        self._road(draw, proxy)
        self._buildings(draw, proxy)
        self._police(draw, proxy)
        self._player(draw, proxy)
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
    ap.add_argument("--out", type=Path, default=Path("/tmp/chase/data"))
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
        strip.save("/tmp/opencode/chase-preview.png")
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
