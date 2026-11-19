"""A small pseudo-3D city chase world with turning roads and detailed cars.

    python3 scripts/racing_env.py --preview
    python3 scripts/racing_env.py --episodes 300 --steps 128 --out /tmp/racing2/data

Two 64x64 views of one state: a pixel frame and a wireframe proxy. The engine
owns the road curvature, the cars and the chase; the proxy fixes the structure
and a VQVAE + transformer learns the appearance, as in CodeWorldModel.
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
SHOULDER = (90, 92, 96)
LANE = (225, 225, 210)
BUILDINGS = [(150, 110, 95), (110, 130, 150), (165, 145, 110), (120, 120, 140)]
CAR = (235, 70, 60)
CAR_DARK = (150, 35, 30)
POLICE = (45, 75, 185)
LIGHT_RED = (255, 60, 60)
LIGHT_BLUE = (70, 150, 255)
ACTIONS = ["stay", "left", "right", "accelerate", "brake"]


class RacingWorld:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.distance = 0.0
        self.lateral = 0.0
        self.speed = 1.2
        self.curve = 0.0
        self.police = [3.5, 6.5]
        self.time = 0
        self.seed = seed

    def step(self, action: int) -> None:
        action = int(action)
        if action == 1:
            self.lateral -= 1.5
        elif action == 2:
            self.lateral += 1.5
        elif action == 3:
            self.speed = min(2.5, self.speed + 0.5)
        elif action == 4:
            self.speed = max(0.4, self.speed - 0.5)
        self.lateral = float(np.clip(self.lateral, -3.2, 3.2))
        self.curve = float(np.clip(self.curve * 0.96 + self.rng.normal(0, 0.04), -0.6, 0.6))
        self.distance += self.speed * (0.7 if abs(self.lateral) > 2.2 else 1.0)
        self.time += 1
        for index, gap in enumerate(self.police):
            # Police close in when the player is slow or drifts wide, and fall
            # back under acceleration. They also cut across on city bends.
            target = 1.0 + (self.speed - 1.0) * 3.0 + abs(self.lateral) * 0.25 + index * 1.4
            target -= abs(self.curve) * 0.6 * (index + 1)
            target = float(np.clip(target + self.rng.normal(0, 0.12), 0.6, 14.0))
            self.police[index] = gap * 0.72 + target * 0.28

    def _centre(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        return 32.0 + self.curve * (t * 16.0 - 4.0)

    def _half(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        return 2.0 + t * 28.0

    def _road(self, draw, proxy: bool) -> None:
        if proxy:
            draw.rectangle((0, 0, SIZE - 1, SIZE - 1), fill=(0, 0, 0))
        else:
            draw.rectangle((0, 0, SIZE - 1, HORIZON - 1), fill=SKY)
        top, bottom = HORIZON, SIZE
        ct, cb = self._centre(top), self._centre(bottom)
        ht, hb = self._half(top), self._half(bottom)
        if proxy:
            draw.line([(ct - ht, top), (cb - hb, bottom)], fill=(220, 220, 220))
            draw.line([(ct + ht, top), (cb + hb, bottom)], fill=(220, 220, 220))
        else:
            draw.polygon([(ct - ht - 2, top), (ct + ht + 2, top),
                          (cb + hb + 3, bottom), (cb - hb - 3, bottom)], fill=SHOULDER)
            draw.polygon([(ct - ht, top), (ct + ht, top),
                          (cb + hb, bottom), (cb - hb, bottom)], fill=ROAD)
        for i in range(8):
            phase = (i + (self.distance * 0.07) % 1.0) / 8.0
            y = int(HORIZON + 1 + (phase ** 1.7) * (SIZE - HORIZON))
            c, half = self._centre(y), self._half(y)
            mark = max(1, int(1 + phase * 2))
            draw.line([(c - 1, y - mark), (c + 1, y + mark)],
                      fill=(200, 200, 200) if proxy else LANE, width=1)

    def _buildings(self, draw, proxy: bool) -> None:
        for side in (-1, 1):
            for i in range(9):
                phase = (i + (self.distance * 0.07) % 1.0) / 9.0
                depth = phase ** 1.7
                y0 = int(HORIZON + depth * (SIZE - HORIZON))
                c = self._centre(y0)
                half = self._half(y0)
                width = int(2 + depth * 13)
                height = int(3 + depth * 26)
                x_edge = c + side * (half + 1)
                box = (x_edge if side > 0 else x_edge - width,
                       max(HORIZON, y0 - height),
                       x_edge + width if side > 0 else x_edge,
                       min(SIZE, y0 + int(2 + depth * 4)))
                colour = (190, 190, 190) if proxy else BUILDINGS[(i + self.seed) % len(BUILDINGS)]
                draw.rectangle(box, fill=colour, outline=(25, 25, 30) if not proxy else None)
                if not proxy and depth > 0.16:
                    for cx in range(2):
                        for cy in range(2):
                            wx = box[0] + int(1 + cx * max(2, width / 3))
                            wy = box[1] + int(2 + cy * 5 * depth)
                            if wx + 1 < box[2] and wy + 1 < box[3]:
                                draw.rectangle((wx, wy, wx + 1, wy + 1), fill=(240, 220, 140))

    @staticmethod
    def _car(draw, x: float, y: int, width: int, height: int, colour, police: bool,
             proxy: bool, flash: bool) -> None:
        left, right = x - width, x + width
        top, bottom = y - height, y
        if proxy:
            draw.rectangle((left, top, right, bottom), outline=(255, 255, 255))
            return
        draw.rectangle((left + 1, bottom - 2, right - 1, bottom), fill=(20, 20, 24))
        draw.rectangle((left, top + 2, right, bottom - 1), fill=colour, outline=(20, 20, 24))
        cabin_w = max(1, int(width * 0.65))
        draw.rectangle((x - cabin_w, top, x + cabin_w, top + max(2, height // 2)),
                       fill=(90, 150, 200) if not police else (35, 55, 120))
        draw.rectangle((left, bottom - 2, left + 2, bottom - 1), fill=(250, 220, 120))
        draw.rectangle((right - 2, bottom - 2, right, bottom - 1), fill=(250, 220, 120))
        if police:
            bar = LIGHT_RED if flash else LIGHT_BLUE
            draw.rectangle((x - 2, top - 2, x + 2, top), fill=bar)
        else:
            draw.rectangle((left + 1, bottom - 3, right - 1, bottom - 2), fill=CAR_DARK)

    def _cars(self, draw, proxy: bool) -> None:
        py = 54
        px = self._centre(py) + self.lateral * 2
        self._car(draw, px, py, 7, 9, CAR, False, proxy, False)
        for index, gap in enumerate(self.police):
            depth = float(np.clip(1.0 - gap / 14.0, 0.04, 0.92))
            y = int(HORIZON + 2 + depth * (SIZE - HORIZON))
            width = max(2, int(2 + depth * 5))
            height = max(2, int(2 + depth * 6))
            # Police steer toward the player's lane as they close in.
            bias = self.lateral * 0.6 * (1.0 - gap / 14.0)
            x = self._centre(y) + bias + (index - 0.5) * (width + 3)
            self._car(draw, x, y, width, height, POLICE, True, proxy,
                      (self.time + index) % 8 < 4)

    def render(self, proxy: bool = False) -> np.ndarray:
        image = Image.new("RGB", (SIZE, SIZE))
        draw = ImageDraw.Draw(image)
        self._road(draw, proxy)
        self._buildings(draw, proxy)
        self._cars(draw, proxy)
        return np.asarray(image)


def policy(world: RacingWorld, rng):
    while True:
        if world.lateral * (1 if world.curve > 0 else -1) > 1.5:
            yield 2 if world.curve > 0 else 1
        elif rng.random() < 0.35:
            yield int(rng.integers(5))
        elif rng.random() < 0.5:
            yield 3 if world.speed < 2.1 else 0
        else:
            yield 1 if rng.random() < 0.5 else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/tmp/racing2/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.preview:
        world = RacingWorld(7)
        walk = policy(world, np.random.default_rng(8))
        strip = Image.new("RGB", (8 * SIZE * 3, SIZE * 3))
        for t in range(8):
            strip.paste(Image.fromarray(world.render()).resize((SIZE * 3, SIZE * 3), Image.NEAREST),
                        (t * SIZE * 3, 0))
            for _ in range(14):
                world.step(next(walk))
        strip.save("/tmp/opencode/racing2-preview.png")
        print("wrote preview")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.empty((args.episodes, args.steps + 1, SIZE, SIZE, 3), dtype=np.uint8)
    proxies = np.empty_like(frames)
    actions = np.empty((args.episodes, args.steps), dtype=np.uint8)
    for episode in range(args.episodes):
        world = RacingWorld(args.seed * 100003 + episode)
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
