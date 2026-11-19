"""A tiny pseudo-3D racing world with GTA-style buildings and police pursuit.

    python3 scripts/racing_env.py --preview
    python3 scripts/racing_env.py --episodes 300 --steps 128 --out /tmp/racing/data

The world keeps two views of the same state:

  frame 64x64 RGB  -- the pixel world a VQVAE + transformer learns to predict
  proxy 64x64 RGB  -- a wireframe that fixes road, buildings and cars

That split is the basic idea of inclusionAI/CodeWorldModel: the engine owns the
state, a structural proxy fixes what moves, and a renderer produces appearance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
HORIZON = 18
SKY = (96, 150, 205)
ROAD = (72, 74, 80)
LANE = (225, 225, 210)
BUILDINGS = [(150, 110, 95), (110, 130, 150), (165, 145, 110), (120, 120, 140)]
CAR = (235, 80, 70)
POLICE = (60, 90, 210)
LIGHT_RED = (255, 60, 60)
LIGHT_BLUE = (70, 140, 255)
ACTIONS = ["stay", "left", "right", "accelerate", "brake"]


class RacingWorld:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.distance = 0.0
        self.lateral = 0.0
        self.speed = 1.0
        self.police = [3.0, 6.0]
        self.time = 0
        self.seed = seed

    def step(self, action: int) -> None:
        action = int(action)
        if action == 1:
            self.lateral -= 1.0
        elif action == 2:
            self.lateral += 1.0
        elif action == 3:
            self.speed = min(2.5, self.speed + 0.25)
        elif action == 4:
            self.speed = max(0.4, self.speed - 0.25)
        self.lateral = float(np.clip(self.lateral, -3, 3))
        self.distance += self.speed
        self.time += 1
        for index, gap in enumerate(self.police):
            desired = max(1.0, gap + (self.speed - 1.2) * 0.5 + 0.02 * index)
            self.police[index] = float(np.clip(desired + self.rng.normal(0, 0.08), 0.7, 12.0))

    def _road(self, draw: ImageDraw.ImageDraw, proxy: bool) -> None:
        cx = 32 - self.lateral * 2
        top = (cx - 3, HORIZON, cx + 3, HORIZON)
        bottom = (cx - 30, SIZE, cx + 30, SIZE)
        if proxy:
            draw.rectangle((0, 0, SIZE - 1, SIZE - 1), fill=(0, 0, 0))
            draw.line([top[:2], bottom[:2]], fill=(220, 220, 220), width=1)
            draw.line([top[2:], bottom[2:]], fill=(220, 220, 220), width=1)
        else:
            draw.rectangle((0, 0, SIZE - 1, HORIZON - 1), fill=SKY)
            draw.polygon([(cx - 3, HORIZON), (cx + 3, HORIZON),
                          (cx + 30, SIZE), (cx - 30, SIZE)], fill=ROAD)
        # Lane dashes move toward the viewer with distance.
        for i in range(7):
            z = ((i + (self.distance * 0.12) % 1.0) / 7.0) ** 1.6
            y = int(HORIZON + 2 + z * (SIZE - HORIZON))
            half = max(1, int(1 + z * 2))
            colour = (200, 200, 200) if proxy else LANE
            draw.line([(cx - half // 2, y), (cx + half // 2, y)], fill=colour, width=max(1, int(z * 2)))

    def _buildings(self, draw: ImageDraw.ImageDraw, proxy: bool) -> None:
        cx = 32 - self.lateral * 2
        for side in (-1, 1):
            for i in range(7):
                z = ((i + (self.distance * 0.12) % 1.0) / 7.0) ** 1.6
                y0 = int(HORIZON + z * (SIZE - HORIZON))
                y1 = min(SIZE, y0 + int(2 + z * 14))
                width = int(2 + z * 12)
                height = int(3 + z * 24)
                offset = int(3 + z * 13)
                x0 = int(cx + side * offset)
                if side < 0:
                    box = (x0 - width, max(HORIZON, y0 - height), x0, y1)
                else:
                    box = (x0, max(HORIZON, y0 - height), x0 + width, y1)
                colour = (190, 190, 190) if proxy else BUILDINGS[(i + self.seed) % len(BUILDINGS)]
                draw.rectangle(box, fill=colour, outline=(30, 30, 30) if not proxy else None)
                if not proxy and z > 0.25:
                    for column in range(2):
                        for window in range(2):
                            wy = box[1] + int(2 + window * 5 * z)
                            wx = box[0] + int(1 + column * max(2, width / 3))
                            if wy + 1 < box[3] and wx + 1 < box[2]:
                                draw.rectangle((wx, wy, wx + 1, wy + 1), fill=(240, 220, 140))

    def _cars(self, draw: ImageDraw.ImageDraw, proxy: bool) -> None:
        # Player car at the bottom centre.
        px = 32
        py = 54
        if proxy:
            draw.rectangle((px - 5, py, px + 5, py + 7), outline=(255, 255, 255))
        else:
            draw.rectangle((px - 5, py, px + 5, py + 7), fill=CAR, outline=(20, 20, 20))
            draw.rectangle((px - 3, py - 2, px + 3, py + 1), fill=(90, 150, 200))
        # Police cars behind, bigger as they get closer.
        for index, gap in enumerate(self.police):
            z = float(np.clip(1.0 - gap / 12.0, 0.05, 0.95))
            y = int(HORIZON + 2 + z * (SIZE - HORIZON))
            width = max(2, int(2 + z * 6))
            x = px + (index - 0.5) * 6
            if proxy:
                draw.rectangle((x - width, y, x + width, y + max(2, int(z * 7))), outline=(255, 255, 255))
            else:
                draw.rectangle((x - width, y, x + width, y + max(2, int(z * 7))),
                               fill=POLICE, outline=(20, 20, 30))
                light = LIGHT_RED if (self.time + index) % 6 < 3 else LIGHT_BLUE
                draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=light)

    def render(self, proxy: bool = False) -> np.ndarray:
        image = Image.new("RGB", (SIZE, SIZE))
        draw = ImageDraw.Draw(image)
        self._buildings(draw, proxy)
        self._road(draw, proxy)
        self._cars(draw, proxy)
        return np.asarray(image)


def policy(world: RacingWorld, rng):
    while True:
        if rng.random() < 0.45:
            yield int(rng.integers(5))
        elif rng.random() < 0.5:
            yield 1 if world.lateral > 0 else 2
        else:
            yield 3 if world.speed < 1.8 else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/tmp/racing/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.preview:
        world = RacingWorld(7)
        policy_rng = np.random.default_rng(8)
        strip = Image.new("RGB", (8 * SIZE * 3, SIZE * 3))
        for t in range(8):
            strip.paste(Image.fromarray(world.render()).resize((SIZE * 3, SIZE * 3), Image.NEAREST),
                        (t * SIZE * 3, 0))
            for _ in range(12):
                world.step(next(policy(world, policy_rng)))
        strip.save("/tmp/opencode/racing-preview.png")
        proxy = Image.fromarray(world.render(proxy=True)).resize((SIZE * 3, SIZE * 3), Image.NEAREST)
        proxy.save("/tmp/opencode/racing-proxy.png")
        print("wrote previews")
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
