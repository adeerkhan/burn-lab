"""A top-down GTA-style city chase with real left/right turns at intersections.

    python3 scripts/racing_env.py --preview
    python3 scripts/racing_env.py --episodes 300 --steps 128 --out /tmp/city/data

The city is a grid of roads and blocks. Buildings vary in size, height, colour
and windows; the player drives on the roads and turns left or right at
intersections. Police follow the player's recorded trail, so they take the same
turns. Two 64x64 views are recorded: the pixel frame and a wireframe proxy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WORLD = 128
BLOCK = 12
ROAD = 4
VIEW = 64
SCALE = 2
CELLS = VIEW // SCALE
ACTIONS = ["stay", "left", "right", "accelerate", "brake"]
PALETTE = np.array([
    [150, 110, 95], [115, 130, 150], [170, 145, 110], [120, 120, 140],
    [140, 100, 110], [100, 140, 125], [165, 160, 150], [110, 105, 135],
], dtype=np.float32)
ROAD_COLOUR = np.array([70, 72, 78], dtype=np.uint8)
SIDEWALK = np.array([120, 122, 126], dtype=np.uint8)
WINDOW = np.array([250, 225, 150], dtype=np.uint8)


def build_city(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the city texture and its structural proxy once per episode."""
    rng = np.random.default_rng(seed)
    texture = np.zeros((WORLD, WORLD, 3), dtype=np.uint8)
    proxy = np.zeros((WORLD, WORLD, 3), dtype=np.uint8)
    for y in range(WORLD):
        for x in range(WORLD):
            road_x = x % BLOCK < ROAD
            road_y = y % BLOCK < ROAD
            on_road = road_x or road_y
            intersection = road_x and road_y
            if on_road:
                colour = ROAD_COLOUR.copy()
                if not intersection:
                    if road_x and x % BLOCK in (1, 2) and y % 3 == 0:
                        colour = np.array([220, 220, 205], dtype=np.uint8)
                    if road_y and y % BLOCK in (1, 2) and x % 3 == 0:
                        colour = np.array([220, 220, 205], dtype=np.uint8)
                if intersection and x % BLOCK in (1, 2) and y % BLOCK in (1, 2):
                    colour = np.array([195, 195, 190], dtype=np.uint8)
                texture[y, x] = colour
                proxy[y, x] = (40, 40, 40)
            else:
                bx, by = x // BLOCK, y // BLOCK
                local_x, local_y = x % BLOCK - ROAD, y % BLOCK - ROAD
                size = BLOCK - ROAD
                # Two or three buildings per block, varied by a stable hash.
                split = 4 + (bx * 7 + by * 13) % 3
                part = 0 if local_y < split else 1
                colour_index = (bx * 11 + by * 5 + part * 3 + seed) % len(PALETTE)
                colour = PALETTE[colour_index].copy()
                shade = 0.82 + 0.06 * ((bx + by + part) % 4)
                colour *= shade
                # Setbacks and shadows give a sense of height.
                if part == 1 and local_x > size - 2:
                    colour *= 0.72
                if local_y == 0 or local_x == 0:
                    colour *= 0.85
                if (local_x * 3 + local_y * 5 + bx + by) % 11 == 0 and local_x > 0 and local_y > 0:
                    colour = WINDOW.astype(np.float32)
                texture[y, x] = np.clip(colour, 0, 255).astype(np.uint8)
                proxy[y, x] = (150, 150, 150) if local_x in (0, size - 1) or local_y in (0, size - 1) else (25, 25, 25)
    # Sidewalk strips bordering the roads.
    for y in range(WORLD):
        for x in range(WORLD):
            if x % BLOCK == ROAD or x % BLOCK == BLOCK - 1 or y % BLOCK == ROAD or y % BLOCK == BLOCK - 1:
                if not (x % BLOCK < ROAD or y % BLOCK < ROAD):
                    texture[y, x] = SIDEWALK
                    proxy[y, x] = (90, 90, 90)
    return texture, proxy


class CityWorld:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed + 1)
        self.texture, self.proxy_texture = build_city(seed)
        start = ROAD // 2 + BLOCK * np.array([3, 4])
        self.x, self.y = float(start[0]), float(start[1])
        self.heading = 0  # 0 up, 1 right, 2 down, 3 left
        self.speed = 1.0
        self.trail: list[tuple[float, float, int]] = [(self.x, self.y, self.heading)]
        self.time = 0

    def _at_intersection(self) -> bool:
        return (self.x % BLOCK) < ROAD + 0.8 and (self.y % BLOCK) < ROAD + 0.8

    def step(self, action: int) -> None:
        action = int(action)
        if action == 1:
            self.heading = (self.heading - 1) % 4 if self._at_intersection() else self.heading
        elif action == 2:
            self.heading = (self.heading + 1) % 4 if self._at_intersection() else self.heading
        elif action == 3:
            self.speed = min(2.5, self.speed + 0.35)
        elif action == 4:
            self.speed = max(0.4, self.speed - 0.35)
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.heading]
        self.x = float(np.clip(self.x + dx * self.speed, ROAD / 2, WORLD - ROAD / 2 - 1))
        self.y = float(np.clip(self.y + dy * self.speed, ROAD / 2, WORLD - ROAD / 2 - 1))
        self.trail.append((self.x, self.y, self.heading))
        if len(self.trail) > 200:
            self.trail.pop(0)
        if self.rng.random() < 0.02:
            self.heading = (self.heading + int(self.rng.integers(-1, 2))) % 4
        self.time += 1

    def _car(self, image: np.ndarray, x: float, y: float, heading: int,
             colour: tuple, police: bool) -> None:
        top = Image.fromarray(image)
        draw = ImageDraw.Draw(top)
        cx, cy = int(x), int(y)
        if heading in (0, 2):
            w, h = 3 * SCALE, 5 * SCALE
        else:
            w, h = 5 * SCALE, 3 * SCALE
        draw.rectangle((cx - w, cy - h, cx + w, cy + h), fill=colour, outline=(15, 15, 20))
        draw.rectangle((cx - w + 2, cy - h + 2, cx + w - 2, cy + h - 2),
                       fill=(95, 150, 200))
        if police:
            flash = (255, 60, 60) if self.time % 6 < 3 else (70, 150, 255)
            draw.rectangle((cx - 2, cy - 2, cx + 2, cy + 2), fill=flash)
        image[:] = np.asarray(top)

    def render(self, proxy: bool = False) -> np.ndarray:
        left = int(np.clip(self.x - CELLS // 2, 0, WORLD - CELLS))
        top = int(np.clip(self.y - CELLS // 2, 0, WORLD - CELLS))
        source = self.proxy_texture if proxy else self.texture
        image = np.repeat(np.repeat(source[top:top + CELLS, left:left + CELLS], SCALE, 0), SCALE, 1)
        # Police follow the player's trail, so they take the same turns.
        for index, gap in enumerate((8, 16)):
            point = self.trail[max(0, len(self.trail) - 1 - gap * (index + 1))]
            x = (point[0] - left) * SCALE
            y = (point[1] - top) * SCALE
            if 0 <= x < VIEW and 0 <= y < VIEW:
                if proxy:
                    draw = ImageDraw.Draw(Image.fromarray(image))
                    draw.rectangle((x - 3, y - 3, x + 3, y + 3), outline=(255, 255, 255))
                    image[:] = np.asarray(Image.fromarray(image))
                else:
                    self._car(image, x, y, point[2], (50, 80, 200), True)
        px, py = (self.x - left) * SCALE, (self.y - top) * SCALE
        if proxy:
            draw = ImageDraw.Draw(Image.fromarray(image))
            draw.rectangle((px - 3, py - 4, px + 3, py + 4), outline=(255, 255, 255))
            image[:] = np.asarray(Image.fromarray(image))
        else:
            self._car(image, px, py, self.heading, (235, 70, 60), False)
        return image


def policy(world: CityWorld, rng):
    while True:
        if world._at_intersection() and rng.random() < 0.55:
            yield 1 if rng.random() < 0.5 else 2
        elif rng.random() < 0.25:
            yield int(rng.integers(5))
        elif rng.random() < 0.5:
            yield 3 if world.speed < 1.9 else 0
        else:
            yield 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/tmp/city/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.preview:
        world = CityWorld(7)
        walk = policy(world, np.random.default_rng(8))
        strip = Image.new("RGB", (8 * VIEW * 3, VIEW * 3))
        for t in range(8):
            strip.paste(Image.fromarray(world.render()).resize((VIEW * 3, VIEW * 3), Image.NEAREST),
                        (t * VIEW * 3, 0))
            for _ in range(16):
                world.step(next(walk))
        strip.save("/tmp/opencode/city-preview.png")
        print("wrote preview")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.empty((args.episodes, args.steps + 1, VIEW, VIEW, 3), dtype=np.uint8)
    proxies = np.empty_like(frames)
    actions = np.empty((args.episodes, args.steps), dtype=np.uint8)
    for episode in range(args.episodes):
        world = CityWorld(args.seed * 100003 + episode)
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
        "episodes": args.episodes, "steps": args.steps, "size": VIEW, "actions": ACTIONS,
    }))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
