"""Third-person Manhattan chase with real corners, towering blocks and police.

    python3 scripts/racing_env.py --preview
    python3 scripts/racing_env.py --episodes 300 --steps 128 --out /tmp/chase2/data

Road corners are drawn as a shifting vanishing point with a cross street, not a
straight diagonal. Buildings vary wildly in height and style. Police approach
smoothly from behind with flashing bars and a glow on the road.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
HORIZON = 12
ROAD = (62, 64, 70)
SIDEWALK = (100, 103, 108)
LANE = (230, 228, 205)
SKY = (245, 178, 120)
BUILDINGS = [
    (125, 122, 130), (150, 138, 120), (105, 128, 150), (140, 110, 108),
    (160, 152, 140), (115, 138, 125), (145, 128, 150), (130, 145, 158),
]
ACTIONS = ["stay", "left", "right", "accelerate", "brake"]
FLASH_RED = (255, 60, 60)
FLASH_BLUE = (70, 150, 255)


class ChaseWorld:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed + 1)
        self.seed = seed
        self.distance = 0.0
        self.speed = 1.4
        self.turn_phase = 0.0
        self.turn_direction = 0
        self.intersection = False
        self.police = [5.0, 9.0]
        self.time = 0

    def step(self, action: int) -> None:
        action = int(action)
        if self.intersection and action in (1, 2) and self.turn_phase == 0:
            self.turn_direction = -1 if action == 1 else 1
            self.turn_phase = 1.0
        if action == 3:
            self.speed = min(2.7, self.speed + 0.35)
        elif action == 4:
            self.speed = max(0.4, self.speed - 0.35)
        if self.turn_phase > 0:
            self.turn_phase = max(0.0, self.turn_phase - 0.11)
        self.distance += self.speed
        self.time += 1
        self.intersection = (self.distance % 30) < 1.6
        for index, gap in enumerate(self.police):
            target = 1.0 + (self.speed - 1.0) * 3.4 + abs(self.turn_direction) * 1.4 * self.turn_phase
            target += index * 2.0 + self.rng.normal(0, 0.08)
            target = float(np.clip(target, 0.8, 16.0))
            self.police[index] = gap * 0.78 + target * 0.22

    def _vanish(self) -> float:
        # A corner shifts the vanishing point to the side; it eases back when
        # the turn is complete, so the road curves rather than shearing.
        strength = np.sin(self.turn_phase * np.pi) if self.turn_phase > 0 else 0.0
        return 32.0 + self.turn_direction * strength * 22.0

    @staticmethod
    def _lerp(a, b, t):
        return a * (1 - t) + b * t

    def road_centre(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        v = self._vanish()
        return self._lerp(32.0, v, (1 - t) ** 1.5)

    def road_half(self, y: int) -> float:
        t = (y - HORIZON) / max(1, SIZE - HORIZON)
        return 2.0 + t * 31.0

    def _sky(self, draw, proxy: bool) -> None:
        if proxy:
            draw.rectangle((0, 0, SIZE - 1, SIZE - 1), fill=(0, 0, 0))
            return
        for y in range(HORIZON):
            t = y / max(1, HORIZON)
            draw.line([(0, y), (SIZE, y)], fill=(int(235 + 10 * t), int(165 + 15 * t), int(120 + 20 * t)))
        for i in range(16):
            x = i * 4
            height = 3 + (i * 7 + self.seed) % 11
            draw.rectangle((x, HORIZON - height, x + 3, HORIZON),
                           fill=(105, 102, 112), outline=(70, 68, 76))

    def _road(self, draw, proxy: bool) -> None:
        ct, cb = self.road_centre(HORIZON), self.road_centre(SIZE)
        ht, hb = self.road_half(HORIZON), self.road_half(SIZE)
        if proxy:
            draw.line([(ct - ht, HORIZON), (cb - hb, SIZE)], fill=(225, 225, 225))
            draw.line([(ct + ht, HORIZON), (cb + hb, SIZE)], fill=(225, 225, 225))
            if self.intersection:
                draw.line([(0, HORIZON + 5), (SIZE, HORIZON + 5)], fill=(180, 180, 180))
        else:
            draw.polygon([(ct - ht - 3, HORIZON), (ct + ht + 3, HORIZON),
                          (cb + hb + 4, SIZE), (cb - hb - 4, SIZE)], fill=SIDEWALK)
            draw.polygon([(ct - ht, HORIZON), (ct + ht, HORIZON),
                          (cb + hb, SIZE), (cb - hb, SIZE)], fill=ROAD)
        for lane in (-1, 1):
            for i in range(9):
                phase = (i + (self.distance * 0.1) % 1.0) / 9.0
                y = int(HORIZON + 1 + (phase ** 1.7) * (SIZE - HORIZON))
                c = self.road_centre(y)
                offset = int(lane * (3 + phase * 10))
                mark = max(1, int(1 + phase * 2))
                draw.line([(c + offset, y - mark), (c + offset, y + mark)],
                          fill=(210, 208, 195) if proxy else LANE, width=1)
        if self.intersection and not proxy:
            y = HORIZON + 5
            c = self.road_centre(y)
            for x in range(int(c - 12), int(c + 12), 4):
                draw.line([(x, y), (x + 2, y + 2)], fill=(235, 235, 225), width=1)

    def _buildings(self, draw, proxy: bool) -> None:
        for side in (-1, 1):
            for i in range(8):
                phase = (i + (self.distance * 0.1) % 1.0) / 8.0
                depth = phase ** 1.5
                y0 = int(HORIZON + depth * (SIZE - HORIZON))
                c = self.road_centre(y0)
                half = self.road_half(y0)
                # Wildly varied heights: some mid-rise, some towers.
                seed = (i * 17 + self.seed + (side + 1) * 31) % 19
                height = int(10 + depth * 38 + seed * 2.2)
                width = int(5 + depth * 12 + seed % 4)
                x_edge = c + side * (half + 2)
                top = max(1, y0 - height)
                box = (x_edge if side > 0 else x_edge - width, top,
                       x_edge + width if side > 0 else x_edge, min(SIZE, y0 + 3))
                colour = (195, 195, 195) if proxy else BUILDINGS[(i + seed) % len(BUILDINGS)]
                if not proxy and depth < 0.4:
                    colour = tuple(int(v * 0.82) for v in colour)
                draw.rectangle(box, fill=colour, outline=(30, 30, 36) if not proxy else None)
                if not proxy:
                    # Facade rows and varied lit windows.
                    rows = max(2, int(height / 6))
                    cols = max(1, int(width / 4))
                    for row in range(rows):
                        wy = top + 3 + row * 5
                        if wy > box[3] - 2:
                            break
                        for col in range(cols):
                            wx = box[0] + 2 + col * 4
                            if wx + 1 >= box[2]:
                                continue
                            lit = (row * 5 + col * 3 + i + seed) % 9 < 4
                            draw.rectangle((wx, wy, wx + 1, wy + 1),
                                           fill=(255, 232, 170) if lit else (55, 65, 80))
                    if depth > 0.55 and seed % 3 == 0:
                        draw.rectangle((box[0] + 1, top + 1, box[2] - 1, top + 3),
                                       fill=(90, 95, 110))
                        draw.line([(box[0] + 2, top), (box[0] + 2, top - 4)], fill=(60, 60, 65))

    def _glow(self, draw, x: float, y: int, colour) -> None:
        for radius in (7, 5, 3):
            shade = tuple(int(v * (0.6 + radius * 0.05)) for v in colour)
            draw.ellipse((x - radius, y - radius // 2, x + radius, y + radius // 2), outline=shade)

    def _car(self, draw, x: float, y: int, width: int, height: int,
             colour, police: bool, proxy: bool, flash: bool, depth: float = 0.5) -> None:
        left, right = x - width, x + width
        top = y - height
        if proxy:
            draw.rectangle((left, top, right, y), outline=(255, 255, 255))
            return
        draw.ellipse((left - 2, y - 2, right + 2, y + 3), fill=(20, 20, 24))
        draw.rectangle((left, top + 2, right, y - 1), fill=colour, outline=(15, 15, 20))
        draw.polygon([(left + 1, top + 2), (left + 3, top), (right - 3, top), (right - 1, top + 2)],
                     fill=tuple(int(v * 0.75) for v in colour))
        draw.rectangle((left + 2, top, right - 2, top + 2), fill=(80, 130, 175) if not police else (40, 70, 140))
        draw.rectangle((left, y - 3, left + 2, y - 1), fill=(255, 90, 70))
        draw.rectangle((right - 2, y - 3, right, y - 1), fill=(255, 90, 70))
        if police:
            draw.rectangle((x - 2, top - 1, x + 2, top), fill=FLASH_RED if flash else FLASH_BLUE)
            if depth > 0.5:
                self._glow(draw, x, top, FLASH_RED if flash else FLASH_BLUE)

    def _player(self, draw, proxy: bool) -> None:
        x = self.road_centre(48) + (self.turn_direction * self.turn_phase * 3)
        self._car(draw, x, 48, 6, 8, (225, 60, 52), False, proxy, False)

    def _police(self, draw, proxy: bool) -> None:
        for index, gap in enumerate(self.police):
            depth = float(np.clip(1.0 - gap / 16.0, 0.02, 0.95))
            y = int(62 - depth * 8)
            width = max(2, int(2 + depth * 4))
            height = max(2, int(2 + depth * 5))
            x = self.road_centre(y) + (index - 0.5) * (width + 4)
            self._car(draw, x, y, width, height, (45, 75, 185), True, proxy,
                      (self.time + index) % 8 < 4, depth)

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
        if world.turn_phase > 0:
            yield 0
        elif world.intersection and rng.random() < 0.6:
            yield 1 if rng.random() < 0.5 else 2
        elif rng.random() < 0.22:
            yield int(rng.integers(5))
        elif rng.random() < 0.5:
            yield 3 if world.speed < 2.1 else 0
        else:
            yield 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/tmp/chase2/data"))
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
        strip.save("/tmp/opencode/chase2-preview.png")
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
