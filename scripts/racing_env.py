"""Vice City: a third-person sunset chase with palms and neon.

    python3 scripts/racing_env.py --preview
    python3 scripts/racing_env.py --episodes 600 --steps 160 --out /tmp/vice/data

A low sun sits at the end of the boulevard, palm trees line the road, towers
have wildly different heights and neon signs, and the player car is pursued by
police with flashing light bars. Two 64x64 views are recorded: the pixel frame
and a wireframe proxy of the same structure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 64
S = 2
OUT = SIZE * S
HORIZON = 12
ROAD = (58, 60, 66)
SIDEWALK = (105, 100, 105)
LANE = (235, 225, 195)
BUILDINGS = [
    (120, 118, 132), (150, 135, 120), (105, 128, 150), (140, 110, 115),
    (160, 150, 140), (115, 138, 125), (145, 128, 155), (130, 145, 160),
]
NEON = [(255, 90, 190), (90, 230, 255), (255, 200, 70), (150, 255, 130)]
FLASH_RED = (255, 60, 60)
FLASH_BLUE = (70, 150, 255)
ACTIONS = ["stay", "left", "right", "accelerate", "brake"]


class ScaledDraw:
    """Draw in the old 64x64 coordinate space onto a 128x128 image."""

    def __init__(self, image):
        self.image = image
        self.draw = ImageDraw.Draw(image)
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf", 8 * S)
        except Exception:
            self.font = ImageFont.load_default()

    def _box(self, box):
        return tuple(int(v * S) for v in box)

    def _points(self, points):
        return [(int(x * S), int(y * S)) for x, y in points]

    def rectangle(self, box, **kwargs):
        self.draw.rectangle(self._box(box), **kwargs)

    def polygon(self, xy, **kwargs):
        self.draw.polygon(self._points(xy), **kwargs)

    def line(self, xy, **kwargs):
        self.draw.line(self._points(xy), **kwargs)

    def ellipse(self, box, **kwargs):
        self.draw.ellipse(self._box(box), **kwargs)

    def text(self, xy, text, **kwargs):
        kwargs.setdefault("font", self.font)
        self.draw.text((int(xy[0] * S), int(xy[1] * S)), text, **kwargs)


class ViceWorld:
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
            self.turn_phase = max(0.0, self.turn_phase - 0.1)
        self.distance += self.speed
        self.time += 1
        self.intersection = (self.distance % 34) < 1.6
        for index, gap in enumerate(self.police):
            target = 2.4 + (self.speed - 1.0) * 3.0 + abs(self.turn_direction) * self.turn_phase
            target += index * 2.6 + self.rng.normal(0, 0.05)
            target = float(np.clip(target, 1.8, 16.0))
            self.police[index] = gap * 0.78 + target * 0.22

    def _vanish(self) -> float:
        strength = np.sin(self.turn_phase * np.pi) if self.turn_phase > 0 else 0.0
        return 32.0 + self.turn_direction * strength * 20.0

    @staticmethod
    def _lerp(a, b, t):
        return a * (1 - t) + b * t

    def road_centre(self, y: int) -> float:
        t = min(1.0, max(0.0, (y - HORIZON) / max(1, SIZE - HORIZON)))
        return self._lerp(32.0, self._vanish(), (1 - t) ** 1.5)

    def road_half(self, y: int) -> float:
        t = min(1.0, max(0.0, (y - HORIZON) / max(1, SIZE - HORIZON)))
        return 2.0 + t * 31.0

    def _sky(self, draw, proxy: bool) -> None:
        if proxy:
            draw.rectangle((0, 0, SIZE - 1, SIZE - 1), fill=(0, 0, 0))
            return
        for y in range(HORIZON):
            t = y / max(1, HORIZON)
            draw.line([(0, y), (SIZE, y)], fill=(int(255 - 30 * t), int(150 + 60 * t), int(95 + 55 * t)))
        # Low sun at the end of the boulevard.
        sun_x = int(self._vanish())
        for radius, colour in ((7, (255, 170, 90)), (5, (255, 205, 120)), (3, (255, 240, 190))):
            draw.ellipse((sun_x - radius, HORIZON - radius, sun_x + radius, HORIZON + radius),
                         fill=colour)
        for i in range(14):
            x = i * 5
            height = 2 + (i * 7 + self.seed) % 8
            draw.rectangle((x, HORIZON - height, x + 4, HORIZON), fill=(140, 110, 105))

    def _road(self, draw, proxy: bool) -> None:
        ct, cb = self.road_centre(HORIZON), self.road_centre(SIZE)
        ht, hb = self.road_half(HORIZON), self.road_half(SIZE)
        if proxy:
            draw.line([(ct - ht, HORIZON), (cb - hb, SIZE)], fill=(225, 225, 225))
            draw.line([(ct + ht, HORIZON), (cb + hb, SIZE)], fill=(225, 225, 225))
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

    def _buildings(self, draw, proxy: bool) -> None:
        for side in (-1, 1):
            for i in range(8):
                phase = (i + (self.distance * 0.1) % 1.0) / 8.0
                depth = phase ** 1.5
                y0 = int(HORIZON + depth * (SIZE - HORIZON))
                c = self.road_centre(y0)
                half = self.road_half(y0)
                seed = (i * 17 + self.seed + (side + 1) * 29) % 23
                height = int(6 + depth * 24 + (seed % 7) * 5 + (seed // 7) * 4)
                width = int(5 + depth * 12 + seed % 5)
                x_edge = c + side * (half + 2)
                top = max(2, y0 - height)
                box = (x_edge if side > 0 else x_edge - width, top,
                       x_edge + width if side > 0 else x_edge, min(SIZE, y0 + 3))
                colour = (195, 195, 195) if proxy else BUILDINGS[(i + seed) % len(BUILDINGS)]
                if not proxy and depth < 0.4:
                    colour = tuple(int(v * 0.82) for v in colour)
                draw.rectangle(box, fill=colour, outline=(30, 30, 36) if not proxy else None)
                if not proxy:
                    rows = max(2, int(height / 6))
                    cols = max(1, width // 4)
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
                    # Neon sign on some facades.
                    if seed % 4 == 0 and depth > 0.2:
                        neon = NEON[(i + self.seed) % len(NEON)]
                        ny = top + 4 + (seed % 3) * 6
                        if ny + 4 < box[3] and box[2] - box[0] > 9:
                            draw.rectangle((box[0] + 1, ny, box[2] - 1, ny + 4), fill=(35, 25, 45))
                            draw.text((box[0] + 2, ny), "VICE CITY", fill=neon)
                    if depth > 0.5 and seed % 3 == 0:
                        draw.line([(box[0] + 2, top), (box[0] + 2, top - 3)], fill=(60, 60, 65))

    def _palms(self, draw, proxy: bool) -> None:
        for side in (-1, 1):
            for i in range(6):
                phase = (i + (self.distance * 0.1) % 1.0) / 6.0
                depth = float(phase) ** 1.5
                y = int(HORIZON + 2 + depth * (SIZE - HORIZON))
                c = self.road_centre(y)
                half = self.road_half(y)
                x = int(c + side * (half + 1.5))
                height = int(2 + depth * 9)
                colour = (220, 220, 220) if proxy else (95, 75, 60)
                draw.line([(x, y), (x, y - height)], fill=colour, width=max(1, int(1 + depth)))
                leaf = (220, 220, 220) if proxy else (45, 130, 80)
                spread = max(2, int(2 + depth * 5))
                draw.line([(x - spread, y - height), (x, y - height - 2), (x + spread, y - height)],
                          fill=leaf, width=1)

    def _car(self, draw, x: float, y: int, width: int, height: int,
             colour, police: bool, proxy: bool, flash: bool = False) -> None:
        left, right = x - width, x + width
        top = y - height
        if proxy:
            draw.rectangle((left, top, right, y), outline=(255, 255, 255))
            return
        draw.ellipse((left - 2, y - 2, right + 2, y + 3), fill=(20, 20, 24))
        draw.rectangle((left, top + 2, right, y - 1), fill=colour, outline=(12, 12, 18))
        draw.polygon([(left + 1, top + 2), (left + 3, top), (right - 3, top), (right - 1, top + 2)],
                     fill=tuple(int(v * 0.75) for v in colour))
        draw.rectangle((left + 2, top, right - 2, top + 2), fill=(80, 130, 175) if not police else (40, 70, 140))
        draw.rectangle((left, y - 3, left + 2, y - 1), fill=(255, 90, 70))
        draw.rectangle((right - 2, y - 3, right, y - 1), fill=(255, 90, 70))
        if y - 3 >= top + 4:
            draw.rectangle((left - 1, top + 4, left, y - 3), fill=(15, 15, 20))
            draw.rectangle((right, top + 4, right + 1, y - 3), fill=(15, 15, 20))
        if police:
            draw.rectangle((x - 2, top - 2, x + 2, top - 1), fill=FLASH_RED if flash else FLASH_BLUE)
            draw.line([(x - 3, y), (x - 3, y + 2)], fill=FLASH_RED if flash else FLASH_BLUE)

    def _player(self, draw, proxy: bool) -> None:
        x = self.road_centre(49) + self.turn_direction * self.turn_phase * 3
        self._car(draw, x, 49, 6, 8, (235, 70, 60), False, proxy)

    def _police(self, draw, proxy: bool) -> None:
        for index, gap in enumerate(self.police):
            depth = float(np.clip(1.0 - gap / 16.0, 0.02, 0.95))
            y = int(63 - depth * 6)
            width = max(4, int(4 + depth * 2))
            height = max(5, int(5 + depth * 3))
            lane = -1 if index % 2 == 0 else 1
            x = self.road_centre(y) + lane * (width + 3) + (index // 2) * 2
            self._car(draw, x, y, width, height, (45, 75, 185), True, proxy,
                      (self.time + index) % 8 < 4)

    def render(self, proxy: bool = False) -> np.ndarray:
        image = Image.new("RGB", (OUT, OUT))
        draw = ScaledDraw(image)
        self._sky(draw, proxy)
        self._road(draw, proxy)
        self._buildings(draw, proxy)
        self._palms(draw, proxy)
        self._police(draw, proxy)
        self._player(draw, proxy)
        return np.asarray(image)


def policy(world: ViceWorld, rng):
    while True:
        if world.turn_phase > 0:
            yield 0
        elif world.intersection and rng.random() < 0.55:
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
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--out", type=Path, default=Path("/tmp/vice/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.preview:
        world = ViceWorld(7)
        walk = policy(world, np.random.default_rng(8))
        strip = Image.new("RGB", (8 * OUT, OUT))
        for t in range(8):
            strip.paste(Image.fromarray(world.render()),
                        (t * OUT, 0))
            for _ in range(18):
                world.step(next(walk))
        strip.save("/tmp/opencode/vice-preview.png")
        print("wrote preview")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.empty((args.episodes, args.steps + 1, OUT, OUT, 3), dtype=np.uint8)
    proxies = np.empty_like(frames)
    actions = np.empty((args.episodes, args.steps), dtype=np.uint8)
    for episode in range(args.episodes):
        world = ViceWorld(args.seed * 100003 + episode)
        walk = policy(world, np.random.default_rng(episode + 1))
        frames[episode, 0] = world.render()
        proxies[episode, 0] = world.render(proxy=True)
        for t in range(args.steps):
            action = next(walk)
            actions[episode, t] = action
            world.step(action)
            frames[episode, t + 1] = world.render()
            proxies[episode, t + 1] = world.render(proxy=True)
        if episode % 100 == 0:
            print(f"episode {episode}", flush=True)
    (args.out / "frames.bin").write_bytes(np.ascontiguousarray(frames).reshape(-1).tobytes())
    (args.out / "proxy.bin").write_bytes(np.ascontiguousarray(proxies).reshape(-1).tobytes())
    (args.out / "actions.bin").write_bytes(np.ascontiguousarray(actions).tobytes())
    (args.out / "meta.json").write_text(json.dumps({
        "episodes": args.episodes, "steps": args.steps, "size": OUT, "actions": ACTIONS,
    }))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
