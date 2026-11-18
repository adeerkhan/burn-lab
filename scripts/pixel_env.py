"""A 32x32 BSP dungeon rendered at 2 px/cell (64x64 output) for the pixel model.

    python3 scripts/pixel_env.py --episodes 300 --steps 128

This is the dataset for the VQVAE + transformer path: rooms and corridors are
readable at 2 px walls, while the frame stays 64x64. Frames and actions use the
same flat u8 layout the Rust pipeline reads.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

GRID = 32
SCALE = 2
DELTA = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
FLOOR = np.array([30, 36, 48], dtype=np.float32)
WALL = np.array([176, 160, 136], dtype=np.float32)
WARM = np.array([1.0, 0.72, 0.38], dtype=np.float32)


def generate_rooms(rng: np.random.Generator):
    grid = np.ones((GRID, GRID), dtype=np.uint8)
    rooms = []

    def split(x, y, width, height, depth):
        if width <= 7 or height <= 7:
            rw, rh = max(1, width - 2), max(1, height - 2)
            grid[y + 1:y + 1 + rh, x + 1:x + 1 + rw] = 0
            rooms.append((x + 1, y + 1, rw, rh))
            return
        if depth == 0 or width < 9 or height < 9:
            rw = int(rng.integers(4, min(width - 2, 9) + 1))
            rh = int(rng.integers(4, min(height - 2, 9) + 1))
            rx = int(rng.integers(x + 1, x + width - rw))
            ry = int(rng.integers(y + 1, y + height - rh))
            grid[ry:ry + rh, rx:rx + rw] = 0
            rooms.append((rx, ry, rw, rh))
            return
        if width >= height:
            cut = int(rng.integers(width // 3, 2 * width // 3))
            split(x, y, cut, height, depth - 1)
            split(x + cut, y, width - cut, height, depth - 1)
        else:
            cut = int(rng.integers(height // 3, 2 * height // 3))
            split(x, y, width, cut, depth - 1)
            split(x, y + cut, width, height - cut, depth - 1)

    split(1, 1, GRID - 2, GRID - 2, 3)
    for (ax, ay, aw, ah), (bx, by, bw, bh) in zip(rooms, rooms[1:]):
        axc, ayc = ax + aw // 2, ay + ah // 2
        bxc, byc = bx + bw // 2, by + bh // 2
        if rng.random() < 0.5:
            grid[ayc, min(axc, bxc):max(axc, bxc) + 1] = 0
            grid[min(ayc, byc):max(ayc, byc) + 1, bxc] = 0
        else:
            grid[min(ayc, byc):max(ayc, byc) + 1, axc] = 0
            grid[byc, min(axc, bxc):max(axc, bxc) + 1] = 0
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    return grid, rooms


def component(grid, start):
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DELTA[1:]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID and 0 <= ny < GRID and grid[ny, nx] == 0 and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return seen


def bfs_step(grid, start, goal):
    q = deque([(start, -1)])
    previous = {}
    while q:
        cell, first = q.popleft()
        if cell == goal:
            return max(first, 0)
        for action, (dx, dy) in enumerate(DELTA[1:], start=1):
            nxt = (cell[0] + dx, cell[1] + dy)
            if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and grid[nxt[1], nxt[0]] == 0 and nxt not in previous:
                previous[nxt] = cell
                q.append((nxt, first if first >= 0 else action))
    return 0


class PixelDungeon:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.rng = rng
        for _ in range(12):
            self.grid, self.rooms = generate_rooms(rng)
            floor = [(x, y) for y in range(GRID) for x in range(GRID) if self.grid[y, x] == 0]
            start = floor[int(rng.integers(len(floor)))]
            region = component(self.grid, start)
            if len(region) >= 180:
                break
        self.region = region
        self.start = start
        self.player = start
        far = [c for c in region if abs(c[0] - start[0]) + abs(c[1] - start[1]) > GRID // 2]
        self.goal = far[int(rng.integers(len(far)))] if far else start
        spots = [c for c in region if abs(c[0] - start[0]) + abs(c[1] - start[1]) > 4]
        rng.shuffle(spots)
        self.enemies = spots[:2]
        self.diamonds = spots[2:7]
        self.hits = 0
        self.caught = 0
        self.collected = 0
        yy, xx = np.mgrid[0:GRID * SCALE, 0:GRID * SCALE]
        self.px = xx.astype(np.float32)
        self.py = yy.astype(np.float32)

    def _free(self, cell):
        x, y = cell
        return 0 <= x < GRID and 0 <= y < GRID and self.grid[y, x] == 0

    def step(self, action: int):
        action = int(action)
        dx, dy = DELTA[action]
        nxt = (self.player[0] + dx, self.player[1] + dy)
        if self._free(nxt):
            self.player = nxt
        else:
            self.hits += 1
        if self.player in self.diamonds:
            self.diamonds.remove(self.player)
            self.collected += 1
        for index, (x, y) in enumerate(self.enemies):
            gap = abs(x - self.player[0]) + abs(y - self.player[1])
            if gap <= 6 and self.rng.random() < 0.7:
                mx = int(np.sign(self.player[0] - x))
                my = int(np.sign(self.player[1] - y))
                for move in ((x + mx, y), (x, y + my)):
                    if self._free(move):
                        self.enemies[index] = move
                        x, y = move
            elif self.rng.random() < 0.4:
                for mx, my in DELTA[1:]:
                    if self._free((x + mx, y + my)) and self.rng.random() < 0.5:
                        self.enemies[index] = (x + mx, y + my)
                        break
            if self.enemies[index] == self.player:
                self.player = self.start
                self.caught += 1
        return self.render()

    def _blit(self, image, cell, colour, core=None):
        x, y = cell[0] * SCALE, cell[1] * SCALE
        image[max(0, y):y + 4, max(0, x):x + 4] = colour
        if core is not None:
            image[y + 1:y + 3, x + 1:x + 3] = core

    def render(self):
        wx = (self.px // SCALE).astype(np.int32)
        wy = (self.py // SCALE).astype(np.int32)
        cells = self.grid[wy, wx]
        variation = (((wx * 7 + wy * 13) % 9 - 4)[..., None]) * 1.4
        image = np.where(cells[..., None] == 0, FLOOR + variation, WALL).astype(np.float32)
        sx = self.player[0] * SCALE + SCALE / 2
        sy = self.player[1] * SCALE + SCALE / 2
        distance = np.hypot(self.px + 0.5 - sx, self.py + 0.5 - sy)
        torch = 0.16 + 0.84 * np.exp(-np.square(distance / 18.0))
        image = image * torch[..., None] + (WARM * 48.0) * (torch * torch)[..., None]
        for diamond in self.diamonds:
            self._blit(image, diamond, np.array([70, 220, 255], dtype=np.float32),
                       np.array([235, 255, 255], dtype=np.float32))
        for enemy in self.enemies:
            self._blit(image, enemy, np.array([250, 70, 82], dtype=np.float32),
                       np.array([120, 12, 20], dtype=np.float32))
        self._blit(image, self.goal, np.array([60, 250, 140], dtype=np.float32),
                   np.array([225, 255, 230], dtype=np.float32))
        self._blit(image, self.player, np.array([255, 214, 92], dtype=np.float32),
                   np.array([255, 250, 220], dtype=np.float32))
        return np.clip(image, 0, 255).astype(np.uint8)


def policy(dungeon: PixelDungeon, rng):
    while True:
        if dungeon.enemies:
            nearest = min(dungeon.enemies,
                          key=lambda e: abs(e[0] - dungeon.player[0]) + abs(e[1] - dungeon.player[1]))
            gap = abs(nearest[0] - dungeon.player[0]) + abs(nearest[1] - dungeon.player[1])
            if gap <= 3 and rng.random() < 0.7:
                dx = int(np.sign(dungeon.player[0] - nearest[0]))
                dy = int(np.sign(dungeon.player[1] - nearest[1]))
                for action, (ddx, ddy) in enumerate(DELTA[1:], start=1):
                    if (ddx, ddy) in ((dx, 0), (0, dy)) and dungeon._free(
                            (dungeon.player[0] + ddx, dungeon.player[1] + ddy)):
                        yield action
                        break
                else:
                    yield int(rng.integers(5))
                continue
        targets = dungeon.diamonds or [dungeon.goal]
        target = min(targets, key=lambda c: abs(c[0] - dungeon.player[0]) + abs(c[1] - dungeon.player[1]))
        yield bfs_step(dungeon.grid, dungeon.player, target) if rng.random() < 0.5 else int(rng.integers(5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/pixel/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.empty((args.episodes, args.steps + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((args.episodes, args.steps), dtype=np.uint8)
    for episode in range(args.episodes):
        seed = args.seed * 100003 + episode
        dungeon = PixelDungeon(seed)
        rng = np.random.default_rng(seed + 1)
        walk = policy(dungeon, rng)
        frames[episode, 0] = dungeon.render()
        for t in range(args.steps):
            action = next(walk)
            actions[episode, t] = action
            frames[episode, t + 1] = dungeon.step(action)
        if episode % 50 == 0:
            print(f"episode {episode}", flush=True)
    flat = np.ascontiguousarray(frames).reshape(-1)
    (args.out / "frames.bin").write_bytes(flat.tobytes())
    (args.out / "actions.bin").write_bytes(np.ascontiguousarray(actions).tobytes())
    (args.out / "meta.json").write_text(json.dumps({
        "episodes": args.episodes, "steps": args.steps, "size": 64,
        "actions": ["stay", "up", "down", "left", "right"],
    }))
    print(f"wrote {args.out}: {frames.shape}, {flat.size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
