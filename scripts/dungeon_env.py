"""Cellular-automata dungeon with a torch-lit navigator for burn-lab.

    python3 scripts/dungeon_env.py --episodes 300 --steps 128

A 64x64 cave carved by the classic 4-5 rule, rendered one pixel per cell so the
map is four times larger than the earlier 32x32 version at the same 64x64
output. The player carries a warm torch, collects cyan diamonds, avoids red
enemies that chase when close, and can head for a green exit. Actions are the
same five ids as the FireflyRoom, and the output files keep the layout the Rust
world model already reads:

    frames.bin   uint8, episode*(steps+1) + time, 3*64*64 interleaved bytes
    actions.bin  uint8, episode*steps + time
    meta.json    episode count, step count, action names
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

DIR_NAMES = ["stay", "up", "down", "left", "right"]
DELTA = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
GRID = 64
SCALE = 1
FLOOR = np.array([40, 46, 62], dtype=np.float32)
WALL = np.array([158, 144, 124], dtype=np.float32)
WARM = np.array([1.0, 0.72, 0.38], dtype=np.float32)


def generate(grid_rng: np.random.Generator) -> np.ndarray:
    """Classic cave CA: random fill then four iterations of the 4-5 rule."""
    grid = (grid_rng.random((GRID, GRID)) < 0.46).astype(np.uint8)
    for _ in range(4):
        padded = np.pad(grid, 1, mode="constant", constant_values=1)
        neighbours = sum(
            padded[1 + dy:1 + dy + GRID, 1 + dx:1 + dx + GRID]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
        ) + grid
        grid = (neighbours >= 5).astype(np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    return grid


def component(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
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


def bfs_step(grid: np.ndarray, start, goal) -> int:
    """Action id that starts the shortest path, or 0 if unreachable."""
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


class Dungeon:
    def __init__(self, seed: int, n_enemies: int = 3, n_diamonds: int = 5):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.grid = generate(rng)
        floor = [(x, y) for y in range(GRID) for x in range(GRID) if self.grid[y, x] == 0]
        if len(floor) < 120:
            raise RuntimeError("degenerate dungeon")
        start = floor[int(rng.integers(len(floor)))]
        self.start = start
        reachable = component(self.grid, start)
        candidates = [c for c in reachable if abs(c[0] - start[0]) + abs(c[1] - start[1]) > GRID // 2]
        self.goal = candidates[int(rng.integers(len(candidates)))] if candidates else start
        self.player = start
        far = [c for c in reachable if abs(c[0] - start[0]) + abs(c[1] - start[1]) > 10]
        far = far or list(reachable)
        rng.shuffle(far)
        self.enemies = far[: min(n_enemies, len(far))]
        self.diamonds = far[len(self.enemies): len(self.enemies) + n_diamonds]
        self.hits = 0
        self.caught = 0
        self.collected = 0
        yy, xx = np.mgrid[0:GRID, 0:GRID]
        self.xs = xx.astype(np.float32)
        self.ys = yy.astype(np.float32)

    def _free(self, cell) -> bool:
        x, y = cell
        return 0 <= x < GRID and 0 <= y < GRID and self.grid[y, x] == 0

    def step(self, action: int) -> np.ndarray:
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
        if self.player == self.goal:
            options = [c for c in component(self.grid, self.player)
                       if abs(c[0] - self.player[0]) + abs(c[1] - self.player[1]) > GRID // 2]
            self.goal = options[int(self.rng.integers(len(options)))] if options else self.player

        for index, enemy in enumerate(self.enemies):
            distance = abs(enemy[0] - self.player[0]) + abs(enemy[1] - self.player[1])
            if distance <= 10 and self.rng.random() < 0.72:
                dx = int(np.sign(self.player[0] - enemy[0]))
                dy = int(np.sign(self.player[1] - enemy[1]))
                choices = [(enemy[0] + dx, enemy[1]), (enemy[0], enemy[1] + dy)]
                choices = [c for c in choices if self._free(c)]
                if choices:
                    enemy = choices[int(self.rng.integers(len(choices)))]
            elif self.rng.random() < 0.55:
                moves = [(enemy[0] + ddx, enemy[1] + ddy) for ddx, ddy in DELTA[1:]]
                moves = [c for c in moves if self._free(c)]
                if moves:
                    enemy = moves[int(self.rng.integers(len(moves)))]
            self.enemies[index] = enemy
            if enemy == self.player:
                self.player = self.start
                self.caught += 1
        return self.render()

    def _blit(self, image: np.ndarray, cell, colour, core=None) -> None:
        x, y = cell
        x0, x1 = max(0, x - 1), min(GRID, x + 2)
        y0, y1 = max(0, y - 1), min(GRID, y + 2)
        image[y0:y1, x0:x1] = colour
        if core is not None and 0 <= x < GRID and 0 <= y < GRID:
            image[y, x] = core

    def render(self) -> np.ndarray:
        cells = self.grid
        variation = (((self.xs * 7 + self.ys * 13).astype(np.int32)) % 9 - 4)[..., None] * 1.6
        image = np.where(cells[..., None] == 0, FLOOR + variation, WALL).astype(np.float32)

        px, py = self.player
        distance = np.hypot(self.xs - px, self.ys - py)
        torch = 0.24 + 0.76 * np.exp(-np.square(distance / 22.0))
        image = image * torch[..., None] + (WARM * 54.0) * (torch * torch)[..., None]

        for diamond in self.diamonds:
            glow = np.exp(-np.square(np.hypot(self.xs - diamond[0], self.ys - diamond[1]) / 2.6))
            image += glow[..., None] * np.array([40, 150, 190], dtype=np.float32) * 0.5
            self._blit(image, diamond, np.array([70, 220, 255], dtype=np.float32),
                       np.array([235, 255, 255], dtype=np.float32))
        for enemy in self.enemies:
            glow = np.exp(-np.square(np.hypot(self.xs - enemy[0], self.ys - enemy[1]) / 2.4))
            image += glow[..., None] * np.array([190, 30, 40], dtype=np.float32) * 0.45
            self._blit(image, enemy, np.array([250, 70, 82], dtype=np.float32),
                       np.array([120, 12, 20], dtype=np.float32))
        self._blit(image, self.goal, np.array([60, 250, 140], dtype=np.float32),
                   np.array([225, 255, 230], dtype=np.float32))
        self._blit(image, self.player, np.array([255, 214, 92], dtype=np.float32),
                   np.array([255, 250, 220], dtype=np.float32))
        return np.clip(image, 0, 255).astype(np.uint8)


def navigate(dungeon: Dungeon, rng: np.random.Generator):
    """Collect diamonds, dodge enemies, then head for the exit."""
    while True:
        if dungeon.enemies:
            nearest = min(dungeon.enemies,
                          key=lambda e: abs(e[0] - dungeon.player[0]) + abs(e[1] - dungeon.player[1]))
            gap = abs(nearest[0] - dungeon.player[0]) + abs(nearest[1] - dungeon.player[1])
            if gap <= 6 and rng.random() < 0.78:
                dx = int(np.sign(dungeon.player[0] - nearest[0]))
                dy = int(np.sign(dungeon.player[1] - nearest[1]))
                for action, (ddx, ddy) in enumerate(DELTA[1:], start=1):
                    if (ddx, ddy) in ((dx, 0), (0, dy)):
                        if dungeon._free((dungeon.player[0] + ddx, dungeon.player[1] + ddy)):
                            yield action
                            break
                else:
                    yield int(rng.integers(5))
                continue
        targets = dungeon.diamonds or [dungeon.goal]
        target = min(targets, key=lambda c: abs(c[0] - dungeon.player[0]) + abs(c[1] - dungeon.player[1]))
        if rng.random() < 0.85:
            yield bfs_step(dungeon.grid, dungeon.player, target)
        else:
            yield int(rng.integers(5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/dungeon/data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enemies", type=int, default=3)
    ap.add_argument("--diamonds", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = np.empty((args.episodes, args.steps + 1, GRID, GRID, 3), dtype=np.uint8)
    actions = np.empty((args.episodes, args.steps), dtype=np.uint8)
    for episode in range(args.episodes):
        seed = args.seed * 100003 + episode
        dungeon = Dungeon(seed, args.enemies, args.diamonds)
        policy = navigate(dungeon, np.random.default_rng(seed + 1))
        frames[episode, 0] = dungeon.render()
        for t in range(args.steps):
            action = next(policy)
            actions[episode, t] = action
            frames[episode, t + 1] = dungeon.step(action)
        if episode % 50 == 0:
            print(f"episode {episode}", flush=True)

    flat = np.ascontiguousarray(frames).reshape(-1)
    (args.out / "frames.bin").write_bytes(flat.tobytes())
    (args.out / "actions.bin").write_bytes(np.ascontiguousarray(actions).tobytes())
    (args.out / "meta.json").write_text(json.dumps({
        "episodes": args.episodes, "steps": args.steps, "size": GRID, "actions": DIR_NAMES,
    }))
    print(f"wrote {args.out}: frames {frames.shape}, {flat.size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
