"""Cellular-automata dungeon with a torch-lit navigator for burn-lab.

    python3 scripts/dungeon_env.py --episodes 300 --steps 128

Generates a cave with the classic 4-5 rule cellular automaton, then drives a
player toward a goal cell. Frames are 64x64 RGB u8, actions are the same five
ids as the FireflyRoom (stay / up / down / left / right), and the output files
have exactly the layout the Rust world model already reads:

    frames.bin   uint8, episode*(steps+1) + time, 3*64*64 interleaved bytes
    actions.bin  uint8, episode*steps + time
    meta.json    episode count, step count, action names

The player carries a torch: only nearby cells are bright, so navigating is a
real task rather than reading a fully visible map.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

DIR_NAMES = ["stay", "up", "down", "left", "right"]
DELTA = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
GRID = 32
SCALE = 2
CELLS = GRID * GRID


def generate(grid_rng: np.random.Generator) -> np.ndarray:
    """Classic cave CA: random fill then 4 iterations of the 4-5 rule."""
    grid = (grid_rng.random((GRID, GRID)) < 0.45).astype(np.uint8)
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
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.grid = generate(rng)
        floor = [(x, y) for y in range(GRID) for x in range(GRID) if self.grid[y, x] == 0]
        if len(floor) < 40:
            raise RuntimeError("degenerate dungeon")
        start = floor[int(rng.integers(len(floor)))]
        reachable = component(self.grid, start)
        candidates = [c for c in reachable if abs(c[0] - start[0]) + abs(c[1] - start[1]) > GRID // 2]
        self.goal = candidates[int(rng.integers(len(candidates)))] if candidates else start
        self.player = start
        self.hits = 0
        yy, xx = np.mgrid[0:64, 0:64]
        self.xs = xx / SCALE
        self.ys = yy / SCALE

    def step(self, action: int) -> np.ndarray:
        action = int(action)
        dx, dy = DELTA[action]
        nx, ny = self.player[0] + dx, self.player[1] + dy
        if 0 <= nx < GRID and 0 <= ny < GRID and self.grid[ny, nx] == 0:
            self.player = (nx, ny)
        else:
            self.hits += 1
        if self.player == self.goal:
            self.goal = (self.goal[0] + 7) % (GRID - 2) + 1, (self.goal[1] + 5) % (GRID - 2) + 1
            if self.grid[self.goal[1], self.goal[0]] == 1:
                self.goal = self.player
        return self.render()

    def render(self) -> np.ndarray:
        cells = self.grid[np.minimum(self.ys.astype(int), GRID - 1),
                          np.minimum(self.xs.astype(int), GRID - 1)]
        floor_colour = np.array([42, 46, 56], dtype=np.float32)
        wall_colour = np.array([150, 156, 168], dtype=np.float32)
        image = np.where(cells[..., None] == 0, floor_colour, wall_colour)

        px = self.player[0] * SCALE + SCALE / 2 - 0.5
        py = self.player[1] * SCALE + SCALE / 2 - 0.5
        distance = np.hypot(self.xs * SCALE - px, self.ys * SCALE - py)
        torch = 0.18 + 0.82 * np.exp(-np.square(distance / 11.0))
        image *= torch[..., None]

        goal_mask = (np.abs(self.xs - self.goal[0]) < 1.0) & (np.abs(self.ys - self.goal[1]) < 1.0)
        image[goal_mask] = np.array([70, 255, 130], dtype=np.float32) * (0.4 + 0.6 * torch[goal_mask, None])
        player_mask = (np.abs(self.xs - self.player[0]) < 1.0) & (np.abs(self.ys - self.player[1]) < 1.0)
        image[player_mask] = np.array([255, 214, 92], dtype=np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)


def navigate(dungeon: Dungeon, rng: np.random.Generator):
    """Mostly follow the shortest path, with some random exploration."""
    while True:
        if rng.random() < 0.82:
            yield bfs_step(dungeon.grid, dungeon.player, dungeon.goal)
        else:
            yield int(rng.integers(5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/dungeon/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = np.empty((args.episodes, args.steps + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((args.episodes, args.steps), dtype=np.uint8)
    for episode in range(args.episodes):
        seed = args.seed * 100003 + episode
        dungeon = Dungeon(seed)
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
        "episodes": args.episodes, "steps": args.steps, "size": 64, "actions": DIR_NAMES,
    }))
    print(f"wrote {args.out}: frames {frames.shape}, {flat.size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
