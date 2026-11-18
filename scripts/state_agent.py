#!/usr/bin/env python3
"""Play the dungeon with the learned state model (model-predictive control).

    python3 scripts/state_agent.py --steps 720 --out /tmp/dungeon8/agent

At every step the agent samples short action sequences, rolls them through the
learned avatar model, scores the resulting positions (near diamonds/exit, away
from enemies) and executes the first action of the best sequence in the real
environment. The dream panel is rendered from the predicted avatar plus the
observed cave state. This is model-based planning with no policy training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dungeon_env import Dungeon, DELTA, GRID  # noqa: E402
from state_model import AvatarNet  # noqa: E402

HORIZON = 6
SAMPLES = 48


def make_dungeon(wall, state, seed: int) -> Dungeon:
    dungeon = Dungeon.__new__(Dungeon)
    dungeon.grid = wall
    dungeon.player = (int(state[0]), int(state[1]))
    dungeon.start = dungeon.player
    dungeon.enemies = []
    for index in range(4):
        x, y = int(state[2 + 2 * index]), int(state[3 + 2 * index])
        if x != 255:
            dungeon.enemies.append((x, y))
    dungeon.diamonds = []
    for index in range(8):
        x, y = int(state[10 + 2 * index]), int(state[11 + 2 * index])
        if x != 255:
            dungeon.diamonds.append((x, y))
    dungeon.goal = (int(state[26]), int(state[27]))
    dungeon.hits = dungeon.caught = dungeon.collected = 0
    dungeon.rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    dungeon.px = xx.astype(np.float32)
    dungeon.py = yy.astype(np.float32)
    return dungeon


def features(player, wall, action: int) -> np.ndarray:
    out = np.zeros(13, dtype=np.float32)
    x, y = player
    out[0], out[1] = x / GRID, y / GRID
    out[2 + action] = 1.0
    out[7] = wall[y, x]
    dx, dy = DELTA[action]
    tx, ty = min(GRID - 1, max(0, x + dx)), min(GRID - 1, max(0, y + dy))
    out[8] = wall[ty, tx]
    for index, (nx, ny) in enumerate(DELTA[1:]):
        sx, sy = min(GRID - 1, max(0, x + nx)), min(GRID - 1, max(0, y + ny))
        out[9 + index] = wall[sy, sx]
    return out


def step_player(model, wall, player, action):
    with torch.no_grad():
        raw = model(torch.from_numpy(features(player, wall, action)).unsqueeze(0)).numpy()[0]
    candidate = (int(np.clip(round(raw[0] * GRID), 0, GRID - 1)),
                 int(np.clip(round(raw[1] * GRID), 0, GRID - 1)))
    if abs(candidate[0] - player[0]) + abs(candidate[1] - player[1]) <= 1 and wall[candidate[1], candidate[0]] == 0:
        return candidate
    return player


def score(player, diamonds, goal, enemies) -> float:
    targets = diamonds if diamonds else [goal]
    distance = min(abs(player[0] - t[0]) + abs(player[1] - t[1]) for t in targets)
    value = -distance
    for enemy in enemies:
        gap = abs(player[0] - enemy[0]) + abs(player[1] - enemy[1])
        value -= 26.0 * float(np.exp(-gap / 5.0))
    if player in diamonds:
        value += 18.0
    if player == goal:
        value += 12.0
    return value


def choose(model, wall, player, action, diamonds, goal, enemies, rng) -> int:
    best_value, best_action = -1e9, 0
    for candidate in range(5):
        reverse = {1: 2, 2: 1, 3: 4, 4: 3}
        if candidate == reverse.get(action):
            continue  # discourage immediate backtracking
        values = []
        for _ in range(SAMPLES):
            current = player
            sequence = [candidate] + [int(rng.integers(5)) for _ in range(HORIZON - 1)]
            for move in sequence:
                current = step_player(model, wall, current, move)
            values.append(score(current, diamonds, goal, enemies))
        value = float(np.mean(values))
        if value > best_value:
            best_value, best_action = value, candidate
    return best_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/demo"))
    ap.add_argument("--model", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/state/avatar-model.pt"))
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/agent"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.data / "meta.json").read_text())
    states = np.memmap(args.data / "state.bin", dtype=np.uint8, mode="r").reshape(
        int(meta["episodes"]), int(meta["steps"]) + 1, 28
    )
    walls = np.memmap(args.data / "walls.bin", dtype=np.uint8, mode="r").reshape(-1, GRID, GRID)
    wall = walls[0]
    model = AvatarNet()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    env = make_dungeon(wall, states[0, 0], args.seed)
    draw = make_dungeon(wall, states[0, 0], args.seed + 1)
    rng = np.random.default_rng(args.seed)
    action = 0
    scores = []
    chosen = []
    for step in range(args.steps):
        action = choose(model, wall, env.player, action, env.diamonds, env.goal, env.enemies, rng)
        chosen.append(action)
        real = env.step(action)
        # Dream: learned avatar + observed cave state.
        draw.player = env.player
        draw.enemies = list(env.enemies)
        draw.diamonds = list(env.diamonds)
        draw.goal = env.goal
        dream = draw.render()
        strip = np.concatenate([real, dream], axis=1)
        from PIL import Image, ImageDraw
        image = Image.fromarray(strip)
        canvas = ImageDraw.Draw(image)
        canvas.text((6, 6), f"step {step:03d}  score {env.collected}  diamonds {len(env.diamonds)}",
                    fill=(255, 255, 0))
        image.save(args.out / f"step-{step:04d}.png")
        scores.append(env.collected)
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{args.steps} diamonds {env.collected}", flush=True)
    print(f"collected {env.collected} diamonds, caught {env.caught}, last pos {env.player}")
    np.array([0, 0, 0, 0] + chosen, dtype=np.uint8).tofile(args.out / "actions.bin")


if __name__ == "__main__":
    main()
