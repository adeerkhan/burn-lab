#!/usr/bin/env python3
"""Train a state-first world model for the dungeon avatar.

The cave is static and known; the learned part is the avatar transition. For
each (player cell, action, local walls) the model predicts the next cell. A
second, tiny head predicts whether the move is blocked. The frame is rendered
by the known environment, so pixels are exact.

    python3 scripts/state_model.py --data /tmp/dungeon8/data --out /tmp/dungeon8/state
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

GRID = 64
DELTA = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]


def features(states: np.ndarray, walls: np.ndarray, indices) -> np.ndarray:
    out = np.zeros((len(indices), 13), dtype=np.float32)
    for row, (episode, time) in enumerate(indices):
        state = states[episode, time]
        wall = walls[episode]
        x, y = int(state[0]), int(state[1])
        action = int(states_action[episode, time])
        out[row, 0] = x / GRID
        out[row, 1] = y / GRID
        out[row, 2 + action] = 1.0
        out[row, 7] = wall[y, x]
        dx, dy = DELTA[action]
        tx, ty = min(GRID - 1, max(0, x + dx)), min(GRID - 1, max(0, y + dy))
        out[row, 8] = wall[ty, tx]
        for index, (nx, ny) in enumerate(DELTA[1:]):
            sx, sy = min(GRID - 1, max(0, x + nx)), min(GRID - 1, max(0, y + ny))
            out[row, 9 + index] = wall[sy, sx]
    return out


class AvatarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
            nn.Linear(96, 2),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    global states_action
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/data"))
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/state"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    states = np.memmap(args.data / "state.bin", dtype=np.uint8, mode="r").reshape(-1, 129, 28)
    states_action = np.memmap(args.data / "actions.bin", dtype=np.uint8, mode="r").reshape(-1, 128)
    walls = np.memmap(args.data / "walls.bin", dtype=np.uint8, mode="r").reshape(-1, GRID, GRID)
    episodes = states.shape[0]
    train_eps = episodes - 30
    print(f"{episodes} episodes, train {train_eps}")

    model = AvatarNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(args.seed)

    for step in range(1, args.steps + 1):
        eps = rng.integers(0, train_eps, args.batch)
        times = rng.integers(0, 128, args.batch)
        indices = list(zip(eps, times))
        x = torch.from_numpy(features(states, walls, indices))
        target = np.stack([
            [states[e, t + 1, 0] / GRID, states[e, t + 1, 1] / GRID] for e, t in indices
        ]).astype(np.float32)
        y = torch.from_numpy(target)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step == 1 or step % 250 == 0:
            with torch.no_grad():
                ve = rng.integers(train_eps, episodes, 256)
                vt = rng.integers(0, 128, 256)
                vi = list(zip(ve, vt))
                vx = torch.from_numpy(features(states, walls, vi))
                vp = model(vx)
                pred_cells = (vp.numpy() * GRID).round().astype(int)
                true_cells = np.stack([[states[e, t + 1, 0], states[e, t + 1, 1]] for e, t in vi])
                exact = (pred_cells == true_cells).all(axis=1).mean()
            print(f"step {step:>5} mse {loss.item():.6f} exact {exact:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "avatar-model.pt")
    (args.out / "avatar-config.json").write_text(json.dumps({"features": 13, "hidden": 96}))
    print(f"wrote {args.out / 'avatar-model.pt'}")


if __name__ == "__main__":
    main()
