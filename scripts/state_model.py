#!/usr/bin/env python3
"""Train the state-first avatar and enemy transition models for the dungeon.

The avatar is trained counterfactually: every recorded state is expanded into
all five actions with the true next cell computed from the static walls. That
removes the "action is predictable from the frame" problem and makes the
learned avatar effectively exact.

The enemy model learns the move distribution (stay/up/down/left/right) from
(enemy cell, player cell, action, local walls). The policy is stochastic, so
the model is trained with cross-entropy and sampled at play time.

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


def avatar_features(player, wall, action: int) -> np.ndarray:
    out = np.zeros(13, dtype=np.float32)
    x, y = player
    out[0], out[1] = x / GRID, y / GRID
    out[2 + action] = 1.0
    out[7] = wall[y, x]
    for index, (dx, dy) in enumerate(DELTA):
        if index != action:
            continue
        tx, ty = min(GRID - 1, max(0, x + dx)), min(GRID - 1, max(0, y + dy))
        out[8] = wall[ty, tx]
    for index, (dx, dy) in enumerate(DELTA[1:]):
        nx, ny = min(GRID - 1, max(0, x + dx)), min(GRID - 1, max(0, y + dy))
        out[9 + index] = wall[ny, nx]
    return out


def enemy_features(enemy, player, wall, action: int) -> np.ndarray:
    out = np.zeros(17, dtype=np.float32)
    x, y = enemy
    px, py = player
    out[0], out[1] = x / GRID, y / GRID
    out[2], out[3] = px / GRID, py / GRID
    out[4 + action] = 1.0
    out[9] = wall[y, x]
    for index, (dx, dy) in enumerate(DELTA[1:]):
        nx, ny = min(GRID - 1, max(0, x + dx)), min(GRID - 1, max(0, y + dy))
        out[10 + index] = wall[ny, nx]
    out[14] = np.sign(px - x)
    out[15] = np.sign(py - y)
    out[16] = min(1.0, (abs(px - x) + abs(py - y)) / GRID)
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


class AvatarMoveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
            nn.Linear(96, 5),
        )

    def forward(self, x):
        return self.net(x)


class EnemyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
            nn.Linear(96, 5),
        )

    def forward(self, x):
        return self.net(x)


def load(data: Path):
    meta = json.loads((data / "meta.json").read_text())
    steps = int(meta["steps"])
    episodes = int(meta["episodes"])
    states = np.memmap(data / "state.bin", dtype=np.uint8, mode="r").reshape(episodes, steps + 1, 28)
    actions = np.memmap(data / "actions.bin", dtype=np.uint8, mode="r").reshape(episodes, steps)
    walls = np.memmap(data / "walls.bin", dtype=np.uint8, mode="r").reshape(episodes, GRID, GRID)
    return states, actions, walls, episodes, steps


def avatar_batch(states, walls, indices):
    xs, ys = [], []
    for episode, time in indices:
        x, y = int(states[episode, time, 0]), int(states[episode, time, 1])
        wall = walls[episode]
        for action in range(5):
            dx, dy = DELTA[action]
            tx, ty = min(GRID - 1, max(0, x + dx)), min(GRID - 1, max(0, y + dy))
            if wall[ty, tx]:
                tx, ty = x, y
            xs.append(avatar_features((x, y), wall, action))
            ys.append([tx / GRID, ty / GRID])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.array(ys, dtype=np.float32))


def enemy_batch(states, actions, walls, indices):
    xs, ys = [], []
    for episode, time in indices:
        action = int(actions[episode, time])
        wall = walls[episode]
        player = (int(states[episode, time, 0]), int(states[episode, time, 1]))
        for slot in range(4):
            ex, ey = int(states[episode, time, 2 + 2 * slot]), int(states[episode, time, 3 + 2 * slot])
            if ex == 255:
                continue
            nx, ny = int(states[episode, time + 1, 2 + 2 * slot]), int(states[episode, time + 1, 3 + 2 * slot])
            label = 0
            for index, (dx, dy) in enumerate(DELTA[1:], start=1):
                if (nx, ny) == (ex + dx, ey + dy):
                    label = index
                    break
            xs.append(enemy_features((ex, ey), player, wall, action))
            ys.append(label)
    if not xs:
        return None, None
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.array(ys, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/data"))
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/dungeon8/state"))
    ap.add_argument("--avatar-steps", type=int, default=3000)
    ap.add_argument("--enemy-steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    states, actions, walls, episodes, _ = load(args.data)
    train_eps = episodes - 30
    rng = np.random.default_rng(args.seed)
    print(f"{episodes} episodes, train {train_eps}")

    avatar = AvatarNet()
    opt = torch.optim.Adam(avatar.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    for step in range(1, args.avatar_steps + 1):
        eps = rng.integers(0, train_eps, args.batch)
        times = rng.integers(0, 128, args.batch)
        x, y = avatar_batch(states, walls, list(zip(eps, times)))
        pred = avatar(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0:
            with torch.no_grad():
                ve = rng.integers(train_eps, episodes, 64)
                vt = rng.integers(0, 128, 64)
                vx, vy = avatar_batch(states, walls, list(zip(ve, vt)))
                exact = ((avatar(vx).numpy() * GRID).round() == (vy.numpy() * GRID).round()).all(axis=1).mean()
            print(f"avatar {step:>5} mse {loss.item():.6f} exact {exact:.3f}", flush=True)

    move_net = AvatarMoveNet()
    opt = torch.optim.Adam(move_net.parameters(), lr=args.lr)
    ce_move = nn.CrossEntropyLoss()
    for step in range(1, args.avatar_steps + 1):
        eps = rng.integers(0, train_eps, args.batch)
        times = rng.integers(0, 128, args.batch)
        xs, ys = [], []
        for episode, time in zip(eps, times):
            x, y = int(states[episode, time, 0]), int(states[episode, time, 1])
            wall = walls[episode]
            for action in range(5):
                dx, dy = DELTA[action]
                tx, ty = min(GRID - 1, max(0, x + dx)), min(GRID - 1, max(0, y + dy))
                label = action if wall[ty, tx] == 0 else 0
                xs.append(avatar_features((x, y), wall, action))
                ys.append(label)
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.array(ys, dtype=np.int64))
        pred = move_net(x)
        loss = ce_move(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0:
            with torch.no_grad():
                acc = (move_net(x).argmax(1) == y).float().mean().item()
            print(f"move   {step:>5} ce {loss.item():.4f} exact {acc:.3f}", flush=True)

    enemy = EnemyNet()
    opt = torch.optim.Adam(enemy.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    for step in range(1, args.enemy_steps + 1):
        eps = rng.integers(0, train_eps, args.batch)
        times = rng.integers(0, 128, args.batch)
        x, y = enemy_batch(states, actions, walls, list(zip(eps, times)))
        if x is None:
            continue
        pred = enemy(x)
        loss = ce(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0:
            with torch.no_grad():
                ve = rng.integers(train_eps, episodes, 64)
                vt = rng.integers(0, 128, 64)
                vx, vy = enemy_batch(states, actions, walls, list(zip(ve, vt)))
                acc = (enemy(vx).argmax(1) == vy).float().mean() if vx is not None else torch.tensor(0.0)
            print(f"enemy  {step:>5} ce {loss.item():.4f} exact {acc.item():.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(avatar.state_dict(), args.out / "avatar-model.pt")
    torch.save(move_net.state_dict(), args.out / "avatar-move.pt")
    torch.save(enemy.state_dict(), args.out / "enemy-model.pt")
    (args.out / "config.json").write_text(json.dumps({"avatar": 13, "enemy": 17, "hidden": 96}))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
