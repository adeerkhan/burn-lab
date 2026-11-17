"""FireflyRoom for the burn-lab world-model example.

A deliberately small port of `avbiswas/computer_vision_recipes` world_model/lightroom/env.py:
a dark 64x64 room, 2-3 boxes, a flashlight with tank controls and cast shadows,
and 1-2 wandering fireflies that fly above the boxes. Exact 2D ray casting in numpy.

Records flat u8 frames and int8 actions the Rust loader can seek into.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DIR_NAMES = ["stay", "up", "down", "left", "right"]
PALETTE = [
    (1.00, 0.62, 0.15),
    (0.20, 0.85, 1.00),
    (1.00, 0.25, 0.80),
    (0.55, 1.00, 0.25),
]
LAMP_STEP = 0.03
TURN_STEP = np.deg2rad(10)
LAMP_R = 0.045
BEAM_HALF, BEAM_SOFT = np.deg2rad(25), np.deg2rad(7)
FLY_R, FLY_SPEED, FLY_TURN = 0.03, 0.018, 0.18
AMBIENT = np.array([0.02, 0.025, 0.035])
BOX_RGB = np.array([0.29, 0.31, 0.36])
LAMP_RGB = np.array([1.0, 0.88, 0.62])


class FireflyRoom:
    def __init__(self, res=64, seed=None):
        self.res = res
        self.rng = np.random.default_rng(seed)
        c = (np.arange(res) + 0.5) / res
        self.px, self.py = np.meshgrid(c, c)

    def reset(self):
        rng = self.rng
        self.n_boxes = int(rng.integers(2, 4))
        self.n_flies = int(rng.integers(1, 3))
        for _ in range(500):
            boxes = [
                dict(c=rng.uniform(0.18, 0.82, 2), w=rng.uniform(0.09, 0.20),
                     h=rng.uniform(0.07, 0.15), ang=rng.uniform(0, np.pi))
                for _ in range(self.n_boxes)
            ]
            self.polys = [self._poly(b) for b in boxes]
            self.lamp = rng.uniform(0.12, 0.88, 2)
            self.heading = rng.uniform(-np.pi, np.pi)
            if self._placements_ok():
                break
        else:
            raise RuntimeError("could not place objects")
        colours = rng.choice(len(PALETTE), self.n_flies, replace=False)
        self.fly_rgb = np.array([PALETTE[int(i)] for i in colours])
        self.fly_pos = rng.uniform(0.1, 0.9, (self.n_flies, 2))
        self.fly_head = rng.uniform(0, 2 * np.pi, self.n_flies)
        return self.render()

    def _placements_ok(self):
        for p in self.polys:
            if (p < 0.03).any() or (p > 0.97).any():
                return False
            if self._inside(p, self.lamp[0], self.lamp[1]):
                return False
        return True

    def _poly(self, b):
        ca, sa = np.cos(b["ang"]), np.sin(b["ang"])
        hw, hh = b["w"] / 2, b["h"] / 2
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        return local @ np.array([[ca, -sa], [sa, ca]]).T + b["c"]

    @staticmethod
    def _inside(poly, x, y):
        inside = np.ones(np.shape(x), dtype=bool)
        for i in range(4):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
            inside &= (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0
        return inside

    def step(self, action):
        action = int(action)
        if action == 1 or action == 2:
            sign = 1 if action == 1 else -1
            new = self.lamp + sign * LAMP_STEP * np.array([np.cos(self.heading), np.sin(self.heading)])
            r = LAMP_R * 1.5
            ok = (new > r).all() and (new < 1 - r).all()
            if ok:
                ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                xs = np.r_[new[0], new[0] + r * np.cos(ang)]
                ys = np.r_[new[1], new[1] + r * np.sin(ang)]
                ok = not any(self._inside(p, xs, ys).any() for p in self.polys)
            if ok:
                self.lamp = new
        elif action == 3 or action == 4:
            self.heading = (self.heading + (-1 if action == 3 else 1) * TURN_STEP + np.pi) % (2 * np.pi) - np.pi

        self.fly_head += self.rng.normal(0, FLY_TURN, self.n_flies)
        v = FLY_SPEED * np.c_[np.cos(self.fly_head), np.sin(self.fly_head)]
        pos = self.fly_pos + v
        lo, hi = FLY_R, 1 - FLY_R
        for k in range(2):
            low, high = pos[:, k] < lo, pos[:, k] > hi
            pos[low, k] = 2 * lo - pos[low, k]
            pos[high, k] = 2 * hi - pos[high, k]
            v[low | high, k] *= -1
        self.fly_pos, self.fly_head = pos, np.arctan2(v[:, 1], v[:, 0])
        return self.render()

    def _visibility(self, lx, ly, polys):
        px, py = self.px, self.py
        lit = np.ones_like(px, dtype=bool)
        dx, dy = lx - px, ly - py
        for p in polys:
            for i in range(4):
                (ax, ay), (bx, by) = p[i], p[(i + 1) % 4]
                ex, ey = bx - ax, by - ay
                den = dx * ey - dy * ex
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = ((ax - px) * ey - (ay - py) * ex) / den
                    u = ((ax - px) * dy - (ay - py) * dx) / den
                lit &= ~((np.abs(den) > 1e-12) & (t > 1e-6) & (t < 1) & (u >= 0) & (u <= 1))
        return lit

    def render(self):
        px, py = self.px, self.py
        img = np.broadcast_to(AMBIENT, px.shape + (3,)).copy()
        box_mask = np.zeros_like(px, dtype=bool)
        for p in self.polys:
            box_mask |= self._inside(p, px, py)

        lx, ly = self.lamp
        hx, hy = np.cos(self.heading), np.sin(self.heading)
        d_lamp = np.hypot(px - lx, py - ly)
        cos_to_pixel = ((px - lx) * hx + (py - ly) * hy) / np.maximum(d_lamp, 1e-6)
        inner, outer = np.cos(BEAM_HALF), np.cos(BEAM_HALF + BEAM_SOFT)
        edge = np.clip((cos_to_pixel - outer) / (inner - outer), 0, 1)
        cone = edge * edge * (3 - 2 * edge)
        light = cone * np.exp(-(d_lamp / 0.8) ** 2) + 0.35 * np.exp(-(d_lamp / 0.10) ** 2)
        img += (light * self._visibility(lx, ly, self.polys))[..., None] * LAMP_RGB * 0.95

        fly_d = []
        for (fx, fy), rgb in zip(self.fly_pos, self.fly_rgb):
            occluders = [p for p in self.polys if not self._inside(p, np.array([fx]), np.array([fy]))[0]]
            d = np.hypot(px - fx, py - fy)
            fly_d.append(d)
            img += (np.exp(-(d / 0.22) ** 2) * self._visibility(fx, fy, occluders))[..., None] * rgb * 0.85

        tint = sum(np.exp(-(d / 0.14) ** 2)[..., None] * rgb * 0.35 for d, rgb in zip(fly_d, self.fly_rgb))
        img[box_mask] = (BOX_RGB + tint)[box_mask]
        img[d_lamp < LAMP_R * 1.35] = (0.08, 0.08, 0.10)
        img[d_lamp < LAMP_R] = (0.92, 0.92, 0.88)
        lens = np.hypot(px - (lx + hx * LAMP_R * 0.75), py - (ly + hy * LAMP_R * 0.75))
        img[lens < LAMP_R * 0.55] = (1.0, 0.85, 0.35)
        for d, rgb in zip(fly_d, self.fly_rgb):
            img[d < FLY_R] = rgb
            img[d < FLY_R * 0.45] = 0.5 + 0.5 * rgb
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def lamp_policy(env, rng):
    """Scripted player: chunks of one action, 30% stay / 35% forward / others turn."""
    while True:
        a = int(rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.35, 0.10, 0.125, 0.125]))
        for _ in range(int(rng.integers(2, 11))):
            yield a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("/home/adeerkhan/tmp/world/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = np.empty((args.episodes, args.steps + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((args.episodes, args.steps), dtype=np.int8)
    for episode in range(args.episodes):
        seed = args.seed * 100003 + episode
        rng = np.random.default_rng(seed + 1)
        env = FireflyRoom(seed=seed)
        frames[episode, 0] = env.reset()
        policy = lamp_policy(env, rng)
        for t in range(args.steps):
            action = next(policy)
            actions[episode, t] = action
            frames[episode, t + 1] = env.step(action)
        if episode % 100 == 0:
            print(f"episode {episode}", flush=True)

    np.save(args.out / "frames.npy", frames)
    np.save(args.out / "actions.npy", actions)
    (args.out / "meta.json").write_text(json.dumps({
        "episodes": args.episodes, "steps": args.steps, "size": 64,
        "actions": DIR_NAMES,
    }))
    print(f"wrote {args.out}: frames {frames.shape}, actions {actions.shape}")


if __name__ == "__main__":
    main()
