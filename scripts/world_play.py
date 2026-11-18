#!/usr/bin/env python3
"""Play inside the world model's dream, next to the real FireflyRoom.

Interactive (needs pygame-ce):

    python3 scripts/world_play.py --ckpt /tmp/world/wm6-residual \\
        --eval-data /tmp/world/vq2

Headless, driven by a script of action ids (no display needed):

    python3 scripts/world_play.py --script /tmp/world/actions.txt \\
        --frames /tmp/world/play --ckpt ... --eval-data ...

The model runs in a Rust child process (`world-play`) that streams one dream
frame per action over stdin/stdout. Both rooms receive the same action, and
only the first few real frames are ever shown to the model; after that it sees
its own predictions. Space re-syncs it from the real room's recent frames.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_env import FireflyRoom  # noqa: E402

ACTION_LABELS = ["stay", "forward", "backward", "turn left", "turn right"]
FRAME_BYTES = 3 * 64 * 64


class DreamClient:
    """Line-based command protocol to the Rust inference server."""

    def __init__(self, binary: str, ckpt: str, eval_data: str, data: str, passes: int | None, delta_scale: float | None = None):
        cmd = [
            binary, "world-play", "--backend", "wgpu",
            "--ckpt", ckpt, "--eval-data", eval_data,
        ]
        if data:
            cmd += ["--data", data]
        if passes is not None:
            cmd += ["--passes", str(passes)]
        if delta_scale is not None:
            cmd += ["--delta-scale", str(delta_scale)]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def action(self, action: int) -> np.ndarray:
        self.process.stdin.write(f"a {int(action)}\n".encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().decode().strip()
        if not header.startswith("F "):
            raise RuntimeError(f"unexpected reply: {header!r}")
        size = int(header.split()[1])
        raw = self.process.stdout.read(size)
        if len(raw) != size:
            raise RuntimeError("short frame from world-play")
        return np.frombuffer(raw, dtype=np.uint8).reshape(64, 64, 3)

    def preview(self, action: int) -> np.ndarray:
        """Predict without committing, for planning."""
        self.process.stdin.write(f"p {int(action)}\n".encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().decode().strip()
        if not header.startswith("F "):
            raise RuntimeError(f"unexpected reply: {header!r}")
        size = int(header.split()[1])
        raw = self.process.stdout.read(size)
        return np.frombuffer(raw, dtype=np.uint8).reshape(64, 64, 3)

    def sync(self, frames: list[tuple[int, np.ndarray]]) -> None:
        """Re-sync from (action, frame) records, oldest first."""
        self.process.stdin.write(f"s {len(frames)}\n".encode())
        for action, frame in frames:
            self.process.stdin.write(bytes([int(action)]) + np.ascontiguousarray(frame).tobytes())
        self.process.stdin.flush()

    def close(self) -> None:
        try:
            self.process.stdin.write(b"q\n")
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()


def make_env(seed: int, context: int, recorded: np.ndarray | None, kind: str = "firefly"):
    if kind == "dungeon":
        from dungeon_env import Dungeon
        env = Dungeon(seed=seed)
        frames = [env.render()]
    else:
        env = FireflyRoom(seed=seed)
        frames = [env.reset()]
    actions = []
    for t in range(context - 1):
        action = int(recorded[t]) if recorded is not None else 0
        frames.append(env.step(action))
        actions.append(action)
    return env, frames, actions


def headless(args, client: DreamClient) -> None:
    recorded = np.fromfile(args.actions, dtype=np.uint8) if args.actions else None
    script = (
        [int(line) for line in Path(args.script).read_text().split() if line.strip()]
        if args.script else list(recorded[:args.steps]) if recorded is not None else [0] * args.steps
    )
    env, real_frames, actions = make_env(args.seed, args.context, recorded, args.env)
    client.sync([(0, real_frames[0])] + [(actions[i - 1], real_frames[i]) for i in range(1, len(real_frames))])
    args.frames.mkdir(parents=True, exist_ok=True)
    for step, action in enumerate(script[: args.steps]):
        real = env.step(action)
        dream = client.action(action)
        strip = np.concatenate([real, dream], axis=1)
        from PIL import Image
        Image.fromarray(strip).save(args.frames / f"step-{step:04d}.png")
        real_frames.append(real)
        actions.append(action)
        real_frames.pop(0)
        actions.pop(0)
        if args.resync_every and (step + 1) % args.resync_every == 0:
            client.sync([(0, real_frames[0])] + [(actions[i - 1], real_frames[i]) for i in range(1, len(real_frames))])
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{min(len(script), args.steps)}", flush=True)
    print(f"wrote {len(list(args.frames.glob('step-*.png')))} frames to {args.frames}")
    if hasattr(env, "collected"):
        print(f"collected {env.collected} diamonds, caught {env.caught} times")


def interactive(args, client: DreamClient) -> None:
    import pygame

    pygame.init()
    scale = 6
    size = (64 * scale, 64 * scale)
    screen = pygame.display.set_mode((size[0] * 2, size[1] + 34))
    pygame.display.set_caption("FireflyRoom: real (left) vs dream (right)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    episode = args.seed
    recorded = None
    if args.actions:
        all_actions = np.fromfile(args.actions, dtype=np.uint8)
        recorded = all_actions[episode * 128:(episode + 1) * 128]
    env, real_frames, actions = make_env(episode, args.context, recorded, args.env)
    client.sync([(0, real_frames[0])] + [(actions[i - 1], real_frames[i]) for i in range(1, len(real_frames))])
    keymap = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }
    step = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap:
                action = keymap[event.key]
                real = env.step(action)
                dream = client.action(action)
                real_frames.append(real)
                actions.append(action)
                real_frames.pop(0)
                actions.pop(0)
                step += 1
                screen.fill((12, 13, 17))
                for i, image in enumerate((real_frames[-1], dream)):
                    surface = pygame.surfarray.make_surface(np.ascontiguousarray(image).swapaxes(0, 1))
                    screen.blit(pygame.transform.scale(surface, size), (i * size[0], 0))
                text = font.render(
                    f"step {step}  action {ACTION_LABELS[action]}   "
                    "arrows: move/turn   space: re-sync   n: new room   q: quit",
                    True, (210, 210, 215),
                )
                screen.blit(text, (8, size[1] + 6))
                pygame.display.flip()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                client.sync(list(zip([0] + actions, real_frames)))
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                episode += 1
                if args.actions:
                    all_actions = np.fromfile(args.actions, dtype=np.uint8)
                    recorded = all_actions[episode * 128:(episode + 1) * 128]
                env, real_frames, actions = make_env(episode, args.context, recorded, args.env)
                client.sync([(0, real_frames[0])] + [(actions[i - 1], real_frames[i]) for i in range(1, len(real_frames))])
                step = 0
        clock.tick(30)
    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="/home/adeerkhan/github/burn/burn-lab/target/release/diffusion")
    ap.add_argument("--ckpt", default="/home/adeerkhan/tmp/world/wm6-residual")
    ap.add_argument("--eval-data", default="/home/adeerkhan/tmp/world/vq2")
    ap.add_argument("--data", default="/home/adeerkhan/tmp/world/data")
    ap.add_argument("--actions", default="/home/adeerkhan/tmp/world/data/actions.bin")
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--delta-scale", type=float, default=None)
    ap.add_argument("--env", choices=["firefly", "dungeon"], default="firefly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--resync-every", type=int, default=0)
    ap.add_argument("--script", type=Path, default=None)
    ap.add_argument("--frames", type=Path, default=None)
    args = ap.parse_args()

    client = DreamClient(args.binary, args.ckpt, args.eval_data, args.data, args.passes, args.delta_scale)
    try:
        if args.script or args.frames:
            if args.frames is None:
                args.frames = Path("/home/adeerkhan/tmp/world/play")
            headless(args, client)
        else:
            interactive(args, client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
