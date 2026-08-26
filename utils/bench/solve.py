#!/usr/bin/env python3
"""Time-to-solve benchmark for MLX RL agents.

Examples
--------
    .venv/bin/python utils/bench/solve.py \
        --target cartpole_ppo --envs 32 128 256

    .venv/bin/python utils/bench/solve.py \
        --target pendulum_sac --envs 32 128 256 --max-timesteps 1_000_000

The solve mode reports:
    solved?, wall-clock TTS, timesteps to solve, environment steps to solve,
    training FPS at solve, final/best evaluation reward, slope and noise.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from typing import Iterable

try:
    from .common import _fmt, _make_env, _make_model, _mlx, _recent_progress
except ImportError:
    from common import _fmt, _make_env, _make_model, _mlx, _recent_progress

try:
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tabulate is required. Install it with: pip install tabulate") from exc


__test__ = False

SOLVE_CRITERIA = {
    "cartpole": 475.0,
    "pendulum": -200.0,
}

TARGETS = {
    "cartpole_ppo": ("cartpole", "ppo"),
    "pendulum_ppo": ("pendulum", "ppo"),
    "pendulum_sac": ("pendulum", "sac"),
    "pendulum_td3": ("pendulum", "td3"),
}


def _evaluate(model, env_name: str, n_envs: int, seed: int, episodes: int) -> tuple[float, float]:
    """Evaluate deterministic policy on completed vectorized episodes."""
    mx = _mlx()
    env = _make_env(env_name, n_envs, seed + 10_000)
    obs = env.reset()
    running = mx.zeros((n_envs,), dtype=mx.float32)
    returns: list[float] = []
    while len(returns) < episodes:
        action = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        running = running + reward
        mx.eval(obs, running, done)
        done_values = done.tolist()
        return_values = running.tolist()
        for i, is_done in enumerate(done_values):
            if is_done and len(returns) < episodes:
                returns.append(float(return_values[i]))
        running = mx.where(done, 0.0, running)
        mx.eval(running)
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return mean, std


def _slope_noise(values: Iterable[float], window: int = 32) -> tuple[float, float]:
    values = list(values)[-window:]
    if len(values) < 2:
        return 0.0, 0.0
    slope = (values[-1] - values[0]) / (len(values) - 1)
    mean = statistics.fmean(values)
    noise = statistics.pstdev(values) if len(values) > 1 else 0.0
    return slope, noise


def run_solve_one(args, target: str, n_envs: int) -> dict:
    env_name, algo = TARGETS[target]
    logdir = os.path.join(args.logdir, "solve", f"{target}_{n_envs}")
    os.makedirs(logdir, exist_ok=True)
    _mlx()
    env = _make_env(env_name, n_envs, args.seed)
    model = _make_model(env_name, algo, env, args.seed, logdir=logdir, verbose=0)
    threshold = SOLVE_CRITERIA[env_name]

    wall_start = time.perf_counter()
    eval_count = 0
    reward_history: list[float] = []
    best_reward = -math.inf
    solved = False
    solve_elapsed = None
    solve_timesteps = None
    solve_reward = None
    solve_std = None
    n_steps = getattr(model, "n_steps", 1)
    rollout = n_envs * n_steps
    max_timesteps = max(int(args.max_timesteps), n_envs * 16_000)
    chunk = max(int(args.eval_interval), rollout)
    chunk = ((chunk + rollout - 1) // rollout) * rollout

    # Each call uses an increasing absolute target because the MLX port keeps
    # cumulative total_timesteps on the model, matching SB3's calling style.
    while model.total_timesteps < max_timesteps:
        target_steps = min(max_timesteps, model.total_timesteps + chunk)
        model.learn(target_steps)
        eval_count += 1
        mean_reward, std_reward = _evaluate(model, env_name, n_envs, args.seed, args.eval_episodes)
        reward_history.append(mean_reward)
        best_reward = max(best_reward, mean_reward)
        if mean_reward >= threshold:
            solved = True
            solve_elapsed = time.perf_counter() - wall_start
            solve_timesteps = model.total_timesteps
            solve_reward = mean_reward
            solve_std = std_reward
            break

    elapsed = time.perf_counter() - wall_start
    progress = _recent_progress(model)
    slope, noise = _slope_noise(reward_history)
    train_fps = progress.get("time/fps", 0.0)
    solved_s = solve_elapsed if solved else None
    solved_steps = solve_timesteps if solved else None
    return {
        "target": target,
        "n_envs": n_envs,
        "solved": "PASS" if solved else "FAIL",
        "tts_s": solved_s,
        "tts_timesteps": solved_steps,
        "tts_env_steps": (solved_steps if solved else None),
        "train_fps": train_fps,
        "eval_reward": solve_reward if solved else (reward_history[-1] if reward_history else None),
        "eval_std": solve_std if solved else None,
        "best_reward": best_reward if reward_history else None,
        "reward_slope": slope,
        "reward_noise": noise,
        "evals": eval_count,
        "max_time_s": elapsed,
    }


def run_solve(args) -> list[dict]:
    rows = []
    targets = [args.target] if args.target != "all" else list(TARGETS)
    auto = args.envs is None
    for target in targets:
        n_envs = args.start_envs if auto else None
        env_iter = iter(args.envs) if not auto else None
        plateau_frac = args.plateau_pct / 100.0
        best_fps = 0.0
        failures = 0
        patience = 1
        while True:
            if auto:
                if n_envs > args.max_envs:
                    break
            else:
                try:
                    n_envs = next(env_iter)
                except StopIteration:
                    break
            row = run_solve_one(args, target, n_envs)
            rows.append(row)
            if auto:
                train_fps = row["train_fps"]
                if train_fps > best_fps * (1.0 + plateau_frac):
                    best_fps = float(train_fps)
                    failures = 0
                else:
                    failures += 1
                if failures > patience:
                    break
                n_envs *= 2
    return rows


def print_solve(rows: list[dict]) -> None:
    table = []
    for r in rows:
        table.append([
            r["target"],
            r["n_envs"],
            r["solved"],
            _fmt(r["tts_s"], 2),
            f"{int(r['tts_timesteps']):,}" if r["tts_timesteps"] is not None else "-",
            f"{int(r['tts_env_steps']):,}" if r["tts_env_steps"] is not None else "-",
            f"{r['train_fps']:,.0f}",
            _fmt(r["eval_reward"], 2),
            _fmt(r["eval_std"], 2),
            _fmt(r["best_reward"], 2),
            _fmt(r["reward_slope"], 3),
            _fmt(r["reward_noise"], 2),
            r["evals"],
        ])
    print("\nSOLVE / TIME-TO-SOLVE (TTS)")
    print(tabulate(
        table,
        headers=[
            "target", "n_envs", "solve", "TTS s", "TTS timesteps", "TTS env steps",
            "train FPS", "eval reward", "eval std", "best reward", "slope", "noise", "evals",
        ],
        tablefmt="github",
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=[*TARGETS, "all"], default=None,
                        help="target to solve (required unless --smoke)")
    parser.add_argument("--envs", nargs="+", type=int, default=None,
                        help="explicit list of n_envs to test (overrides auto-doubling)")
    parser.add_argument("--start-envs", type=int, default=4,
                        help="first n_envs when auto-doubling (default: 4)")
    parser.add_argument("--max-envs", type=int, default=100_000,
                        help="maximum n_envs when auto-doubling (default: 100000)")
    parser.add_argument("--plateau-pct", type=float, default=5.0,
                        help="stop auto-doubling when train FPS gain falls below this %% (default: 5.0)")
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-timesteps", type=int, default=500_000)
    parser.add_argument("--smoke", action="store_true",
                        help="quick smoke test with --envs 16 32 64 and 200k max timesteps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", default="runs/bench")
    parser.set_defaults(func=run_solve, printer=print_solve)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        targets = ["cartpole_ppo", "pendulum_sac", "pendulum_td3"]
        args.envs = [16, 32, 64]
        for args.target in targets:
            rows = args.func(args)
            args.printer(rows)
        return 0
    if args.target is None:
        parser.error("--target is required unless --smoke")
    rows = args.func(args)
    args.printer(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
