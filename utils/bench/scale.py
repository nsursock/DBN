#!/usr/bin/env python3
"""Benchmark MLX RL environment/training throughput and memory.

Examples
--------
    .venv/bin/python utils/bench/scale.py normal --env cartpole --algo ppo --envs 1 4 16 64 256
    .venv/bin/python utils/bench/scale.py normal --env pendulum --algo sac --envs 1 4 16 64 256
    .venv/bin/python utils/bench/scale.py normal --env pendulum --algo td3 --envs 1 4 16 64 256

The normal mode reports:
    env FPS, training FPS, RSS/peak RSS, and best-effort macOS temperature.

This file is intentionally standalone and is not part of the fast pytest suite.
"""
from __future__ import annotations

import argparse
import os
import time

try:
    from .common import (
        _fmt,
        _make_env,
        _make_model,
        _mlx,
        _recent_progress,
        _rss_mb,
        _peak_rss_mb,
        _temperature_c,
    )
except ImportError:
    from common import (
        _fmt,
        _make_env,
        _make_model,
        _mlx,
        _recent_progress,
        _rss_mb,
        _peak_rss_mb,
        _temperature_c,
    )

try:
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tabulate is required. Install it with: pip install tabulate") from exc


def benchmark_env_fps(env_name: str, n_envs: int, seed: int, steps: int, warmup: int) -> float:
    """Measure raw vectorized environment throughput, excluding agent inference."""
    mx = _mlx()
    env = _make_env(env_name, n_envs, seed)
    obs = env.reset()
    is_discrete = getattr(env.action_space, "n", None) is not None
    if is_discrete:
        action = mx.zeros((n_envs,), dtype=mx.int32)
    else:
        action = mx.zeros((n_envs, env.action_dim), dtype=mx.float32)

    for _ in range(warmup):
        obs, *_ = env.step(action)
    mx.eval(obs)

    start = time.perf_counter()
    for _ in range(steps):
        obs, *_ = env.step(action)
    mx.eval(obs)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return steps * n_envs / elapsed


def run_normal(args) -> list[dict]:
    rows = []
    auto = args.envs is None
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
        _mlx()
        env = _make_env(args.env, n_envs, args.seed)
        algo = args.algo
        logdir = os.path.join(args.logdir, "normal", f"{args.env}_{algo}_{n_envs}")
        os.makedirs(logdir, exist_ok=True)

        rss_before = _rss_mb()
        temp_before = _temperature_c()
        env_fps = benchmark_env_fps(args.env, n_envs, args.seed, args.env_steps, args.warmup)

        model = _make_model(args.env, algo, env, args.seed, logdir=logdir, verbose=0)
        if args.train_steps is not None:
            target_steps = args.train_steps
        elif args.algo == "ppo":
            n_steps = getattr(model, "n_steps", 256)
            target_steps = max(100_000, min(1_000_000, n_envs * n_steps * 50))
        else:
            target_steps = max(20_000, min(1_000_000, n_envs * 1000))
        start = time.perf_counter()
        model.learn(target_steps)
        elapsed = max(time.perf_counter() - start, 1e-9)
        progress = _recent_progress(model)
        train_fps = progress.get("time/fps", (target_steps / elapsed))
        rss_after = _rss_mb()
        peak_rss = _peak_rss_mb()
        temp_after = _temperature_c()
        temp = temp_after if temp_after is not None else temp_before

        rows.append(
            {
                "env": args.env,
                "algo": algo.upper(),
                "n_envs": n_envs,
                "env_fps": env_fps,
                "train_fps": train_fps,
                "rss_mb": rss_after,
                "peak_rss_mb": peak_rss,
                "rss_delta_mb": rss_after - rss_before,
                "temp_c": temp,
                "wall_s": elapsed,
            }
        )
        if auto:
            if train_fps > best_fps * (1.0 + plateau_frac):
                best_fps = float(train_fps)
                failures = 0
            else:
                failures += 1
            if failures > patience:
                break
            n_envs *= 2
    return rows


def print_normal(rows: list[dict]) -> None:
    table = []
    for r in rows:
        table.append([
            r["env"],
            r["algo"],
            r["n_envs"],
            f"{r['env_fps']:,.0f}",
            f"{r['train_fps']:,.0f}",
            f"{r['rss_mb']:.1f}",
            f"{r['peak_rss_mb']:.1f}",
            f"{r['rss_delta_mb']:+.1f}",
            _fmt(r["temp_c"], 1),
            f"{r['wall_s']:.2f}",
        ])
    print("\nNORMAL / THROUGHPUT")
    print(tabulate(
        table,
        headers=["env", "algo", "n_envs", "env FPS", "train FPS", "RSS MB", "peak MB", "ΔRSS MB", "temp °C", "wall s"],
        tablefmt="github",
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    normal = sub.add_parser("normal", help="throughput / memory / temperature benchmark")
    normal.add_argument("--env", choices=["cartpole", "pendulum"], default=None,
                        help="environment (required unless --smoke)")
    normal.add_argument("--algo", choices=["ppo", "sac", "td3"], default=None,
                        help="algorithm (required unless --smoke)")
    normal.add_argument("--envs", nargs="+", type=int, default=None,
                        help="explicit list of n_envs to test (overrides auto-doubling)")
    normal.add_argument("--start-envs", type=int, default=16,
                        help="first n_envs when auto-doubling (default: 16)")
    normal.add_argument("--max-envs", type=int, default=100_000,
                        help="maximum n_envs when auto-doubling (default: 100000)")
    normal.add_argument("--plateau-pct", type=float, default=5.0,
                        help="stop auto-doubling when train FPS gain falls below this %% (default: 5.0)")
    normal.add_argument("--env-steps", type=int, default=2_000, help="raw env steps used for environment FPS")
    normal.add_argument("--warmup", type=int, default=200)
    normal.add_argument("--train-steps", type=int, default=None,
                        help="training steps; if omitted, PPO uses min(1M, n_envs * n_steps * 4) to amortize compile")
    normal.add_argument("--smoke", action="store_true",
                        help="quick smoke test with --envs 16 32 64 and 100k train steps")
    normal.add_argument("--seed", type=int, default=0)
    normal.add_argument("--logdir", default="runs/bench")
    normal.set_defaults(func=run_normal, printer=print_normal)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        combos = [("cartpole", "ppo"), ("pendulum", "sac"), ("pendulum", "td3")]
        rows = []
        for args.env, args.algo in combos:
            args.envs = [16, 32, 64]
            args.train_steps = args.train_steps or 100_000
            rows.extend(args.func(args))
        args.printer(rows)
        return 0 if rows else 1
    if args.env is None or args.algo is None:
        parser.error("--env and --algo are required unless --smoke")
    rows = args.func(args)
    args.printer(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
