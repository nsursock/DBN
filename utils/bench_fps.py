#!/usr/bin/env python3
"""Benchmark MLX RL environment/training throughput and time-to-solve.

Examples
--------
Normal throughput benchmark:
    python bench_fps.py normal --env cartpole --algo ppo --envs 1 4 16 64 256
    python bench_fps.py normal --env pendulum --algo sac --envs 1 4 16 64 256
    python bench_fps.py normal --env pendulum --algo td3 --envs 1 4 16 64 256

TTS benchmark:
    python bench_fps.py solve --target cartpole_ppo --envs 32 128 256
    python bench_fps.py solve --target pendulum_sac --envs 32 128 256
    python bench_fps.py solve --target pendulum_td3 --envs 32 128 256

The normal mode reports:
    env FPS, training FPS, RSS/peak RSS, and best-effort macOS temperature.

The solve mode reports:
    solved?, wall-clock TTS, timesteps to solve, environment steps to solve,
    training FPS at solve, final/best evaluation reward, slope and noise.

This file is intentionally standalone and is not part of the fast pytest suite.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

try:
    from tabulate import tabulate
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tabulate is required. Install it with: pip install tabulate") from exc

def _components():
    try:
        from .cartpole import CartPoleMLX
        from .pendulum import PendulumMLX
        from .ppo import PPO
        from .sac import SAC
        from .td3 import TD3
    except ImportError:
        from cartpole import CartPoleMLX
        from pendulum import PendulumMLX
        from ppo import PPO
        from sac import SAC
        from td3 import TD3
    return CartPoleMLX, PendulumMLX, PPO, SAC, TD3


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


def _mlx():
    try:
        import mlx.core as mx
        return mx
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("MLX is required. Install mlx before running this benchmark.") from exc


def _rss_mb() -> float:
    """Current RSS in MB using only stdlib facilities."""
    try:
        import psutil  # optional, more accurate current RSS

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KB.
        return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _peak_rss_mb() -> float:
    """Peak RSS in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _temperature_c() -> float | None:
    """Best-effort macOS temperature reading; returns None when unavailable.

    No privileged operation is required. Different Macs expose temperature through
    different utilities, so several commonly installed tools are tried.
    """
    commands = [
        ["osx-cpu-temp"],
        ["smctemp"],
        ["powermetrics", "--samplers", "smc", "-n", "1", "-i", "100"],
    ]
    patterns = [
        re.compile(r"(-?\d+(?:\.\d+)?)\s*°?C", re.I),
        re.compile(r"CPU\s+die\s+temperature:\s*(-?\d+(?:\.\d+)?)", re.I),
        re.compile(r"CPU\s+temperature:\s*(-?\d+(?:\.\d+)?)", re.I),
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
            continue
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for pattern in patterns:
            match = pattern.search(output)
            if match:
                try:
                    value = float(match.group(1))
                    if -20.0 < value < 130.0:
                        return value
                except ValueError:
                    pass
    return None


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _make_env(env_name: str, n_envs: int, seed: int):
    CartPoleMLX, PendulumMLX, *_ = _components()
    if env_name == "cartpole":
        return CartPoleMLX(n_envs=n_envs, seed=seed)
    if env_name == "pendulum":
        return PendulumMLX(n_envs=n_envs, seed=seed)
    raise ValueError(f"unknown environment: {env_name}")


def _make_model(env_name: str, algo: str, env, seed: int, logdir: str | None = None, verbose: int = 0):
    _, _, PPO, SAC, TD3 = _components()
    kwargs = dict(seed=seed, verbose=verbose, tensorboard_log=logdir)
    if algo == "ppo":
        if env_name == "cartpole":
            kwargs.update(n_steps=256, batch_size=256, n_epochs=4, learning_rate=3e-4, gamma=0.99)
        else:
            kwargs.update(n_steps=512, batch_size=512, n_epochs=5, learning_rate=3e-4, gamma=0.99, ent_coef=0.0)
        return PPO("MlpPolicy", env, **kwargs)
    if algo == "sac":
        kwargs.update(
            buffer_size=200_000,
            learning_starts=2_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )
        return SAC("MlpPolicy", env, **kwargs)
    if algo == "td3":
        kwargs.update(
            buffer_size=200_000,
            learning_starts=2_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )
        return TD3("MlpPolicy", env, **kwargs)
    raise ValueError(f"unknown algorithm: {algo}")


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


def _recent_progress(model) -> dict[str, float]:
    """Read the final progress.csv generated by the model, when available."""
    path = getattr(model, "csv_path", None)
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {}
        row = rows[-1]
        result = {}
        for key, value in row.items():
            if key:
                try:
                    result[key] = float(value)
                except (TypeError, ValueError):
                    pass
        return result
    except OSError:
        return {}


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
    max_timesteps = int(args.max_timesteps)
    chunk = int(args.eval_interval)

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
    for target in targets:
        for n_envs in args.envs:
            rows.append(run_solve_one(args, target, n_envs))
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
    sub = parser.add_subparsers(dest="mode", required=True)

    normal = sub.add_parser("normal", help="throughput / memory / temperature benchmark")
    normal.add_argument("--env", choices=["cartpole", "pendulum"], required=True)
    normal.add_argument("--algo", choices=["ppo", "sac", "td3"], required=True)
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
    normal.add_argument("--seed", type=int, default=0)
    normal.add_argument("--logdir", default="runs/bench")
    normal.set_defaults(func=run_normal, printer=print_normal)

    solve = sub.add_parser("solve", help="time-to-solve benchmark")
    solve.add_argument("--target", choices=[*TARGETS, "all"], required=True)
    solve.add_argument("--envs", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    solve.add_argument("--eval-interval", type=int, default=10_000)
    solve.add_argument("--eval-episodes", type=int, default=20)
    solve.add_argument("--max-timesteps", type=int, default=500_000)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--logdir", default="runs/bench")
    solve.set_defaults(func=run_solve, printer=print_solve)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = args.func(args)
    args.printer(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
