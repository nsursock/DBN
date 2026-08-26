from __future__ import annotations

import csv
import os
import re
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

__test__ = False


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
        max_steps = 500 if env_name == "cartpole" else 512
        n_steps = min(max_steps, max(64, 65_536 // env.n_envs))
        n_steps = max(1, n_steps)
        rollout = env.n_envs * n_steps
        batch_size = max(1, rollout // 16)
        n_epochs = max(1, min(4, 512 // env.n_envs))
        kwargs.update(n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=3e-4, gamma=0.99)
        if env_name != "cartpole":
            kwargs["ent_coef"] = 0.0
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
