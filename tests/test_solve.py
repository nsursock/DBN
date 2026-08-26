"""Long-running solve benchmarks; intentionally excluded from the normal pytest suite.

Examples:
    python test_solve.py cartpole_ppo
    python test_solve.py pendulum_ppo
    python test_solve.py pendulum_sac
    python test_solve.py all
"""
from __future__ import annotations

__test__ = False  # Explicitly exclude this long-running harness from pytest collection.

import argparse
import statistics
import sys

import mlx.core as mx

from cartpole import CartPoleMLX
from pendulum import PendulumMLX
from ppo import PPO
from sac import SAC


CRITERIA = {
    "cartpole": (475.0, 100),   # Gymnasium's standard 500-step solved threshold.
    "pendulum": (-200.0, 100),  # Average return >= -200.
}


def evaluate(model, env, episodes=100):
    obs = env.reset()
    completed = 0
    returns = []
    running = mx.zeros((env.n_envs,), dtype=mx.float32)
    while completed < episodes:
        action = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        running = running + reward
        done = terminated | truncated
        done_np = done.tolist()
        vals = running.tolist()
        for i, d in enumerate(done_np):
            if d and completed < episodes:
                returns.append(float(vals[i]))
                completed += 1
        running = mx.where(done, 0.0, running)
        mx.eval(obs, running)
    return statistics.mean(returns), statistics.pstdev(returns) if len(returns) > 1 else 0.0


def run(kind, n_envs, train_steps, seed, logdir):
    if kind.startswith("cartpole"):
        env = CartPoleMLX(n_envs=n_envs, seed=seed)
        model_cls = PPO
        algo = "PPO"
        train_steps = max(train_steps, 200_000)
        model = model_cls("MlpPolicy", env, n_steps=256, batch_size=256, n_epochs=4, learning_rate=3e-4, gamma=0.99, verbose=1, seed=seed, tensorboard_log=logdir)
        env_name = "cartpole"
    elif kind.startswith("pendulum_ppo"):
        env = PendulumMLX(n_envs=n_envs, seed=seed)
        model = PPO("MlpPolicy", env, n_steps=512, batch_size=512, n_epochs=5, learning_rate=3e-4, gamma=0.99, ent_coef=0.0, verbose=1, seed=seed, tensorboard_log=logdir)
        algo = "PPO"
        env_name = "pendulum"
    else:
        env = PendulumMLX(n_envs=n_envs, seed=seed)
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=200_000, learning_starts=2_000, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, verbose=1, seed=seed, tensorboard_log=logdir)
        algo = "SAC"
        env_name = "pendulum"

    print(f"\nTraining {algo} on {env_name} with n_envs={n_envs}, total_timesteps={train_steps}")
    model.learn(train_steps)
    mean_r, std_r = evaluate(model, env, CRITERIA[env_name][1])
    threshold = CRITERIA[env_name][0]
    passed = mean_r >= threshold
    print(f"SOLVE {algo:4s} {env_name:9s}: mean={mean_r:.2f} std={std_r:.2f} threshold={threshold:.2f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["cartpole_ppo", "pendulum_ppo", "pendulum_sac", "all"])
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", default="runs/mlx")
    args = parser.parse_args(argv)
    targets = ["cartpole_ppo", "pendulum_ppo", "pendulum_sac"] if args.target == "all" else [args.target]
    results = [run(t, args.n_envs, args.timesteps, args.seed, args.logdir) for t in targets]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
