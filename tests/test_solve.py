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
import math
import statistics
import sys

import mlx.core as mx
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "scripts",
    ),
)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean: float, sigma: float, n_envs: int = 1, action_dim: int = 1, theta: float = 0.15, dt: float = 1.0):
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.dt = float(dt)
        self.n_envs = int(n_envs)
        self.action_dim = int(action_dim)
        self.x = mx.zeros((self.n_envs, self.action_dim), dtype=mx.float32)

    def __call__(self, n: int | None = None):
        if n is not None and n != self.n_envs:
            self.n_envs = int(n)
            self.x = mx.zeros((self.n_envs, self.action_dim), dtype=mx.float32)
        noise = self.x
        self.x = (
            self.x
            + self.theta * (self.mean - self.x) * self.dt
            + self.sigma * math.sqrt(self.dt) * mx.random.normal(shape=self.x.shape)
        )
        mx.eval(self.x)
        return noise


from cartpole import CartPoleMLX
from pendulum import PendulumMLX
from ppo import PPO
from sac import SAC
from td3 import TD3


CRITERIA = {
    "cartpole": (475.0, 100),   # Gymnasium's standard 500-step solved threshold.
    "pendulum": (-200.0, 100),  # Average return >= -200.
}


def evaluate(model, env, episodes=100):
    obs = env.reset()
    n_envs = env.n_envs
    current = mx.zeros((n_envs,), dtype=mx.float32)
    returns = []

    while len(returns) < episodes:
        action = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        current = current + reward
        done = terminated | truncated
        mx.eval(current, done)

        if mx.any(done).item():
            for i, finished in enumerate(done.tolist()):
                if finished:
                    returns.append(current.tolist()[i])
                    if len(returns) >= episodes:
                        break
            current = mx.where(done, 0.0, current)

        mx.eval(obs, current, done)

    clip = returns[:episodes]
    return statistics.mean(clip), statistics.pstdev(clip) if len(clip) > 1 else 0.0


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
    elif kind.startswith("pendulum_td3"):
        env = PendulumMLX(n_envs=n_envs, seed=seed)
        model = TD3("MlpPolicy", env, learning_rate=3e-4, buffer_size=200_000, learning_starts=2_000, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5, action_noise=OrnsteinUhlenbeckActionNoise(0.0, 0.8, n_envs, env.action_dim, theta=0.1), verbose=1, seed=seed, tensorboard_log=logdir)
        algo = "TD3"
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
    parser.add_argument("target", choices=["cartpole_ppo", "pendulum_ppo", "pendulum_sac", "pendulum_td3", "all"])
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", default="runs/mlx")
    args = parser.parse_args(argv)
    targets = ["cartpole_ppo", "pendulum_ppo", "pendulum_sac", "pendulum_td3"] if args.target == "all" else [args.target]
    results = [run(t, args.n_envs, args.timesteps, args.seed, args.logdir) for t in targets]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
