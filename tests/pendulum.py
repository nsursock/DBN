"""Pure-MLX vectorized Pendulum-v1 dynamics."""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Space:
    shape: tuple
    low: object
    high: object
    n: int | None = None


class PendulumMLX:
    observation_dim = 3
    action_dim = 1
    max_episode_steps = 200

    def __init__(self, n_envs: int = 1, seed: int | None = None):
        self.n_envs = int(n_envs)
        self.seed = 0 if seed is None else int(seed)
        self.observation_space = Space((3,), -1.0, 1.0)
        self.action_space = Space((1,), -2.0, 2.0)
        self._episode_length = mx.zeros((self.n_envs,), dtype=mx.int32)
        self.reset()

    def _initial_obs(self):
        theta = mx.random.uniform(shape=(self.n_envs,), low=-math.pi, high=math.pi)
        theta_dot = mx.random.uniform(shape=(self.n_envs,), low=-1.0, high=1.0)
        return mx.stack([mx.cos(theta), mx.sin(theta), theta_dot], axis=1)

    def reset(self):
        self.state = self._initial_obs()
        self._episode_length = mx.zeros((self.n_envs,), dtype=mx.int32)
        mx.eval(self.state)
        return self.state

    def step(self, actions):
        u = mx.clip(mx.array(actions, dtype=mx.float32).reshape((self.n_envs,)), -2.0, 2.0)
        cos_t, sin_t, theta_dot = [self.state[:, i] for i in range(3)]
        theta = mx.arctan2(sin_t, cos_t)
        m, l, g, dt = 1.0, 1.0, 10.0, 0.05
        cost = mx.square(((theta + math.pi) % (2.0 * math.pi) - math.pi))
        reward = -(cost + 0.1 * mx.square(theta_dot) + 0.001 * mx.square(u))
        reward = reward.astype(mx.float32)
        new_theta_dot = theta_dot + (-3.0 * g / (2.0 * l) * mx.sin(theta + math.pi) + 3.0 / (m * l * l) * u) * dt
        new_theta_dot = mx.clip(new_theta_dot, -8.0, 8.0)
        new_theta = theta + new_theta_dot * dt
        self._episode_length = self._episode_length + 1
        terminated = mx.zeros((self.n_envs,), dtype=mx.bool_)
        truncated = self._episode_length >= self.max_episode_steps
        done = truncated
        next_state = mx.stack([mx.cos(new_theta), mx.sin(new_theta), new_theta_dot], axis=1)
        fresh = self._initial_obs()
        next_state = mx.where(done[:, None], fresh, next_state)
        self._episode_length = mx.where(done, 0, self._episode_length)
        self.state = next_state
        return next_state, reward, terminated, truncated, {}
