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

    def _reset_state(self, key=None):
        key, k1 = (mx.random.split(key) if key is not None else (None, None))
        theta = mx.random.uniform(shape=(self.n_envs,), low=-math.pi, high=math.pi, key=k1)
        key, k2 = (mx.random.split(key) if key is not None else (None, None))
        theta_dot = mx.random.uniform(shape=(self.n_envs,), low=-1.0, high=1.0, key=k2)
        return mx.stack([mx.cos(theta), mx.sin(theta), theta_dot], axis=1)

    def _initial_obs(self, key=None):
        return self._reset_state(key=key)

    def reset(self):
        self.state = self._initial_obs()
        self._episode_length = mx.zeros((self.n_envs,), dtype=mx.int32)
        mx.eval(self.state)
        return self.state

    def _dynamics(self, state, episode_length, actions, reset_state):
        u = mx.clip(actions.reshape((self.n_envs,)), -2.0, 2.0)
        cos_t, sin_t, theta_dot = [state[:, i] for i in range(3)]
        theta = mx.arctan2(sin_t, cos_t)
        m, l, g, dt = 1.0, 1.0, 10.0, 0.05
        cost = mx.square(((theta + math.pi) % (2.0 * math.pi) - math.pi))
        reward = -(cost + 0.1 * mx.square(theta_dot) + 0.001 * mx.square(u))
        reward = reward.astype(mx.float32)
        new_theta_dot = theta_dot + (-3.0 * g / (2.0 * l) * mx.sin(theta + math.pi) + 3.0 / (m * l * l) * u) * dt
        new_theta_dot = mx.clip(new_theta_dot, -8.0, 8.0)
        new_theta = theta + new_theta_dot * dt
        episode_length = episode_length + 1
        terminated = mx.zeros((self.n_envs,), dtype=mx.bool_)
        truncated = episode_length >= self.max_episode_steps
        done = truncated
        next_state = mx.stack([mx.cos(new_theta), mx.sin(new_theta), new_theta_dot], axis=1)
        next_state = mx.where(done[:, None], reset_state, next_state)
        episode_length = mx.where(done, 0, episode_length)
        return next_state, reward, terminated, truncated, episode_length

    def step(self, actions):
        actions = mx.array(actions, dtype=mx.float32).reshape((self.n_envs, 1))
        reset_state = self._reset_state()
        next_state, reward, terminated, truncated, episode_length = self._dynamics(
            self.state, self._episode_length, actions, reset_state
        )
        self._episode_length = episode_length
        self.state = next_state
        return next_state, reward, terminated, truncated, {}
