"""Pure-MLX vectorized CartPole dynamics.

This follows Gymnasium's CartPole equations closely enough for the standard
solved criterion while avoiding one Python/Gym call per environment step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx


@dataclass
class Space:
    shape: tuple
    low: object = None
    high: object = None
    n: int | None = None


class CartPoleMLX:
    observation_dim = 4
    action_dim = 2
    max_episode_steps = 500

    def __init__(self, n_envs: int = 1, seed: int | None = None):
        self.n_envs = int(n_envs)
        self.seed = 0 if seed is None else int(seed)
        self.observation_space = Space((4,))
        self.action_space = Space((), n=2)
        self.action_high = 1
        self._episode_reward = mx.zeros((self.n_envs,), dtype=mx.float32)
        self._episode_length = mx.zeros((self.n_envs,), dtype=mx.int32)
        self._reset_all()

    def _noise(self, shape, scale):
        return mx.random.uniform(shape=shape, low=-scale, high=scale)

    def _initial_state(self, mask=None, state=None):
        x = self._noise((self.n_envs,), 0.05)
        x_dot = self._noise((self.n_envs,), 0.05)
        theta = self._noise((self.n_envs,), 0.05)
        theta_dot = self._noise((self.n_envs,), 0.05)
        fresh = mx.stack([x, x_dot, theta, theta_dot], axis=1)
        if state is None or mask is None:
            return fresh
        return mx.where(mask[:, None], fresh, state)

    def _reset_all(self):
        self.state = self._initial_state()
        self._episode_reward = mx.zeros((self.n_envs,), dtype=mx.float32)
        self._episode_length = mx.zeros((self.n_envs,), dtype=mx.int32)
        mx.eval(self.state)
        return self.state

    def reset(self):
        self.state = self._initial_state()
        self._episode_reward = mx.zeros((self.n_envs,), dtype=mx.float32)
        self._episode_length = mx.zeros((self.n_envs,), dtype=mx.int32)
        mx.eval(self.state)
        return self.state

    def step(self, actions):
        actions = mx.array(actions, dtype=mx.int32).reshape((self.n_envs,))
        x, x_dot, theta, theta_dot = [self.state[:, i] for i in range(4)]
        force = mx.where(actions == 1, 10.0, -10.0)
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masscart + masspole
        length = 0.5
        polemass_length = masspole * length
        tau = 0.02
        costheta = mx.cos(theta)
        sintheta = mx.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        theta_acc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * costheta / total_mass
        x = x + tau * x_dot
        x_dot = x_dot + tau * x_acc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * theta_acc

        self._episode_length = self._episode_length + 1
        terminated = (x < -2.4) | (x > 2.4) | (theta < -12.0 * math.pi / 180.0) | (theta > 12.0 * math.pi / 180.0)
        truncated = self._episode_length >= self.max_episode_steps
        done = terminated | truncated
        reward = mx.where(terminated, 0.0, 1.0).astype(mx.float32)
        self._episode_reward = self._episode_reward + reward

        next_state = mx.stack([x, x_dot, theta, theta_dot], axis=1)
        next_state = self._initial_state(done, next_state)
        self._episode_reward = mx.where(done, 0.0, self._episode_reward)
        self._episode_length = mx.where(done, 0, self._episode_length)
        self.state = next_state
        return next_state, reward, terminated, truncated, {}
