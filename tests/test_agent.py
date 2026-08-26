"""Fast pytest coverage: imports, vectorization, finite outputs, and predict API.

This file is intentionally smoke-level. Long solve runs live in test_solve.py and
should not be part of the normal fast pytest suite.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
    ),
)

import mlx.core as mx

from scripts.ppo import PPO
from scripts.sac import SAC
from tests.cartpole import CartPoleMLX
from tests.pendulum import PendulumMLX


def test_cartpole_vector_step_fast():
    env = CartPoleMLX(n_envs=4, seed=0)
    obs = env.reset()
    nxt, rew, term, trunc, _ = env.step(mx.zeros((4,), dtype=mx.int32))
    mx.eval(nxt, rew, term, trunc)
    assert nxt.shape == (4, 4)
    assert rew.shape == (4,)
    assert bool(mx.all(mx.isfinite(nxt)).item())


def test_pendulum_vector_step_fast():
    env = PendulumMLX(n_envs=4, seed=0)
    obs = env.reset()
    nxt, rew, term, trunc, _ = env.step(mx.zeros((4, 1), dtype=mx.float32))
    mx.eval(nxt, rew, term, trunc)
    assert nxt.shape == (4, 3)
    assert rew.shape == (4,)
    assert bool(mx.all(mx.isfinite(nxt)).item())


def test_ppo_predict_contract():
    env = CartPoleMLX(n_envs=4, seed=1)
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, verbose=0)
    action = model.predict(env.reset(), deterministic=True)
    assert action.shape == (4,)
    assert bool(mx.all((action >= 0) & (action < 2)).item())


def test_sac_predict_contract():
    env = PendulumMLX(n_envs=4, seed=1)
    model = SAC("MlpPolicy", env, batch_size=8, learning_starts=8, verbose=0)
    action = model.predict(env.reset(), deterministic=True)
    mx.eval(action)
    assert action.shape == (4, 1)
    assert bool(mx.all(mx.isfinite(action)).item())
    assert bool(mx.all(action <= 1.0).item()) and bool(mx.all(action >= -1.0).item())
