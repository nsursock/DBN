"""Pure-MLX TD3 agent with a Stable-Baselines3-shaped API.

Designed for batched continuous-control environments such as PendulumMLX.
The core update follows TD3: twin critics, clipped target-policy noise,
clipped double-Q targets, delayed actor updates, and Polyak target updates.
"""
from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from tqdm import tqdm

try:
    from .agent import MLXAgent
except ImportError:
    from agent import MLXAgent


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, net_arch=(256, 256)):
        super().__init__()
        dims = (int(in_dim),) + tuple(int(x) for x in net_arch) + (int(out_dim),)
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = mx.maximum(x, 0.0)
        return x


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, net_arch=(256, 256)):
        super().__init__()
        self.body = MLP(obs_dim, action_dim, net_arch)

    def __call__(self, obs):
        return mx.tanh(self.body(obs))


class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, net_arch=(256, 256)):
        super().__init__()
        self.body = MLP(obs_dim + action_dim, 1, net_arch)

    def __call__(self, obs, action):
        return self.body(mx.concatenate([obs, action], axis=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int):
        self.size = int(size)
        self.obs = mx.zeros((self.size, obs_dim), dtype=mx.float32)
        self.actions = mx.zeros((self.size, action_dim), dtype=mx.float32)
        self.rewards = mx.zeros((self.size,), dtype=mx.float32)
        self.next_obs = mx.zeros((self.size, obs_dim), dtype=mx.float32)
        self.dones = mx.zeros((self.size,), dtype=mx.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, actions, rewards, next_obs, dones):
        n = int(obs.shape[0])
        # The buffer write is intentionally batched: one Python call stores all envs.
        idx = (mx.arange(n, dtype=mx.int32) + self.pos) % self.size
        self.obs[idx] = mx.array(obs, dtype=mx.float32)
        self.actions[idx] = mx.array(actions, dtype=mx.float32)
        self.rewards[idx] = mx.array(rewards, dtype=mx.float32).reshape(-1)
        self.next_obs[idx] = mx.array(next_obs, dtype=mx.float32)
        self.dones[idx] = mx.array(dones, dtype=mx.float32).reshape(-1)
        self.pos = (self.pos + n) % self.size
        self.full = self.full or (self.pos == 0)
        mx.eval(self.obs, self.actions, self.rewards, self.next_obs, self.dones)

    @property
    def length(self) -> int:
        return self.size if self.full else self.pos

    def sample(self, batch_size: int):
        idx = mx.random.randint(0, self.length, shape=(int(batch_size),), dtype=mx.int32)
        return (
            mx.take(self.obs, idx, axis=0),
            mx.take(self.actions, idx, axis=0),
            mx.take(self.rewards, idx, axis=0),
            mx.take(self.next_obs, idx, axis=0),
            mx.take(self.dones, idx, axis=0),
        )


def _polyak(target, source, tau: float):
    target.update(tree_map(lambda t, s: (1.0 - tau) * t + tau * s,
                           target.parameters(), source.parameters()))


def _finite(x) -> bool:
    return bool(mx.all(mx.isfinite(x)).item())


class TD3(MLXAgent):
    """TD3 implementation with the main SB3 constructor parameters.

    SB3 reference parameters mirrored here include learning_rate, buffer_size,
    learning_starts, batch_size, tau, gamma, train_freq, gradient_steps,
    action_noise, policy_delay, target_policy_noise, target_noise_clip,
    stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, and device.
    """

    def __init__(
        self,
        policy: str = "MlpPolicy",
        env=None,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
        csv_log_path=None,
        **kwargs,
    ):
        del policy, replay_buffer_class, replay_buffer_kwargs
        del optimize_memory_usage, use_sde, sde_sample_freq, device, _init_setup_model, kwargs
        if env is None:
            raise ValueError("env is required")
        if getattr(env.action_space, "n", None) is not None:
            raise ValueError("TD3 requires a continuous Box-like action space")

        if seed is not None:
            mx.random.seed(int(seed))

        self.env = env
        self.n_envs = int(env.n_envs)
        self.obs_dim = int(env.observation_dim)
        self.action_dim = int(env.action_dim)
        self.action_low = float(getattr(env.action_space, "low", -1.0))
        self.action_high = float(getattr(env.action_space, "high", 1.0))
        self.action_scale = max(abs(self.action_low), abs(self.action_high))
        if self.action_scale <= 0:
            raise ValueError("invalid action range")

        self.learning_rate = float(learning_rate)
        self.buffer_size = int(buffer_size)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.action_noise = action_noise
        self.verbose = int(verbose)
        self.stats_window_size = int(stats_window_size)
        self.total_timesteps = 0
        self.gradient_step = 0
        self.policy_kwargs = policy_kwargs or {}

        net_arch = tuple(self.policy_kwargs.get("net_arch", (256, 256)))
        self.actor = DeterministicActor(self.obs_dim, self.action_dim, net_arch)
        self.actor_target = DeterministicActor(self.obs_dim, self.action_dim, net_arch)
        self.q1 = QNet(self.obs_dim, self.action_dim, net_arch)
        self.q2 = QNet(self.obs_dim, self.action_dim, net_arch)
        self.q1_target = QNet(self.obs_dim, self.action_dim, net_arch)
        self.q2_target = QNet(self.obs_dim, self.action_dim, net_arch)

        self.actor_target.update(self.actor.parameters())
        self.q1_target.update(self.q1.parameters())
        self.q2_target.update(self.q2.parameters())

        self.actor_opt = optim.Adam(learning_rate=self.learning_rate)
        self.q1_opt = optim.Adam(learning_rate=self.learning_rate)
        self.q2_opt = optim.Adam(learning_rate=self.learning_rate)
        self.replay = ReplayBuffer(self.buffer_size, self.obs_dim, self.action_dim)

        mx.eval(
            self.actor.parameters(), self.actor_target.parameters(),
            self.q1.parameters(), self.q2.parameters(),
            self.q1_target.parameters(), self.q2_target.parameters(),
        )
        self._compile_update()

        if csv_log_path:
            self.csv_path = csv_log_path
            self._logdir = os.path.dirname(csv_log_path)
        else:
            env_name = self.env.__class__.__name__.replace("MLX", "").lower()
            algo_name = self.__class__.__name__.lower()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._logdir = os.path.join("logs", ts)
            os.makedirs(self._logdir, exist_ok=True)
            self.csv_path = os.path.join(self._logdir, f"progress_{env_name}_{algo_name}_{self.n_envs}envs.csv")
        self._csv_file = None
        self._csv_writer = None
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="")

        self.reward_history = deque(maxlen=self.stats_window_size)
        self._recent_rewards = deque(maxlen=32)
        self._iterations = 0

    def _clip_action(self, action):
        return mx.clip(action, self.action_low, self.action_high)

    def _exploration_action(self, obs, deterministic=False):
        action = self.actor(obs) * self.action_scale
        if deterministic:
            return self._clip_action(action)
        if self.action_noise is not None:
            # Compatible with common NormalActionNoise-style objects when supplied.
            try:
                noise = self.action_noise(obs.shape[0])
            except TypeError:
                try:
                    noise = self.action_noise()
                except TypeError:
                    noise = 0.0
            action = action + mx.array(noise, dtype=mx.float32)
        else:
            action = action + 0.1 * self.action_scale * mx.random.normal(shape=action.shape)
        return self._clip_action(action)

    def _target_action(self, next_obs):
        action = self.actor_target(next_obs) * self.action_scale
        noise = mx.random.normal(shape=action.shape) * (
            self.target_policy_noise * self.action_scale
        )
        noise = mx.clip(
            noise,
            -self.target_noise_clip * self.action_scale,
            self.target_noise_clip * self.action_scale,
        )
        return self._clip_action(action + noise)

    def _compile_update(self):
        def q_loss(q_model, obs, actions, target):
            q = q_model(obs, actions)
            return 0.5 * mx.mean(mx.square(q - target))

        def actor_loss(actor_model, obs):
            action = actor_model(obs) * self.action_scale
            return -mx.mean(self.q1(obs, action))

        critic1_vg = nn.value_and_grad(self.q1, q_loss)
        critic2_vg = nn.value_and_grad(self.q2, q_loss)
        actor_vg = nn.value_and_grad(self.actor, actor_loss)

        state = [
            self.actor.state,
            self.actor_target.state,
            self.q1.state,
            self.q2.state,
            self.q1_target.state,
            self.q2_target.state,
            self.q1_opt.state,
            self.q2_opt.state,
        ]

        def step(obs, actions, rewards, next_obs, dones):
            next_action = self._target_action(next_obs)
            target_q1 = self.q1_target(next_obs, next_action)
            target_q2 = self.q2_target(next_obs, next_action)
            target = rewards + self.gamma * (1.0 - dones) * mx.stop_gradient(mx.minimum(target_q1, target_q2))
            loss1, q1_grads = critic1_vg(self.q1, obs, actions, target)
            loss2, q2_grads = critic2_vg(self.q2, obs, actions, target)
            self.q1_opt.update(self.q1, q1_grads)
            self.q2_opt.update(self.q2, q2_grads)

            actor_value, actor_grads = actor_vg(self.actor, obs)
            return loss1, loss2, actor_value, actor_grads

        self._update_step = mx.compile(step, inputs=state, outputs=state)

    def _log(self, timestep, start, rewards, episode_length, train):
        self._iterations += 1
        mean_reward = float(mx.mean(rewards).item())
        ep_len = float(mx.mean(episode_length).item())
        self.reward_history.append(mean_reward)
        smoothed = sum(self.reward_history) / max(1, len(self.reward_history))
        self._recent_rewards.append(smoothed)
        vals = list(self._recent_rewards)
        slope = (vals[-1] - vals[0]) / max(1, len(vals) - 1) if len(vals) > 1 else 0.0
        noise = math.sqrt(sum((v - smoothed) ** 2 for v in vals) / len(vals)) if vals else 0.0
        elapsed = max(time.perf_counter() - start, 1e-9)
        fps = int(timestep / elapsed)
        row = {
            "time/iterations": self._iterations,
            "time/fps": fps,
            "time/time_elapsed": elapsed,
            "time/total_timesteps": timestep,
            "rollout/ep_rew_mean": smoothed,
            "rollout/ep_len_mean": ep_len,
            "rollout/reward_slope": slope,
            "rollout/reward_noise": noise,
            **{f"train/{k}": v for k, v in train.items()},
        }
        if self._csv_file:
            if self._csv_writer is None:
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
                self._csv_writer.writeheader()
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        return row

    def predict(self, observation, deterministic=True):
        obs = mx.array(observation, dtype=mx.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        action = self._exploration_action(obs, deterministic=deterministic)
        mx.eval(action)
        return action

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        tb_log_name="TD3",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        del callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar
        total_timesteps = int(total_timesteps)
        obs = self.env.reset()
        start = time.perf_counter()
        pbar = tqdm(
            total=total_timesteps,
            unit="step",
            disable=(self.verbose == 0),
            dynamic_ncols=True,
            desc="MLX TD3",
        )
        latest_reward = mx.zeros((self.n_envs,), dtype=mx.float32)

        while self.total_timesteps < total_timesteps:
            action = self._exploration_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            episode_length = self.env._episode_length
            done = terminated | truncated
            self.replay.add(obs, action, reward, next_obs, done)
            obs = next_obs
            self.total_timesteps += self.n_envs
            latest_reward = reward

            if self.replay.length >= max(self.learning_starts, self.batch_size):
                for _ in range(max(1, self.gradient_steps)):
                    batch = self.replay.sample(self.batch_size)
                    self.gradient_step += 1
                    q1_loss, q2_loss, actor_loss, actor_grads = self._update_step(*batch)
                    update_actor = self.gradient_step % self.policy_delay == 0
                    _polyak(self.q1_target, self.q1, self.tau)
                    _polyak(self.q2_target, self.q2, self.tau)
                    if update_actor:
                        self.actor_opt.update(self.actor, actor_grads)
                        _polyak(self.actor_target, self.actor, self.tau)
                    mx.eval(
                        self.actor.parameters(), self.actor_opt.state,
                        self.q1.parameters(), self.q2.parameters(),
                        self.q1_opt.state, self.q2_opt.state,
                        self.actor_target.parameters(),
                        self.q1_target.parameters(), self.q2_target.parameters(),
                    )
                    if not (_finite(q1_loss) and _finite(q2_loss) and _finite(actor_loss)):
                        raise FloatingPointError("TD3 produced a non-finite loss")

                s_obs, s_actions, s_rewards, s_next_obs, s_dones = batch
                q1m = self.q1(s_obs, s_actions)
                q2m = self.q2(s_obs, s_actions)
                q1_mean = float(mx.mean(q1m).item())
                q2_mean = float(mx.mean(q2m).item())
                q_mean = (q1_mean + q2_mean) * 0.5
                next_action = self._target_action(s_next_obs)
                target = s_rewards + self.gamma * (1.0 - s_dones) * mx.minimum(self.q1_target(s_next_obs, next_action), self.q2_target(s_next_obs, next_action))
                q = mx.minimum(q1m, q2m)
                explained_var = float((1.0 - mx.var(q - target) / (mx.var(target) + 1e-8)).item())
                actor_loss_val = float(actor_loss.item())
                critic_loss_val = float(((q1_loss + q2_loss) * 0.5).item())
                train = {
                    "loss/actor": actor_loss_val,
                    "loss/critic": critic_loss_val,
                    "loss/q1": float(q1_loss.item()),
                    "loss/q2": float(q2_loss.item()),
                    "value/q1_mean": q1_mean,
                    "value/q2_mean": q2_mean,
                    "value/q_mean": q_mean,
                    "learning_rate": self.learning_rate,
                    "n_updates": self.gradient_step,
                    "explained_variance": explained_var,
                    "std": 0.0,
                    "policy": actor_loss_val,
                    "value": critic_loss_val,
                }
                row = self._log(self.total_timesteps, start, latest_reward, episode_length, train)
                if self.verbose:
                    pbar.set_postfix(
                        fps=row["time/fps"],
                        reward=f"{row['rollout/ep_rew_mean']:.1f}",
                        slope=f"{row['rollout/reward_slope']:.3f}",
                        noise=f"{row['rollout/reward_noise']:.2f}",
                    )
            pbar.update(self.n_envs)

        pbar.close()
        self._resample_csv(100)
        return self

    def save(self, path: str):
        params = {}
        modules = (
            ("actor", self.actor),
            ("q1", self.q1),
            ("q2", self.q2),
            ("actor_target", self.actor_target),
            ("q1_target", self.q1_target),
            ("q2_target", self.q2_target),
        )
        for prefix, module in modules:
            for name, value in module.parameters().items():
                params[f"{prefix}.{name}"] = value
        mx.save_safetensors(path, params)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        params = mx.load(path)
        modules = (
            ("actor", model.actor),
            ("q1", model.q1),
            ("q2", model.q2),
            ("actor_target", model.actor_target),
            ("q1_target", model.q1_target),
            ("q2_target", model.q2_target),
        )
        for prefix, module in modules:
            module.update({
                name[len(prefix) + 1:]: value
                for name, value in params.items()
                if name.startswith(prefix + ".")
            })
        mx.eval(
            model.actor.parameters(), model.actor_target.parameters(),
            model.q1.parameters(), model.q2.parameters(),
            model.q1_target.parameters(), model.q2_target.parameters(),
        )
        return model
