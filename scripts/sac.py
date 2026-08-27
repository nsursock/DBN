"""MLX SAC, API-shaped after Stable-Baselines3, optimized for batched environments and replay."""
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
    def __init__(self, in_dim, out_dim, net_arch=(256, 256)):
        super().__init__()
        self.layers = []
        dims = (in_dim,) + tuple(net_arch) + (out_dim,)
        for a, b in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(a, b))

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = mx.maximum(x, 0)
        return x


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, net_arch=(256, 256)):
        super().__init__()
        self.body = MLP(obs_dim, 2 * action_dim, net_arch)
        self.action_dim = action_dim

    def __call__(self, obs):
        x = self.body(obs)
        mean, log_std = mx.split(x, 2, axis=-1)
        return mean, mx.clip(log_std, -20.0, 2.0)


class QNet(nn.Module):
    def __init__(self, obs_dim, action_dim, net_arch=(256, 256)):
        super().__init__()
        self.body = MLP(obs_dim + action_dim, 1, net_arch)

    def __call__(self, obs, action):
        return self.body(mx.concatenate([obs, action], axis=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, size, obs_dim, action_dim):
        self.size = int(size)
        self.obs = mx.zeros((self.size, obs_dim), dtype=mx.float32)
        self.next_obs = mx.zeros((self.size, obs_dim), dtype=mx.float32)
        self.actions = mx.zeros((self.size, action_dim), dtype=mx.float32)
        self.rewards = mx.zeros((self.size,), dtype=mx.float32)
        self.dones = mx.zeros((self.size,), dtype=mx.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        n = obs.shape[0]
        idx = (mx.arange(n) + self.pos) % self.size
        mx.eval(obs, action, reward, next_obs, done)
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done.astype(mx.float32)
        self.pos = (self.pos + n) % self.size
        if self.pos == 0:
            self.full = True
        mx.eval(self.obs, self.actions, self.rewards, self.next_obs, self.dones)

    @property
    def length(self):
        return self.size if self.full else self.pos

    def sample(self, batch_size):
        idx = mx.random.randint(0, self.length, shape=(batch_size,))
        return (
            mx.take(self.obs, idx, axis=0),
            mx.take(self.actions, idx, axis=0),
            mx.take(self.rewards, idx, axis=0),
            mx.take(self.next_obs, idx, axis=0),
            mx.take(self.dones, idx, axis=0),
        )


class SAC(MLXAgent):
    def __init__(
        self,
        policy: str = "MlpPolicy",
        env=None,
        learning_rate=3e-4,
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
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
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
        if env is None:
            raise ValueError("env is required")
        if getattr(env.action_space, "n", None) is not None:
            raise ValueError("SAC requires a continuous Box-like action space")
        self.model = self
        self.env = env
        self.n_envs = env.n_envs
        self.obs_dim = env.observation_dim
        self.action_dim = env.action_dim
        self.action_low = float(getattr(env.action_space, "low", -1.0))
        self.action_high = float(getattr(env.action_space, "high", 1.0))
        self.action_scale = max(abs(self.action_low), abs(self.action_high))
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.learning_starts = int(learning_starts)
        self.tau, self.gamma = tau, gamma
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.target_update_interval = int(target_update_interval)
        self.verbose = verbose
        self.seed = seed
        self.stats_window_size = stats_window_size
        self.total_timesteps = 0
        self.policy_kwargs = policy_kwargs or {}
        net_arch = tuple(self.policy_kwargs.get("net_arch", (256, 256)))
        self.actor = GaussianActor(self.obs_dim, self.action_dim, net_arch)
        self.q1 = QNet(self.obs_dim, self.action_dim, net_arch)
        self.q2 = QNet(self.obs_dim, self.action_dim, net_arch)
        self.tq1 = QNet(self.obs_dim, self.action_dim, net_arch)
        self.tq2 = QNet(self.obs_dim, self.action_dim, net_arch)
        self.tq1.update(self.q1.parameters())
        self.tq2.update(self.q2.parameters())
        self.actor_opt = optim.Adam(learning_rate=learning_rate)
        self.q1_opt = optim.Adam(learning_rate=learning_rate)
        self.q2_opt = optim.Adam(learning_rate=learning_rate)
        self.auto_entropy = ent_coef == "auto" or (isinstance(ent_coef, str) and ent_coef.startswith("auto"))
        initial_alpha = 1.0 if self.auto_entropy else float(ent_coef)
        self.log_alpha = mx.array(math.log(max(initial_alpha, 1e-8)), dtype=mx.float32)
        self.target_entropy = -float(self.action_dim) if target_entropy == "auto" else float(target_entropy)
        self.replay = ReplayBuffer(self.buffer_size, self.obs_dim, self.action_dim)
        mx.eval(self.actor.parameters(), self.q1.parameters(), self.q2.parameters(), self.tq1.parameters(), self.tq2.parameters())
        self._compile_update()
        self._compile_step_and_act()
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
        self.reward_history = deque(maxlen=stats_window_size)
        self._recent_rewards = []
        self._iterations = 0
        self.n_updates = 0

    def _sample_action(self, obs, deterministic=False, noise=None):
        mean, log_std = self.actor(obs)
        std = mx.exp(log_std)
        if noise is not None:
            raw = mean + std * noise
        elif deterministic:
            raw = mean
        else:
            raw = mean + std * mx.random.normal(shape=mean.shape)
        action = mx.tanh(raw) * self.action_scale
        logp = (-0.5 * (((raw - mean) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))).sum(axis=1)
        tanh_unit = mx.tanh(raw)
        logp = logp - mx.sum(mx.log(1.0 - tanh_unit * tanh_unit + 1e-6), axis=1) - self.action_dim * math.log(self.action_scale)
        return action, logp

    def _compile_update(self):
        def q_loss(model, obs, actions, rewards, next_obs, dones, log_alpha):
            next_a, next_logp = self._sample_action(next_obs, deterministic=False)
            alpha = mx.exp(log_alpha)
            target = mx.minimum(self.tq1(next_obs, next_a), self.tq2(next_obs, next_a)) - alpha * next_logp
            target = rewards + self.gamma * (1.0 - dones) * target
            return 0.5 * mx.mean(mx.square(model(obs, actions) - mx.stop_gradient(target)))

        def actor_loss(model, obs, log_alpha):
            a, lp = self._sample_action_with_actor(model, obs, deterministic=False)
            q = mx.minimum(self.q1(obs, a), self.q2(obs, a))
            return mx.mean(mx.exp(log_alpha) * lp - q)

        q1_vg = nn.value_and_grad(self.q1, q_loss)
        q2_vg = nn.value_and_grad(self.q2, q_loss)
        actor_vg = nn.value_and_grad(self.actor, actor_loss)
        state = [self.actor.state, self.q1.state, self.q2.state,
                 self.tq1.state, self.tq2.state,
                 self.actor_opt.state, self.q1_opt.state, self.q2_opt.state]

        def step(log_alpha, obs, actions, rewards, next_obs, dones):
            q1_loss, q1_grad = q1_vg(self.q1, obs, actions, rewards, next_obs, dones, log_alpha)
            q2_loss, q2_grad = q2_vg(self.q2, obs, actions, rewards, next_obs, dones, log_alpha)
            self.q1_opt.update(self.q1, q1_grad)
            self.q2_opt.update(self.q2, q2_grad)
            actor_loss_value, actor_grad = actor_vg(self.actor, obs, log_alpha)
            self.actor_opt.update(self.actor, actor_grad)
            _, alpha_grad = mx.value_and_grad(
                lambda la: -mx.mean(
                    la * mx.stop_gradient(
                        self._sample_action_with_actor(self.actor, obs, deterministic=False)[1]
                        + self.target_entropy
                    )
                )
            )(log_alpha)
            return q1_loss, q2_loss, actor_loss_value, alpha_grad

        self._update_step = mx.compile(step, inputs=state, outputs=state)

    def _compile_step_and_act(self):
        state = [self.actor.state]

        def step_and_act(obs, episode_length, policy_key, reset_key):
            pkey, skey = mx.random.split(policy_key)
            noise = mx.random.normal(shape=(obs.shape[0], self.action_dim), key=skey)
            action, _ = self._sample_action(obs, noise=noise)
            rkey, rk = mx.random.split(reset_key)
            reset_state = self.env._reset_state(key=rk)
            next_obs, reward, terminated, truncated, next_ep_len = self.env._dynamics(obs, episode_length, action, reset_state)
            done = terminated | truncated
            return next_obs, next_ep_len, action, reward, done, pkey, rkey

        self._step_fn = mx.compile(step_and_act, inputs=state, outputs=state)

    def _sample_action_with_actor(self, actor, obs, deterministic=False):
        mean, log_std = actor(obs)
        std = mx.exp(log_std)
        raw = mean if deterministic else mean + std * mx.random.normal(shape=mean.shape)
        action = mx.tanh(raw) * self.action_scale
        logp = (-0.5 * (((raw - mean) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))).sum(axis=1)
        tanh_unit = mx.tanh(raw)
        logp = logp - mx.sum(mx.log(1.0 - tanh_unit * tanh_unit + 1e-6), axis=1) - self.action_dim * math.log(self.action_scale)
        return action, logp

    def predict(self, observation, deterministic=True):
        obs = mx.array(observation, dtype=mx.float32)
        action, _ = self._sample_action(obs, deterministic=deterministic)
        mx.eval(action)
        return action

    def _log(self, timestep, start, rewards, episode_length, train):
        self._iterations += 1
        mean_reward = float(mx.mean(rewards).item())
        ep_len = float(mx.mean(episode_length).item())
        self.reward_history.append(mean_reward)
        smoothed = sum(self.reward_history) / len(self.reward_history)
        self._recent_rewards.append(smoothed)
        ys = self._recent_rewards[-32:]
        slope = (ys[-1] - ys[0]) / max(1, len(ys) - 1) if len(ys) > 1 else 0.0
        noise = math.sqrt(sum((y - smoothed) ** 2 for y in ys) / len(ys)) if ys else 0.0
        fps = int(timestep / max(time.perf_counter() - start, 1e-9))
        row = {
            "time/iterations": self._iterations,
            "time/fps": fps,
            "time/time_elapsed": time.perf_counter() - start,
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

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 1, tb_log_name="SAC", reset_num_timesteps=True, progress_bar=False):
        del callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar
        obs = self.env.reset()
        episode_length = self.env._episode_length
        seed = self.seed if self.seed is not None else 0
        policy_key = mx.random.key(seed)
        reset_key = mx.random.key(seed + 1)
        start = time.perf_counter()
        pbar = tqdm(total=total_timesteps, unit="step", disable=(self.verbose == 0), dynamic_ncols=True, desc="MLX SAC")
        latest_reward = mx.zeros((self.n_envs,), dtype=mx.float32)
        while self.total_timesteps < total_timesteps:
            next_obs, next_ep_len, action, reward, done, policy_key, reset_key = self._step_fn(obs, episode_length, policy_key, reset_key)
            mx.eval(next_obs, next_ep_len, action, reward, done)
            self.replay.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_length = next_ep_len
            self.total_timesteps += self.n_envs
            latest_reward = reward
            if self.replay.length >= max(self.learning_starts, self.batch_size):
                for _ in range(self.gradient_steps):
                    batch = self.replay.sample(self.batch_size)
                    q1l, q2l, al, alph_grad = self._update_step(
                        self.log_alpha, *batch
                    )
                    self.n_updates += 1
                    if self.auto_entropy:
                        self.log_alpha = self.log_alpha - self.learning_rate * alph_grad
                    mx.eval(self.actor.parameters(), self.q1.parameters(), self.q2.parameters(), self.actor_opt.state, self.q1_opt.state, self.q2_opt.state, self.log_alpha)
                if self.total_timesteps // self.n_envs % self.target_update_interval == 0:
                    self.tq1.update(tree_map(lambda a, b: self.tau * a + (1.0 - self.tau) * b, self.q1.parameters(), self.tq1.parameters()))
                    self.tq2.update(tree_map(lambda a, b: self.tau * a + (1.0 - self.tau) * b, self.q2.parameters(), self.tq2.parameters()))
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = batch
                q1m = self.q1(s_obs, s_actions)
                q2m = self.q2(s_obs, s_actions)
                q1_mean = float(mx.mean(q1m).item())
                q2_mean = float(mx.mean(q2m).item())
                q_mean = (q1_mean + q2_mean) * 0.5
                _, logp = self._sample_action_with_actor(self.actor, s_obs, deterministic=False)
                log_pi_mean = float(mx.mean(logp).item())
                alpha = float(mx.exp(self.log_alpha).item())
                alpha_loss = float((-self.log_alpha * (mx.mean(logp) + self.target_entropy)).item())
                _, log_std = self.actor(s_obs)
                std = float(mx.mean(mx.exp(log_std)).item())
                next_a, next_logp = self._sample_action(s_next_obs, deterministic=False)
                target = s_rewards + self.gamma * (1.0 - s_dones) * (mx.minimum(self.tq1(s_next_obs, next_a), self.tq2(s_next_obs, next_a)) - alpha * next_logp)
                q = mx.minimum(q1m, q2m)
                explained_var = float((1.0 - mx.var(q - target) / (mx.var(target) + 1e-8)).item())
                train = {
                    "loss/policy": float(al.item()),
                    "loss/critic": (float(q1l.item()) + float(q2l.item())) * 0.5,
                    "loss/q1": float(q1l.item()),
                    "loss/q2": float(q2l.item()),
                    "loss/alpha": alpha_loss,
                    "policy/alpha": alpha,
                    "policy/log_pi_mean": log_pi_mean,
                    "value/q_mean": q_mean,
                    "value/q1_mean": q1_mean,
                    "value/q2_mean": q2_mean,
                    "learning_rate": self.learning_rate,
                    "n_updates": self.n_updates,
                    "explained_variance": explained_var,
                    "std": std,
                }
                row = self._log(self.total_timesteps, start, latest_reward, episode_length, train)
                pbar.set_postfix(fps=row["time/fps"], reward=f"{row['rollout/ep_rew_mean']:.1f}", slope=f"{row['rollout/reward_slope']:.3f}", noise=f"{row['rollout/reward_noise']:.2f}")
            pbar.update(self.n_envs)
        pbar.close()
        self.env.state = obs
        self.env._episode_length = episode_length
        self._resample_csv(100)
        return self

    def save(self, path):
        params = {}
        for prefix, module in (("actor", self.actor), ("q1", self.q1), ("q2", self.q2)):
            for name, value in module.parameters().items():
                params[f"{prefix}.{name}"] = value
        mx.save_safetensors(path, params)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        params = mx.load(path)
        for prefix, module in (("actor", model.actor), ("q1", model.q1), ("q2", model.q2)):
            module.update({name[len(prefix) + 1:]: value for name, value in params.items() if name.startswith(prefix + ".")})
        model.tq1.update(model.q1.parameters())
        model.tq2.update(model.q2.parameters())
        mx.eval(model.actor.parameters(), model.q1.parameters(), model.q2.parameters())
        return model
