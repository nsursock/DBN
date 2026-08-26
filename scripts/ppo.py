"""MLX PPO, API-shaped after Stable-Baselines3, optimized for batched Apple GPU/CPU execution."""
from __future__ import annotations

import csv
import math
import os
import time
from collections import deque
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

try:
    from .agent import MLXAgent
except ImportError:
    from agent import MLXAgent


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, net_arch=(64, 64), activation="tanh"):
        super().__init__()
        self.layers = []
        dims = (in_dim,) + tuple(net_arch) + (out_dim,)
        for a, b in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(a, b))
        self.activation = activation

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = mx.tanh(x) if self.activation == "tanh" else mx.maximum(x, 0)
        return x


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, discrete, net_arch=(64, 64)):
        super().__init__()
        self.shared = MLP(obs_dim, 64, net_arch)
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)
        self.log_std = mx.zeros((action_dim,))
        self.discrete = discrete

    def forward(self, obs):
        h = self.shared(obs)
        return self.policy(h), self.value(h).squeeze(-1)

    def __call__(self, obs):
        return self.forward(obs)


class PPO(MLXAgent):
    """Drop-in-shaped PPO for CartPoleMLX and PendulumMLX.

    Common SB3 constructor parameters are accepted so existing configs can be reused.
    Unsupported visual/multi-input policies are intentionally outside this first MLX port.
    """

    def __init__(
        self,
        policy: str = "MlpPolicy",
        env=None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf=None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str = "auto",
        _init_setup_model: bool = True,
        csv_log_path: str | None = None,
        **kwargs,
    ):
        if env is None:
            raise ValueError("env is required")
        self.model = self
        self.env = env
        self.n_envs = env.n_envs
        self.obs_dim = env.observation_dim
        self.is_discrete = getattr(env.action_space, "n", None) is not None
        self.action_dim = env.action_dim if not self.is_discrete else env.action_space.n
        self.action_low = float(getattr(env.action_space, "low", -1.0)) if not self.is_discrete else -1.0
        self.action_high = float(getattr(env.action_space, "high", 1.0)) if not self.is_discrete else 1.0
        self.learning_rate = learning_rate
        self.n_steps, self.batch_size, self.n_epochs = int(n_steps), int(batch_size), int(n_epochs)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_range, self.clip_range_vf = clip_range, clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.target_kl = target_kl
        self.stats_window_size = stats_window_size
        self.verbose = verbose
        self.seed = seed
        self.total_timesteps = 0
        net_arch = (64, 64)
        if policy_kwargs:
            net_arch = tuple(policy_kwargs.get("net_arch", net_arch))
        self.policy_net = PPOActorCritic(self.obs_dim, self.action_dim, self.is_discrete, net_arch)
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        mx.eval(self.policy_net.parameters())
        self._compile_update()
        self._compile_step_and_act()
        self._compile_gae()
        self.csv_path = csv_log_path or self._csv_path_from_tensorboard(tensorboard_log)
        self._csv_file = None
        self._csv_writer = None
        if self.csv_path:
            self._open_csv(self.csv_path)
        self.reward_history = deque(maxlen=stats_window_size)
        self.ep_len_history = deque(maxlen=stats_window_size)
        self._recent_rewards = []

    @staticmethod
    def _csv_path_from_tensorboard(path):
        if not path:
            return None
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "progress.csv")

    def _open_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._csv_file = open(path, "w", newline="")
        self._csv_writer = None

    def _compile_update(self):
        def loss_fn(model, obs, actions, old_logp, advantages, returns, old_values):
            out, values = model(obs)
            if self.normalize_advantage:
                advantages = (advantages - mx.mean(advantages)) / (mx.sqrt(mx.var(advantages)) + 1e-8)
            if self.is_discrete:
                logp_all = out - mx.logsumexp(out, axis=-1, keepdims=True)
                idx = actions.astype(mx.int32)
                logp = mx.take_along_axis(logp_all, idx[:, None], axis=1).squeeze(1)
                entropy = -(mx.exp(logp_all) * logp_all).sum(axis=1)
            else:
                log_std = model.log_std
                std = mx.exp(log_std)
                logp_each = -0.5 * (((actions - out) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
                logp = logp_each.sum(axis=1)
                entropy = mx.broadcast_to(log_std + 0.5 * math.log(2 * math.pi * math.e), logp_each.shape).sum(axis=1)
            log_ratio = logp - old_logp
            log_ratio = mx.clip(log_ratio, -20.0, 20.0)
            ratio = mx.exp(log_ratio)
            pg1 = ratio * advantages
            pg2 = mx.clip(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            policy_loss = -mx.mean(mx.minimum(pg1, pg2))
            if self.clip_range_vf is not None:
                values_clipped = old_values + mx.clip(values - old_values, -self.clip_range_vf, self.clip_range_vf)
                value_loss = 0.5 * mx.mean(mx.maximum(mx.square(values - returns), mx.square(values_clipped - returns)))
            else:
                value_loss = 0.5 * mx.mean(mx.square(values - returns))
            ent_loss = -mx.mean(entropy)
            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss
            approx_kl = 0.5 * mx.mean(mx.square(old_logp - logp))
            clip_fraction = mx.mean((mx.abs(ratio - 1.0) > self.clip_range).astype(mx.float32))
            return total_loss, policy_loss, value_loss, ent_loss, approx_kl, clip_fraction

        loss_and_grad = nn.value_and_grad(self.policy_net, loss_fn)
        state = [self.policy_net.state, self.optimizer.state]

        def step(obs, actions, old_logp, advantages, returns, old_values):
            loss, grads = loss_and_grad(self.policy_net, obs, actions, old_logp, advantages, returns, old_values)
            grads = self._clip_grads(grads)
            self.optimizer.update(self.policy_net, grads)
            return loss

        self._step_impl = mx.compile(step, inputs=state, outputs=state)

    def _clip_grads(self, grads):
        leaves = [v for _, v in tree_flatten(grads)]
        sq = sum(mx.sum(mx.square(g)) for g in leaves)
        norm = mx.sqrt(sq + 1e-8)
        scale = mx.minimum(mx.array(1.0), mx.array(self.max_grad_norm) / norm)
        return tree_map(lambda g: g * scale, grads)

    def _compile_step_and_act(self):
        state = [self.policy_net.state]

        def step_and_act(obs, episode_length, policy_key, reset_key):
            pkey, skey = mx.random.split(policy_key)
            if self.is_discrete:
                u = mx.random.uniform(shape=(obs.shape[0], 1), key=skey)
                action, logp, value = self._act(obs, u=u)
            else:
                noise = mx.random.normal(shape=(obs.shape[0], self.action_dim), key=skey)
                action, logp, value = self._act(obs, noise=noise)
            rkey, rk = mx.random.split(reset_key)
            reset_state = self.env._reset_state(key=rk)
            next_obs, reward, terminated, truncated, next_ep_len = self.env._dynamics(obs, episode_length, action, reset_state)
            done = terminated | truncated
            return next_obs, next_ep_len, action, logp, value, reward, done, pkey, rkey

        self._step_fn = mx.compile(step_and_act, inputs=state, outputs=state)

    def _compile_gae(self):
        def gae_fn(rew_b, done_b, val_b, last_value, T):
            advantages = []
            gae = mx.zeros((self.n_envs,), dtype=mx.float32)
            for t in range(T - 1, -1, -1):
                next_value = last_value if t == T - 1 else val_b[t + 1]
                nonterminal = 1.0 - done_b[t]
                gae = (
                    rew_b[t]
                    + self.gamma * next_value * nonterminal
                    - val_b[t]
                    + self.gamma * self.gae_lambda * nonterminal * gae
                )
                advantages.append(gae)
            advantages = mx.stack(advantages[::-1], axis=0)
            returns = advantages + val_b
            return advantages, returns

        self._gae_fn = mx.compile(gae_fn)

    def _act(self, obs, deterministic=False, u=None, noise=None):
        logits, value = self.policy_net(obs)
        if self.is_discrete:
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            probs = mx.exp(log_probs)
            if u is not None:
                action = mx.sum(u > mx.cumsum(probs, axis=1), axis=1).astype(mx.int32)
            elif deterministic:
                action = mx.argmax(logits, axis=1)
            else:
                u = mx.random.uniform(shape=(obs.shape[0], 1))
                cdf = mx.cumsum(probs, axis=1)
                action = mx.sum(u > cdf, axis=1).astype(mx.int32)
            lp = mx.take_along_axis(log_probs, action[:, None], axis=1).squeeze(1)
        else:
            std = mx.exp(self.policy_net.log_std)
            if noise is not None:
                raw = logits + std * noise
            elif deterministic:
                raw = logits
            else:
                raw = logits + std * mx.random.normal(shape=logits.shape)
            action = mx.clip(raw, self.action_low, self.action_high)
            lp = (-0.5 * (((action - logits) / (std + 1e-8)) ** 2 + 2 * mx.log(std + 1e-8) + math.log(2 * math.pi))).sum(axis=1)
        return action, lp, value

    def predict(self, observation, deterministic=True):
        obs = mx.array(observation, dtype=mx.float32)
        a, _, _ = self._act(obs, deterministic=deterministic)
        mx.eval(a)
        return a

    def _write_row(self, row):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _log(self, timestep, elapsed, rewards, episode_lengths, train):
        ep_reward = float(mx.mean(rewards).item())
        ep_len = float(mx.mean(episode_lengths).item())
        self.reward_history.extend([ep_reward])
        mean_reward = sum(self.reward_history) / len(self.reward_history)
        if len(self._recent_rewards) >= 8:
            ys = self._recent_rewards[-32:] + [mean_reward]
            slope = (ys[-1] - ys[0]) / max(1, len(ys) - 1)
            noise = math.sqrt(sum((y - mean_reward) ** 2 for y in ys) / len(ys))
        else:
            slope, noise = 0.0, 0.0
        self._recent_rewards.append(mean_reward)
        fps = int(timestep / max(elapsed, 1e-9))
        row = {
            "time/fps": fps,
            "time/time_elapsed": elapsed,
            "time/total_timesteps": timestep,
            "rollout/ep_rew_mean": mean_reward,
            "rollout/ep_len_mean": ep_len,
            "rollout/reward_slope": slope,
            "rollout/reward_noise": noise,
            **{f"train/{k}": v for k, v in train.items()},
        }
        self._write_row(row)
        return row

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 1, tb_log_name="PPO", reset_num_timesteps=True, progress_bar=False):
        del callback, tb_log_name, reset_num_timesteps, progress_bar
        obs = self.env.reset()
        episode_length = self.env._episode_length
        seed = self.seed if self.seed is not None else 0
        policy_key = mx.random.key(seed)
        reset_key = mx.random.key(seed + 1)
        start_time = time.perf_counter()
        pbar = tqdm(total=total_timesteps, unit="step", disable=(self.verbose == 0), dynamic_ncols=True, desc="MLX PPO")
        while self.total_timesteps < total_timesteps:
            T = min(self.n_steps, max(1, (total_timesteps - self.total_timesteps + self.n_envs - 1) // self.n_envs))
            obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
            for _ in range(T):
                next_obs, next_ep_len, action, logp, value, reward, done, policy_key, reset_key = self._step_fn(obs, episode_length, policy_key, reset_key)
                obs_buf.append(obs)
                act_buf.append(action)
                rew_buf.append(reward)
                done_buf.append(done)
                logp_buf.append(logp)
                val_buf.append(value)
                obs = next_obs
                episode_length = next_ep_len
            _, _, last_value = self._act(obs, deterministic=True)
            obs_b = mx.stack(obs_buf)
            act_b = mx.stack(act_buf)
            rew_b = mx.stack(rew_buf)
            done_b = mx.stack(done_buf).astype(mx.float32)
            logp_b = mx.stack(logp_buf)
            val_b = mx.stack(val_buf)
            mx.eval(obs_b, act_b, rew_b, done_b, logp_b, val_b, last_value)

            advantages, returns = self._gae_fn(rew_b, done_b, val_b, last_value, T)
            mx.eval(advantages, returns)
            flat_obs = obs_b.reshape((-1, self.obs_dim))
            flat_act = act_b.reshape((-1,) if self.is_discrete else (-1, self.action_dim))
            flat_logp = logp_b.reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)
            flat_val = val_b.reshape(-1)
            mx.eval(flat_obs, flat_act, flat_logp, flat_adv, flat_ret, flat_val)

            n = T * self.n_envs
            target_batches = 32
            n_batches = max(1, min(target_batches, n // self.batch_size))
            mb = (n + n_batches - 1) // n_batches
            total_updates = n_batches * self.n_epochs
            pg, vl, ent, akl, cf = 0.0, 0.0, 0.0, 0.0, 0.0
            for _ in range(self.n_epochs):
                for b in range(n_batches):
                    s = b * mb
                    e = min(s + mb, n)
                    result = self._step_impl(
                        flat_obs[s:e],
                        flat_act[s:e],
                        flat_logp[s:e],
                        flat_adv[s:e],
                        flat_ret[s:e],
                        flat_val[s:e],
                    )
                    pg = pg + result[1]
                    vl = vl + result[2]
                    ent = ent - result[3]
                    akl = akl + result[4]
                    cf = cf + result[5]
            mx.eval(self.policy_net.state, self.optimizer.state, pg, vl, ent, akl, cf)
            if not mx.all(mx.isfinite(mx.stack([pg, vl, ent]))).item():
                raise RuntimeError(f"non-finite losses: pg={pg} vl={vl} ent={ent}")
            train = {
                "policy_gradient_loss": float(pg.item()) / total_updates,
                "value_loss": float(vl.item()) / total_updates,
                "entropy_loss": float(ent.item()) / total_updates,
                "approx_kl": float(akl.item()) / total_updates,
                "clip_fraction": float(cf.item()) / total_updates,
                "clip_range": self.clip_range,
                "n_updates": total_updates,
            }
            self.total_timesteps += T * self.n_envs
            pbar.update(T * self.n_envs)
            row = self._log(self.total_timesteps, time.perf_counter() - start_time, rew_b[-1], mx.ones((self.n_envs,)) * T, train)
            pbar.set_postfix(fps=row["time/fps"], reward=f"{row['rollout/ep_rew_mean']:.1f}", slope=f"{row['rollout/reward_slope']:.3f}", noise=f"{row['rollout/reward_noise']:.2f}")
        pbar.close()
        self.env.state = obs
        self.env._episode_length = episode_length
        return self

    def save(self, path):
        mx.save_safetensors(path, dict(self.policy_net.parameters()))

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        params = mx.load(path)
        model.policy_net.update(params)
        mx.eval(model.policy_net.parameters())
        return model
