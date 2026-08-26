In Stable-Baselines3 (SB3), both SAC and PPO log common stats under `time/`, `rollout/`, and `train/` in TensorBoard. The key difference is that `train/` contains algorithm-specific metrics that reflect their distinct learning approaches.

### 📊 Common Stats: `time/` and `rollout/`

The stats in `time/` and `rollout/` are fundamental and appear for both algorithms.

| Category | Common Stats | Description |
| :--- | :--- | :--- |
| **`time/`** | `fps`, `iterations`, `time_elapsed`, `total_timesteps` | Performance and progress metrics: frames/steps per second, number of updates, time passed, and total environment interactions. |
| **`rollout/`** | `ep_rew_mean`, `ep_len_mean`, `success_rate` | Episode performance: mean reward and episode length (averaged over a rolling window), and success rate for goal-based tasks. |

These are logged via the `Monitor` wrapper, so it's crucial to wrap your environment with it to see this data.

### 🔍 What Differs: Algorithm-Specific `train/` Stats

The `train/` section is where the algorithms diverge, showing their unique loss calculations and optimization processes.

| PPO `train/` Stats | SAC `train/` Stats | Description |
| :--- | :--- | :--- |
| `approx_kl`, `clip_fraction`, `clip_range` | — | PPO-specific trust region metrics. |
| `policy_gradient_loss` | `loss/policy` | Policy network loss. |
| `value_loss` (based on Monte Carlo or TD(λ) estimate) | `critic_loss` (based on TD(0) estimate) | Value function loss. PPO uses on-policy updates; SAC uses off-policy TD learning. |
| — | `loss/q1`, `loss/q2`, `loss/alpha` | SAC's dual Q-networks and temperature loss for entropy tuning. |
| — | `policy/alpha`, `value/q_mean`, `policy/log_pi_mean` | SAC's learned entropy coefficient, mean Q-values, and action log-probabilities. |
| `entropy_loss` | `ent_coef_loss` | Regularization to encourage exploration. |
| `learning_rate`, `loss`, `n_updates`, `explained_variance`, `std` | `learning_rate`, `loss`, `n_updates`, `explained_variance`, `std` | Other common training stats (depending on configuration, e.g., `std` for gSDE). |

Note: There was a historical issue where SAC might not log metrics to TensorBoard automatically in some versions. If you encounter this, using a custom callback is a reliable workaround.