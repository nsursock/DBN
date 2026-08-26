# DBN — Developing a Trading Bot using zero AI fees

DBN is an experimental, self-contained Python framework for training and evaluating reinforcement-learning agents locally on Apple Silicon. It is built on top of [MLX](https://github.com/ml-explore/mlx) and uses Stable-Baselines3-shaped APIs so existing configs, notebooks and muscle memory can be reused without any cloud inference, per-token fees or paid GPU subscriptions.

The codebase is intentionally split between:

- **RL agents and environments** (`scripts/`, `tests/`) for rapid prototyping on classic control tasks.
- **Synthetic market data** (`scripts/data.py`, `scripts/viz.py`, `configs/`) for the trading-bot direction.
- **Native benchmarking tools** (`utils/bench/`) for measuring throughput, memory and time-to-solve on MLX.

## Features

- **Native MLX training** — PPO, SAC and TD3 written in pure MLX with batched, vectorized updates.
- **SB3-style API** — familiar constructor arguments (`learning_rate`, `gamma`, `buffer_size`, `batch_size`, `tensorboard_log`, ...).
- **Vectorized environments** — pure-MLX `CartPoleMLX` and `PendulumMLX` that scale to tens of thousands of parallel envs.
- **Throughput & TTS benchmarks** — `utils/bench/scale.py` and `utils/bench/solve.py` with auto-doubling and plateau detection.
- **Live thermal telemetry** — `smctemp`, `powermetrics` and `osx-cpu-temp` support via a best-effort fallback chain.
- **Synthetic market data** — OHLC, volume, returns, volatility, funding and alpha channels generated as MLX tensors.
- **Minimal footprint** — see `requirements.txt`.

## Requirements

- Python 3.11 (see `.python-version`)
- macOS with Apple Silicon (MLX targets Metal)
- Optional but recommended: `smctemp` (`brew install narugit/smctemp/smctemp`) for live CPU temperature

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```text
.
├── scripts/           # Agents, data generation and visualization
│   ├── agent.py       # Common Agent / MLXAgent interface
│   ├── ppo.py         # MLX PPO
│   ├── sac.py         # MLX SAC
│   ├── td3.py         # MLX TD3
│   ├── data.py        # Synthetic market data
│   └── viz.py         # Plotly candlestick charts
├── tests/             # Vectorized envs and pytest suite
│   ├── cartpole.py    # CartPoleMLX
│   ├── pendulum.py    # PendulumMLX
│   └── test_*.py
├── utils/bench/       # Benchmarking harness
│   ├── common.py      # Shared MLX / telemetry helpers
│   ├── scale.py       # Throughput scaling sweep
│   └── solve.py       # Time-to-solve benchmark
├── configs/           # Market simulation config and Plotly themes
│   ├── config.yaml
│   └── themes/
└── README.md
```

## Quick start

### 1. Throughput benchmark

```bash
.venv/bin/python utils/bench/scale.py normal \
  --env cartpole --algo ppo --max-envs 100_000
```

This auto-doubles `n_envs` until throughput plateaus and reports env/train FPS, RSS/peak memory and thermal state.

### 2. Time-to-solve benchmark

```bash
.venv/bin/python utils/bench/solve.py \
  --target pendulum_sac --envs 32 128 256
```

`--target` can be `cartpole_ppo`, `pendulum_ppo`, `pendulum_sac`, `pendulum_td3` or `all`.

### 3. Smoke tests

```bash
.venv/bin/python -m pytest
```

Or use the built-in smoke flags:

```bash
.venv/bin/python utils/bench/scale.py normal --smoke
.venv/bin/python utils/bench/solve.py --smoke
```

## Example: synthetic market data

```bash
.venv/bin/python - <<'PY'
from scripts.data import generate_market_data, load_config
from scripts.viz import plot_market_data

config = load_config("configs/config.yaml")
tensor, symbols = generate_market_data(config)
plot_market_data(tensor, symbols, theme_name="synthwave", symbol="BTC")
PY
```

Output is written to `outputs/market/{theme_name}.png`.

## Agents

All agents follow a minimal `predict / learn / save / load` interface:

```python
from tests.cartpole import CartPoleMLX
from scripts.ppo import PPO

env = CartPoleMLX(n_envs=64, seed=0)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/cartpole")
model.learn(100_000)
model.save("checkpoints/cartpole_ppo.safetensors")
```

SAC and TD3 work the same way and expect a continuous `PendulumMLX` environment:

```python
from tests.pendulum import PendulumMLX
from scripts.sac import SAC

env = PendulumMLX(n_envs=64, seed=0)
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="runs/pendulum")
model.learn(100_000)
```

## Benchmarks

`utils/bench/common.py` provides shared helpers for:

- RSS and peak memory tracking
- Best-effort thermal monitoring (`smctemp -c`, `sudo powermetrics --samplers thermal`, `osx-cpu-temp`)
- Stable-Baselines3-shaped model and environment construction

### Sample throughput run

CartPole PPO on Apple Silicon, `n_envs` auto-doubling from 16 to 8192:

| env      | algo   |   n_envs |    env FPS |   train FPS |   RSS MB |   peak MB |   ΔRSS MB |   thermal |   wall s |
|----------|--------|----------|------------|-------------|----------|-----------|-----------|-----------|----------|
| cartpole | PPO    |       16 |     50,291 |      84,208 |    313.6 |     313.6 |       257 |      66.4 |     4.75 |
| cartpole | PPO    |       32 |    173,541 |     166,474 |    331.6 |     331.6 |        18 |      69.3 |     4.81 |
| cartpole | PPO    |       64 |    357,939 |     306,718 |    340.3 |     340.3 |       8.6 |      70.8 |     3.26 |
| cartpole | PPO    |      128 |    716,405 |     529,114 |    349.2 |     349.2 |       8.9 |      69.9 |     1.89 |
| cartpole | PPO    |      256 |  1,426,155 |   1,101,501 |    366.6 |     366.6 |      17.4 |      71.2 |     0.91 |
| cartpole | PPO    |      512 |  2,888,197 |   1,750,960 |    370.2 |     370.2 |       3.6 |      69.9 |     0.57 |
| cartpole | PPO    |     1024 |  5,671,038 |   3,115,224 |    373.3 |     373.3 |       3.1 |      69.9 |     0.32 |
| cartpole | PPO    |     2048 | 11,231,887 |   3,610,902 |    380.6 |     380.6 |       7.3 |      70   |     0.28 |
| cartpole | PPO    |     4096 | 22,005,588 |   3,584,210 |    385.9 |     385.9 |       5.3 |      69.9 |     0.28 |
| cartpole | PPO    |     8192 | 41,560,339 |   3,570,957 |    385.9 |     385.9 |         0 |      68.8 |     0.28 |

## Tips

- **MLX graph memory** grows during the first compile; the RSS delta you see on small `n_envs` is usually one-time compilation cost.
- **Thermal readings** require `smctemp` or a `sudoers` NOPASSWD rule for `/usr/bin/powermetrics` if you want the `powermetrics` fallback to work passwordlessly.
- **Vectorization** is the main lever for throughput: `CartPoleMLX` and `PendulumMLX` are designed to step thousands of envs in a single MLX call.

## License

MIT
