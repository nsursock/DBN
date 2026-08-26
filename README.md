# DBN — Developing a Trading Bot using zero AI fees

DBN is an experimental Python framework for training and evaluating self-hosted trading bots with reinforcement learning. The goal is to keep the workflow entirely local — no paid AI API calls, no cloud inference, no per-token fees — while still getting competitive throughput on Apple Silicon.

## Features

- Pure, local MLX-based RL training
- Vectorized environment rollouts
- PPO, SAC and other baseline algorithms
- Minimal dependency footprint
- Built-in FPS/throughput benchmarks

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run a throughput benchmark for a given environment and algorithm:

```bash
PYTHONPATH=scripts:tests:utils .venv/bin/python utils/bench_fps.py normal \
  --env cartpole --algo ppo --max-envs 16384
```

Or with SAC on a continuous control task:

```bash
PYTHONPATH=scripts:tests:utils .venv/bin/python utils/bench_fps.py normal \
  --env pendulum --algo sac --max-envs 100000
```

## Benchmarks

The tables below were produced by `utils/bench_fps.py` on Apple Silicon in `normal` mode.

### CartPole — PPO

| env      | algo   |   n_envs |    env FPS |   train FPS |   RSS MB |   peak MB |   ΔRSS MB | temp °C   |   wall s |
|----------|--------|----------|------------|-------------|----------|-----------|-----------|-----------|----------|
| cartpole | PPO    |       16 |     53,322 |      72,361 |    307   |     307   |     249.2 | N/A       |     2.83 |
| cartpole | PPO    |       32 |    175,332 |     101,863 |    321.1 |     321.1 |      14.1 | N/A       |     4.02 |
| cartpole | PPO    |       64 |    349,578 |     198,614 |    337.9 |     337.9 |      16.8 | N/A       |     4.13 |
| cartpole | PPO    |      128 |    704,116 |     382,621 |    346.3 |     346.3 |       8.4 | N/A       |     2.61 |
| cartpole | PPO    |      256 |  1,415,516 |     739,469 |    352.5 |     352.5 |       6.3 | N/A       |     1.35 |
| cartpole | PPO    |      512 |  2,779,393 |   1,091,651 |    363.4 |     363.4 |      10.8 | N/A       |     0.92 |
| cartpole | PPO    |     1024 |  5,515,208 |   1,375,326 |    378.5 |     378.5 |      15.1 | N/A       |     0.73 |
| cartpole | PPO    |     2048 | 11,097,747 |   1,530,263 |    394.6 |     394.6 |      16.2 | N/A       |     0.66 |
| cartpole | PPO    |     4096 | 20,588,072 |   1,643,364 |    411.6 |     411.6 |      16.9 | N/A       |     0.61 |
| cartpole | PPO    |     8192 | 41,555,064 |   1,564,714 |    421.9 |     421.9 |      10.3 | N/A       |     0.65 |
| cartpole | PPO    |    16384 | 80,663,622 |   1,453,994 |    424.8 |     424.8 |       2.8 | N/A       |     0.7  |

### Pendulum — SAC

| env      | algo   |   n_envs |     env FPS |   train FPS |   RSS MB |   peak MB |   ΔRSS MB | temp °C   |   wall s |
|----------|--------|----------|-------------|-------------|----------|-----------|-----------|-----------|----------|
| pendulum | SAC    |       16 |      77,789 |       2,356 |    240.7 |     240.7 |     182.5 | N/A       |     8.49 |
| pendulum | SAC    |       32 |     224,830 |      14,910 |    250.4 |     250.4 |       9.7 | N/A       |     2.15 |
| pendulum | SAC    |       64 |     473,468 |      28,951 |    252.2 |     252.2 |       1.8 | N/A       |     2.21 |
| pendulum | SAC    |      128 |     947,372 |      55,500 |    254.2 |     254.2 |       1.9 | N/A       |     2.31 |
| pendulum | SAC    |      256 |   1,906,301 |     112,907 |    256   |     256   |       1.8 | N/A       |     2.27 |
| pendulum | SAC    |      512 |   3,815,842 |     218,949 |    256.3 |     256.3 |       0.3 | N/A       |     2.34 |
| pendulum | SAC    |     1024 |   7,599,915 |     437,211 |    258.1 |     258.1 |       1.8 | N/A       |     2.29 |
| pendulum | SAC    |     2048 |  15,122,280 |     838,733 |    260.3 |     260.3 |       2.2 | N/A       |     1.19 |
| pendulum | SAC    |     4096 |  30,213,067 |   1,459,606 |    262.6 |     262.6 |       2.3 | N/A       |     0.69 |
| pendulum | SAC    |     8192 |  60,141,434 |   2,428,536 |    262.8 |     262.8 |       0.2 | N/A       |     0.42 |
| pendulum | SAC    |    16384 | 118,482,521 |   3,708,173 |    264   |     264   |       1.2 | N/A       |     0.27 |
| pendulum | SAC    |    32768 | 215,605,780 |   4,952,519 |    264.2 |     264.2 |       0.1 | N/A       |     0.21 |
| pendulum | SAC    |    65536 | 313,257,681 |   5,844,931 |    264.4 |     264.4 |       0.2 | N/A       |     0.18 |

## License

MIT
