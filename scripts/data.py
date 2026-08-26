# scripts/data.py

import yaml
import mlx.core as mx
import numpy as np


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_market_data(config):
    np.random.seed(config.get("seed", 42))

    n = config["num_candles"]
    symbols = config["symbols"]
    s = len(symbols)

    if s == 0:
        raise ValueError("at least one symbol")

    channels = config.get(
        "channels",
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "volatility",
            "funding",
            "alpha",
        ],
    )

    if len(channels) != 9:
        raise ValueError(
            f"Expected exactly 9 channels, got {len(channels)}"
        )

    channel_names = {
        name: i for i, name in enumerate(channels)
    }

    dt = config.get("dt", 1 / 1440)

    mu = config.get("gbm_drift", 0.05)
    sigma = config.get("gbm_vol", 0.2)
    initial_price = config.get("initial_price", 100.0)

    # GBM returns
    gbm_returns = (
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * np.random.randn(n, s)
    )

    # GBM price path
    prices = initial_price * np.exp(
        np.cumsum(gbm_returns, axis=0)
    )

    # OHLC noise
    noise = np.random.normal(
        0.001,
        0.005,
        (n, s, 4),
    )

    opens = prices * (1 + noise[:, :, 0])

    highs = (
        np.maximum(prices, opens)
        * (1 + np.abs(noise[:, :, 1]))
    )

    lows = (
        np.minimum(prices, opens)
        * (1 - np.abs(noise[:, :, 2]))
    )

    closes = prices * (1 + noise[:, :, 3])

    # Positive volume
    volumes = np.random.exponential(
        1000,
        (n, s),
    )

    # Convert base market data to MLX
    opens_mx = mx.array(opens)
    highs_mx = mx.array(highs)
    lows_mx = mx.array(lows)
    closes_mx = mx.array(closes)
    volumes_mx = mx.array(volumes)

    # Returns
    returns_ta = mx.concatenate(
        [
            mx.zeros((1, s)),
            (
                closes_mx[1:]
                - closes_mx[:-1]
            )
            / closes_mx[:-1],
        ],
        axis=0,
    )

    # Simple absolute price-change volatility proxy
    volatility = mx.abs(
        closes_mx
        - mx.roll(closes_mx, 1, axis=0)
    )

    # Funding: mean-reverting OU-style process
    funding = np.zeros((n, s))

    funding_decay = config.get(
        "funding_decay",
        0.1,
    )
    funding_noise = config.get(
        "funding_noise",
        0.01,
    )

    for t in range(1, n):
        funding[t] = (
            funding[t - 1]
            - funding_decay * funding[t - 1]
            + funding_noise * np.random.randn(s)
        )

    funding_mx = mx.array(funding)

    # AR alpha process
    phi = config.get("ar_phi", 0.0)
    alpha_scale = config.get(
        "alpha_noise",
        0.001,
    )

    alpha = np.zeros((n, s))
    alpha_noise = np.random.randn(n, s)

    for t in range(1, n):
        alpha[t] = (
            phi * alpha[t - 1]
            + alpha_noise[t] * alpha_scale
        )

    alpha_mx = mx.array(alpha)

    # Build (candles, symbols, channels)
    channel_data = {
        "open": opens_mx,
        "high": highs_mx,
        "low": lows_mx,
        "close": closes_mx,
        "volume": volumes_mx,
        "returns": returns_ta,
        "volatility": volatility,
        "funding": funding_mx,
        "alpha": alpha_mx,
    }

    try:
        ordered_channels = [
            channel_data[name]
            for name in channels
        ]
    except KeyError as exc:
        raise ValueError(
            f"Unknown channel in config: {exc.args[0]}"
        ) from exc

    tensor = mx.stack(
        ordered_channels,
        axis=-1,
    )

    expected_shape = (n, s, len(channels))

    if tensor.shape != expected_shape:
        raise RuntimeError(
            f"Generated tensor has shape {tensor.shape}, "
            f"expected {expected_shape}"
        )

    return tensor, symbols