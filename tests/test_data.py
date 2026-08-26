# tests/test_data.py

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from scripts.data import (
    generate_market_data,
    load_config,
)


def test_generate_market_data_shape():
    config = load_config()

    tensor, symbols = generate_market_data(config)

    assert isinstance(symbols, list)

    n = config["num_candles"]
    s = len(config["symbols"])
    c = len(config["channels"])

    expected_shape = (n, s, c)

    assert tensor.shape == expected_shape
    assert symbols == config["symbols"]


def test_generate_market_data_values():
    config = load_config()

    tensor, symbols = generate_market_data(config)

    data = np.asarray(tensor)

    channels = {
        name: i
        for i, name in enumerate(
            config["channels"]
        )
    }

    closes = data[:, :, channels["close"]]
    assert np.all(closes > 0)

    volumes = data[:, :, channels["volume"]]
    assert np.all(volumes > 0)

    returns = data[:, :, channels["returns"]]
    assert not np.any(np.isnan(returns))
    assert not np.any(np.isinf(returns))

    volatility = data[:, :, channels["volatility"]]
    assert np.all(volatility >= 0)

    funding = data[:, :, channels["funding"]]
    assert not np.any(np.isnan(funding))
    assert not np.any(np.isinf(funding))

    alpha = data[:, :, channels["alpha"]]
    assert not np.any(np.isnan(alpha))
    assert not np.any(np.isinf(alpha))


def test_generate_market_data_seed():
    config = load_config()
    config["seed"] = 123

    tensor1, symbols1 = generate_market_data(config)

    config["seed"] = 123

    tensor2, symbols2 = generate_market_data(config)

    assert np.array_equal(
        np.asarray(tensor1),
        np.asarray(tensor2),
    )

    assert symbols1 == symbols2


def test_generate_market_data_empty_symbols():
    config = load_config()
    config["symbols"] = []

    with pytest.raises(
        ValueError,
        match="at least one symbol",
    ):
        generate_market_data(config)


def test_generate_market_data_single_symbol():
    config = load_config()
    config["symbols"] = ["BTC"]

    tensor, symbols = generate_market_data(config)

    assert tensor.shape[1] == 1
    assert symbols == ["BTC"]