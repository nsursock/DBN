# scripts/viz.py

import os

import numpy as np
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_theme(name):
    with open(f"configs/themes/{name}.yaml") as f:
        return yaml.safe_load(f)


def plot_market_data(
    tensor,
    symbols,
    theme_name,
    output_dir="outputs/market",
    symbol=None,
):
    config = load_config()
    channels = config["channels"]

    channel_indices = {
        name: i
        for i, name in enumerate(channels)
    }

    required_channels = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding",
    ]

    missing = [
        name
        for name in required_channels
        if name not in channel_indices
    ]

    if missing:
        raise ValueError(
            f"Missing required channels: {missing}"
        )

    t = load_theme(theme_name)
    d = np.asarray(tensor)

    if d.ndim != 3:
        raise ValueError(
            "Expected tensor with shape "
            "(candles, symbols, channels), "
            f"got {d.shape}"
        )

    if len(symbols) == 0:
        raise ValueError("No symbols provided")

    if d.shape[1] != len(symbols):
        raise ValueError(
            f"Tensor has {d.shape[1]} symbols but "
            f"received {len(symbols)} symbol names"
        )

    if d.shape[2] != len(channels):
        raise ValueError(
            f"Tensor has {d.shape[2]} channels but "
            f"config defines {len(channels)} channels"
        )

    if symbol is None:
        symbol = symbols[0]

    if symbol not in symbols:
        raise ValueError(f"Unknown symbol: {symbol}")

    symbol_idx = symbols.index(symbol)

    x = np.arange(d.shape[0])

    open_idx = channel_indices["open"]
    high_idx = channel_indices["high"]
    low_idx = channel_indices["low"]
    close_idx = channel_indices["close"]
    volume_idx = channel_indices["volume"]
    funding_idx = channel_indices["funding"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.20, 0.18],
        vertical_spacing=0.025,
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=d[:, symbol_idx, open_idx],
            high=d[:, symbol_idx, high_idx],
            low=d[:, symbol_idx, low_idx],
            close=d[:, symbol_idx, close_idx],
            name=f"{symbol} OHLC",
            increasing_line_color=t["primary"],
            increasing_fillcolor=t["primary"],
            decreasing_line_color=t["secondary"],
            decreasing_fillcolor=t["secondary"],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=x,
            y=d[:, symbol_idx, volume_idx],
            name="Volume",
            marker_color=t["secondary"],
            opacity=0.45,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=d[:, symbol_idx, funding_idx],
            name="Funding",
            mode="lines",
            line=dict(
                color=t["primary"],
                width=2,
            ),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{symbol} · Synthetic Market",
            x=0.01,
            font=dict(
                size=22,
                family="JetBrains Mono",
                color=t["neutral"],
            ),
        ),
        height=900,
        paper_bgcolor=t["base-200"],
        plot_bgcolor=t["base-100"],
        font=dict(
            family="JetBrains Mono",
            size=14,
            color=t["neutral"],
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(
            l=70,
            r=30,
            t=90,
            b=50,
        ),
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=t["base-300"],
        zeroline=False,
        tickfont=dict(
            family="JetBrains Mono",
            size=12,
        ),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["base-300"],
        zeroline=False,
        tickfont=dict(
            family="JetBrains Mono",
            size=12,
        ),
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{theme_name}.png",
    )

    fig.write_image(
        output_path,
        width=1600,
        height=900,
        scale=2,
    )