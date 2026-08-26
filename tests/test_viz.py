# tests/test_viz.py

import os
import sys

import numpy as np
import pytest

from PIL import Image

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
from scripts.viz import plot_market_data


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def tensor(config):
    tensor, _ = generate_market_data(config)
    return np.asarray(tensor)


@pytest.fixture
def symbols(config):
    return config["symbols"]


def test_plot_market_data_default_symbol(
    tmp_path,
    tensor,
    symbols,
):
    theme = "synthwave"
    output_dir = tmp_path / "market"

    plot_market_data(
        tensor,
        symbols,
        theme,
        str(output_dir),
    )

    assert (
        output_dir / f"{theme}.png"
    ).exists()


def test_plot_market_data_explicit_symbol(
    tmp_path,
    tensor,
    symbols,
):
    theme = "ghibli"
    output_dir = tmp_path / "market"

    plot_market_data(
        tensor,
        symbols,
        theme,
        str(output_dir),
        symbol=symbols[0],
    )

    assert (
        output_dir / f"{theme}.png"
    ).exists()


def test_plot_market_data_unknown_symbol(
    tmp_path,
    tensor,
    symbols,
):
    theme = "fiesta"
    output_dir = tmp_path / "market"

    with pytest.raises(
        ValueError,
        match="Unknown symbol",
    ):
        plot_market_data(
            tensor,
            symbols,
            theme,
            str(output_dir),
            symbol="UNKNOWN",
        )


def test_plot_market_data_all_themes(
    tmp_path,
    tensor,
    symbols,
):
    themes = [
        "synthwave",
        "ghibli",
        "fiesta",
        "tropical",
    ]

    output_dir = tmp_path / "market"

    for theme in themes:
        plot_market_data(
            tensor,
            symbols,
            theme,
            str(output_dir),
        )

        assert (
            output_dir / f"{theme}.png"
        ).exists()


def test_plot_market_data_channel_mapping(
    tmp_path,
    tensor,
    symbols,
):
    theme = "tropical"
    output_dir = tmp_path / "market"

    plot_market_data(
        tensor,
        symbols,
        theme,
        str(output_dir),
    )

    assert (
        output_dir / f"{theme}.png"
    ).exists()

def test_plot_market_data_creates_valid_png(
    tmp_path,
    tensor,
    symbols,
):
    theme = "synthwave"
    output_dir = tmp_path / "market"

    plot_market_data(
        tensor,
        symbols,
        theme,
        str(output_dir),
    )

    output_file = output_dir / f"{theme}.png"

    assert output_file.exists()
    assert output_file.stat().st_size > 0

    with Image.open(output_file) as image:
        assert image.format == "PNG"
        assert image.width == 1600 * 2
        assert image.height == 900 * 2