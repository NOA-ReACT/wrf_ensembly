"""Tests for compression/quantization config validation."""

import pytest

from wrf_ensembly.config import PostprocessConfig
from wrf_ensembly.postprocess.compression import (
    CompressionConfigError,
    validate_compression_config,
)
from wrf_ensembly.statistics import QUANTIZATION_DISABLED


def cfg(**overrides) -> PostprocessConfig:
    c = PostprocessConfig()
    c.compression = "zlib"
    c.compression_level = 6
    c.significant_digits = 2
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def test_accepts_positive_override():
    validate_compression_config(cfg(significant_digits_overrides={"Z.*": 6}))


def test_accepts_minus_one_override():
    validate_compression_config(
        cfg(significant_digits_overrides={".*FLUX": QUANTIZATION_DISABLED})
    )


@pytest.mark.parametrize("digits", [0, -2, -10])
def test_rejects_other_non_positive_overrides(digits):
    with pytest.raises(CompressionConfigError, match="must be >= 1"):
        validate_compression_config(
            cfg(significant_digits_overrides={".*FLUX": digits})
        )


def test_rejects_invalid_regex():
    with pytest.raises(CompressionConfigError, match="Invalid regex"):
        validate_compression_config(cfg(significant_digits_overrides={"[unclosed": 4}))


def test_rejects_unknown_compression():
    with pytest.raises(CompressionConfigError, match="Unknown compression"):
        validate_compression_config(cfg(compression="lz4"))


def test_rejects_out_of_range_level():
    with pytest.raises(CompressionConfigError, match="compression_level"):
        validate_compression_config(cfg(compression_level=11))


def test_shuffle_with_non_zlib_warns(caplog):
    validate_compression_config(cfg(compression="zstd", shuffle=True))
    assert "no effect" in caplog.text


def test_shuffle_with_zlib_is_silent(caplog):
    validate_compression_config(cfg(compression="zlib", shuffle=True))
    assert "no effect" not in caplog.text
