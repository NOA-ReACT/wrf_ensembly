"""Tests for the significant_digits resolution used when creating output files."""

import netCDF4
import numpy as np
import pytest

from wrf_ensembly.statistics import (
    QUANTIZATION_DISABLED,
    NetCDFFile,
    NetCDFVariable,
    _get_variable_significant_digits,
    create_file,
)


def resolve(name: str, default=3, overrides=None):
    return _get_variable_significant_digits(name, default, overrides)


def test_default_used_when_no_override_matches():
    assert resolve("DUST_1", 3, {"Z.*": 6}) == 3
    assert resolve("DUST_1", 3, None) == 3
    assert resolve("DUST_1", 3, {}) == 3


def test_override_replaces_default():
    assert resolve("ZS", 2, {"Z.*": 6}) == 6


def test_first_matching_pattern_wins():
    overrides = {"air_.*": 4, "air_density": 6}
    assert resolve("air_density", 2, overrides) == 4

    overrides = {"air_density": 6, "air_.*": 4}
    assert resolve("air_density", 2, overrides) == 6


def test_minus_one_disables_quantization_for_that_variable():
    overrides = {".*FLUX": QUANTIZATION_DISABLED}
    assert resolve("DRY_DEP_FLUX", 2, overrides) is None
    # other variables keep the default
    assert resolve("DUST_1", 2, overrides) == 2


def test_disabled_default_beats_any_override():
    """A global `significant_digits = 0` turns quantization off everywhere."""

    assert resolve("ZS", None, {"Z.*": 6}) is None
    assert resolve("DRY_DEP_FLUX", None, {".*FLUX": QUANTIZATION_DISABLED}) is None


@pytest.fixture
def template():
    return NetCDFFile(
        dimensions={"t": 2, "y": 3, "x": 4},
        variables={
            name: NetCDFVariable(
                name=name,
                dimensions=("t", "y", "x"),
                attributes={},
                dtype=np.dtype("float32"),
            )
            for name in ("DUST_1", "DRY_DEP_FLUX")
        },
        global_attributes={},
    )


def test_create_file_omits_quantization_for_disabled_variables(tmp_path, template):
    path = tmp_path / "out.nc"
    ds = create_file(
        path,
        template,
        significant_digits=2,
        significant_digits_overrides={".*FLUX": QUANTIZATION_DISABLED},
    )
    ds.close()

    with netCDF4.Dataset(path) as ds:
        attr = "_QuantizeGranularBitRoundNumberOfSignificantDigits"
        assert ds.variables["DUST_1"].getncattr(attr) == 2
        assert attr not in ds.variables["DRY_DEP_FLUX"].ncattrs()


def test_create_file_roundtrips_disabled_variable_bit_exactly(tmp_path, template):
    rng = np.random.default_rng(0)
    data = rng.uniform(1e5, 1e6, size=(2, 3, 4)).astype(np.float32)

    path = tmp_path / "out.nc"
    ds = create_file(
        path,
        template,
        significant_digits=2,
        significant_digits_overrides={".*FLUX": QUANTIZATION_DISABLED},
    )
    ds.variables["DRY_DEP_FLUX"][:] = data
    ds.variables["DUST_1"][:] = data
    ds.close()

    with netCDF4.Dataset(path) as ds:
        assert np.array_equal(ds.variables["DRY_DEP_FLUX"][:].data, data)
        # the quantized variable is deliberately lossy
        assert not np.array_equal(ds.variables["DUST_1"][:].data, data)
