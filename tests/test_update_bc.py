import datetime as dt

import netCDF4
import numpy as np
import pytest

from wrf_ensembly import update_bc

NZ, NY, NX, WIDTH = 4, 7, 9, 2
START = dt.datetime(2026, 3, 1, 12, tzinfo=dt.timezone.utc)
INTERVAL = dt.timedelta(hours=6)


def make_state(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def field(*shape, scale=1.0, offset=0.0):
        return offset + scale * rng.standard_normal(shape)

    return {
        "U": field(NZ, NY, NX + 1, scale=10),
        "V": field(NZ, NY + 1, NX, scale=10),
        "W": field(NZ + 1, NY, NX, scale=0.1),
        "PH": field(NZ + 1, NY, NX, scale=100),
        "THM": field(NZ, NY, NX, scale=5),
        "MU": field(NY, NX, scale=500),
        "MUB": field(NY, NX, scale=2000, offset=90000),
        "C1H": np.linspace(1.0, 0.0, NZ),
        "C2H": np.linspace(0.0, 90000.0, NZ),
        "C1F": np.linspace(1.0, 0.0, NZ + 1),
        "C2F": np.linspace(0.0, 90000.0, NZ + 1),
        "MAPFAC_UY": field(NY, NX + 1, scale=0.01, offset=1),
        "MAPFAC_VX": field(NY + 1, NX, scale=0.01, offset=1),
        "MAPFAC_MY": field(NY, NX, scale=0.01, offset=1),
        "QVAPOR": np.abs(field(NZ, NY, NX, scale=0.01)),
        "QCLOUD": np.abs(field(NZ, NY, NX, scale=0.001)),
    }


def write_times(var, times):
    for i, t in enumerate(times):
        update_bc._write_wrf_time(var, i, t)


def make_wrfbdy(path, state, n_records=2):
    """A wrfbdy whose first record holds `state` and whose tendencies are random."""

    rng = np.random.default_rng(1)
    coupled = update_bc.couple(state)
    with netCDF4.Dataset(path, "w") as bdy:
        bdy.createDimension("Time", None)
        bdy.createDimension("DateStrLen", 19)
        bdy.createDimension("bdy_width", WIDTH)
        starts = [START + i * INTERVAL for i in range(n_records)]
        for name, times in [
            ("Times", starts),
            (update_bc.THIS_BDY_TIME, starts),
            (update_bc.NEXT_BDY_TIME, [t + INTERVAL for t in starts]),
        ]:
            write_times(bdy.createVariable(name, "S1", ("Time", "DateStrLen")), times)

        for name, field in coupled.items():
            for side, slab in update_bc._edges(field, WIDTH).items():
                dims = (
                    "Time",
                    "bdy_width",
                    *(f"{name}_{side}_{i}" for i in range(slab.ndim - 1)),
                )
                for dim, size in zip(dims[2:], slab.shape[1:]):
                    bdy.createDimension(dim, size)
                value = bdy.createVariable(f"{name}_B{side}", "f4", dims)
                tend = bdy.createVariable(f"{name}_BT{side}", "f4", dims)
                for i in range(n_records):
                    value[i] = slab
                    # At most halve or grow by half over the interval, so nothing changes sign
                    change = rng.uniform(-0.5, 0.5, slab.shape)
                    tend[i] = slab * change / INTERVAL.total_seconds()


def assert_close(actual, desired):
    """Equal up to float32 rounding, relative to the magnitude of the whole field"""
    scale = max(np.abs(desired).max(), np.abs(actual).max())
    np.testing.assert_allclose(actual, desired, rtol=0, atol=1e-5 * scale)


def read_record(path, itime):
    with netCDF4.Dataset(path) as bdy:
        bdy.set_auto_mask(False)
        return {
            name: bdy[name][itime].astype("f8")
            for name in bdy.variables
            if "_B" in name
        }


def test_edges_are_counted_inwards_from_the_boundary():
    field = np.arange(NY * NX).reshape(NY, NX)
    edges = update_bc._edges(field, WIDTH)

    np.testing.assert_array_equal(edges["XS"][0], field[:, 0])
    np.testing.assert_array_equal(edges["XS"][1], field[:, 1])
    np.testing.assert_array_equal(edges["XE"][0], field[:, -1])
    np.testing.assert_array_equal(edges["XE"][1], field[:, -2])
    np.testing.assert_array_equal(edges["YS"][1], field[1, :])
    np.testing.assert_array_equal(edges["YE"][0], field[-1, :])
    np.testing.assert_array_equal(edges["YE"][1], field[-2, :])


def test_coupling_uses_hybrid_coefficients():
    state = make_state(np.random.default_rng(0))
    coupled = update_bc.couple(state)

    mut = state["MU"] + state["MUB"]
    k, j, i = 2, 3, 4
    expected_t = state["THM"][k, j, i] * (state["C1H"][k] * mut[j, i] + state["C2H"][k])
    assert coupled["T"][k, j, i] == pytest.approx(expected_t)
    expected_ph = state["PH"][k, j, i] * (state["C1F"][k] * mut[j, i] + state["C2F"][k])
    assert coupled["PH"][k, j, i] == pytest.approx(expected_ph)

    # U at an interior point uses the average of the two neighbouring mass points, at the
    # western edge the edge mass point itself
    muu = 0.5 * (mut[j, i - 1] + mut[j, i])
    expected_u = state["U"][k, j, i] * (state["C1H"][k] * muu + state["C2H"][k])
    assert coupled["U"][k, j, i] == pytest.approx(expected_u / state["MAPFAC_UY"][j, i])
    expected_u0 = state["U"][k, j, 0] * (state["C1H"][k] * mut[j, 0] + state["C2H"][k])
    assert coupled["U"][k, j, 0] == pytest.approx(
        expected_u0 / state["MAPFAC_UY"][j, 0]
    )

    np.testing.assert_array_equal(coupled["MU"], state["MU"])


def test_unchanged_state_leaves_boundaries_unchanged(tmp_path):
    state = make_state(np.random.default_rng(0))
    path = tmp_path / "wrfbdy_d01"
    make_wrfbdy(path, state)
    before = read_record(path, 0)

    with netCDF4.Dataset(path, "r+") as bdy:
        update_bc.update_boundaries(bdy, state, START)

    after = read_record(path, 0)
    for name in before:
        assert_close(after[name], before[name])


def test_modified_state_keeps_the_value_at_the_end_of_the_interval(tmp_path):
    rng = np.random.default_rng(0)
    state = make_state(rng)
    path = tmp_path / "wrfbdy_d01"
    make_wrfbdy(path, state)
    before = read_record(path, 1), read_record(path, 0)

    # Analysis in the middle of the second record
    analysis_time = START + INTERVAL + dt.timedelta(hours=2)
    modified = dict(state)
    modified["U"] = state["U"] + 5
    modified["THM"] = state["THM"] - 1
    with netCDF4.Dataset(path, "r+") as bdy:
        update_bc.update_boundaries(bdy, modified, analysis_time)

    after = read_record(path, 1)
    old = before[0]
    coupled = update_bc.couple(modified)
    for name in ("U", "T", "MU", "QVAPOR"):
        for side, slab in update_bc._edges(coupled[name], WIDTH).items():
            value, tend = f"{name}_B{side}", f"{name}_BT{side}"
            assert_close(after[value], slab)

            end_old = old[value] + old[tend] * INTERVAL.total_seconds()
            end_new = after[value] + after[tend] * dt.timedelta(hours=4).total_seconds()
            assert_close(end_new, end_old)

    # The first record is untouched, and the record start time moves to the analysis
    for name, value in before[1].items():
        np.testing.assert_array_equal(read_record(path, 0)[name], value)
    with netCDF4.Dataset(path) as bdy:
        this_times = update_bc._parse_wrf_times(bdy[update_bc.THIS_BDY_TIME])
    assert this_times == [START, analysis_time]


def test_moisture_at_the_end_of_the_interval_is_not_negative(tmp_path):
    state = make_state(np.random.default_rng(0))
    path = tmp_path / "wrfbdy_d01"
    make_wrfbdy(path, state)

    # Make the tendency drive QVAPOR negative by the end of the interval
    with netCDF4.Dataset(path, "r+") as bdy:
        for side in update_bc.SIDES:
            value = bdy[f"QVAPOR_B{side}"][0]
            bdy[f"QVAPOR_BT{side}"][0] = -2 * value / INTERVAL.total_seconds()

    with netCDF4.Dataset(path, "r+") as bdy:
        update_bc.update_boundaries(bdy, state, START)

    after = read_record(path, 0)
    for side in update_bc.SIDES:
        value = after[f"QVAPOR_B{side}"]
        end = value + after[f"QVAPOR_BT{side}"] * INTERVAL.total_seconds()
        np.testing.assert_allclose(end, 0, atol=1e-5 * np.abs(value).max())


def test_analysis_time_outside_the_file_is_rejected(tmp_path):
    state = make_state(np.random.default_rng(0))
    path = tmp_path / "wrfbdy_d01"
    make_wrfbdy(path, state, n_records=1)

    with netCDF4.Dataset(path, "r+") as bdy:
        with pytest.raises(ValueError, match="before the first"):
            update_bc.update_boundaries(bdy, state, START - INTERVAL)
        with pytest.raises(ValueError, match="outside boundary record"):
            update_bc.update_boundaries(bdy, state, START + INTERVAL)
