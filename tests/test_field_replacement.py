import datetime as dt
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wrf_ensembly import experiment, field_replacement, segments, update_bc

from test_rebalance import NX, NY, NZ, make_state, write_restart
from test_segments_experiment import SEGMENTS, make_experiment

N_MEMBERS = 3
SHAPE = (1, NZ, NY, NX)


def make_target(path: Path, time, fields: dict[str, float]) -> None:
    """A wrfinput-like file with Times and constant (Time, z, y, x) fields"""

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        for dim, size in (("z", NZ), ("y", NY), ("x", NX)):
            ds.createDimension(dim, size)
        update_bc.write_wrf_time(
            ds.createVariable("Times", "S1", ("Time", "DateStrLen")), 0, time
        )
        for name, value in fields.items():
            ds.createVariable(name, "f4", ("Time", "z", "y", "x"))[:] = np.full(
                SHAPE, value
            )


def make_source(
    path: Path,
    fields: dict[str, np.ndarray],
    time=None,
    n_members: int = N_MEMBERS,
) -> None:
    """
    A field replacement file, with the dimensions (member, Time, z, y, x), from the right
    for arrays with fewer.
    """

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("member", n_members)
        ds.createDimension("Time", None)
        for dim, size in (("z", NZ), ("y", NY), ("x", NX)):
            ds.createDimension(dim, size)
        if time is not None:
            ds.createDimension("DateStrLen", 19)
            update_bc.write_wrf_time(
                ds.createVariable("Times", "S1", ("Time", "DateStrLen")), 0, time
            )
        for name, value in fields.items():
            dims = ("member", "Time", "z", "y", "x")[-value.ndim :]
            ds.createVariable(name, "f4", dims)[:] = value


@pytest.fixture
def time():
    return dt.datetime(2026, 3, 1, 18, tzinfo=dt.timezone.utc)


def test_replace_shared_and_per_member_fields(tmp_path: Path, time):
    make_target(tmp_path / "wrfinput", time, {"BACKG_OH": 0, "BACKG_H2O2": 0, "T": 5})
    per_member = np.stack([np.full(SHAPE, m + 10.0) for m in range(N_MEMBERS)])
    make_source(
        tmp_path / "src.nc",
        {
            "BACKG_OH": np.full(SHAPE, 1.5),
            "BACKG_H2O2": per_member,
            "T": np.zeros(SHAPE),
        },
        time=time,
    )

    with (
        netCDF4.Dataset(tmp_path / "wrfinput", "r+") as target,
        netCDF4.Dataset(tmp_path / "src.nc") as source,
    ):
        replaced = field_replacement.replace_fields(
            target, source, ["BACKG_OH", "BACKG_H2O2"], member_i=2, n_members=N_MEMBERS
        )

    assert replaced == ["BACKG_OH", "BACKG_H2O2"]
    with netCDF4.Dataset(tmp_path / "wrfinput") as ds:
        assert np.all(ds["BACKG_OH"][:] == 1.5)
        assert np.all(ds["BACKG_H2O2"][:] == 12)
        assert np.all(ds["T"][:] == 5)  # Not in the variables
        assert "BACKG_OH, BACKG_H2O2" in ds.replaced_fields


def test_replace_fills_both_time_levels(tmp_path: Path, time):
    make_target(tmp_path / "wrfrst", time, {"THM_1": 1, "THM_2": 2})
    make_source(tmp_path / "src.nc", {"THM": np.full(SHAPE, 5.0)})

    with (
        netCDF4.Dataset(tmp_path / "wrfrst", "r+") as target,
        netCDF4.Dataset(tmp_path / "src.nc") as source,
    ):
        field_replacement.replace_fields(target, source, ["THM"], 0, N_MEMBERS)

    with netCDF4.Dataset(tmp_path / "wrfrst") as ds:
        assert np.all(ds["THM_2"][:] == 5)
        assert np.all(ds["THM_1"][:] == 4)


@pytest.mark.parametrize(
    "fields, variables, match",
    [
        ({"BACKG_OH": np.zeros(SHAPE)}, ["MISSING"], "MISSING not in"),
        ({"OTHER": np.zeros(SHAPE)}, ["OTHER"], "OTHER not in"),
        ({"BACKG_OH": np.zeros((NZ, NY, NX))}, ["BACKG_OH"], "shape"),
        # A member dimension of the wrong size
        ({"BACKG_OH": np.zeros((N_MEMBERS + 1, *SHAPE))}, ["BACKG_OH"], "shape"),
        ({"BACKG_OH": np.full(SHAPE, np.nan)}, ["BACKG_OH"], "non-finite"),
    ],
)
def test_replace_rejects_bad_sources(tmp_path: Path, time, fields, variables, match):
    make_target(tmp_path / "wrfinput", time, {"BACKG_OH": 0})
    make_source(tmp_path / "src.nc", fields, n_members=N_MEMBERS + 1)

    with (
        netCDF4.Dataset(tmp_path / "wrfinput", "r+") as target,
        netCDF4.Dataset(tmp_path / "src.nc") as source,
    ):
        with pytest.raises(ValueError, match=match):
            field_replacement.replace_fields(target, source, variables, 0, N_MEMBERS)

    with netCDF4.Dataset(tmp_path / "wrfinput") as ds:
        assert np.all(ds["BACKG_OH"][:] == 0)


def test_replace_rejects_source_at_another_time(tmp_path: Path, time):
    make_target(tmp_path / "wrfinput", time, {"BACKG_OH": 0})
    make_source(
        tmp_path / "src.nc",
        {"BACKG_OH": np.zeros(SHAPE)},
        time=time + dt.timedelta(hours=6),
    )

    with (
        netCDF4.Dataset(tmp_path / "wrfinput", "r+") as target,
        netCDF4.Dataset(tmp_path / "src.nc") as source,
    ):
        with pytest.raises(ValueError, match="is at"):
            field_replacement.replace_fields(target, source, ["BACKG_OH"], 0, N_MEMBERS)
        # The check for other cycles' files skips it
        field_replacement.check_source(
            source, target, ["BACKG_OH"], N_MEMBERS, check_time=False
        )


# Segments


def test_find_stops_for_field_replacement():
    stops = segments.find_stops(10, {2}, set(), field_replacement_stops={2, 4})

    assert stops == {
        2: segments.STOP_OBSERVATIONS,
        4: segments.STOP_FIELD_REPLACEMENT,
        9: segments.STOP_LAST_CYCLE,
    }


FIELD_REPLACEMENT = """
[field_replacement]
enabled = true
variables = ["BACKG_OH"]
"""


def make_replacing_experiment(path: Path, extra: str = "") -> experiment.Experiment:
    exp = make_experiment(path, extra=SEGMENTS + FIELD_REPLACEMENT + extra)
    exp.paths.field_replacement.mkdir()
    return exp


def test_segment_stops_before_a_replacement_file(tmp_path: Path):
    exp = make_replacing_experiment(tmp_path / "exp")
    # Cycle 0 has no `cycle` before it, so it is no stop
    exp.paths.field_replacement_path(0).touch()
    exp.paths.field_replacement_path(2).touch()

    assert exp.field_replacement_stops() == {1}
    plan = exp.plan_segment(0)
    assert (plan.last, plan.stop_reason) == (1, segments.STOP_FIELD_REPLACEMENT)


def test_disabled_replacement_files_are_no_stops(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    exp.paths.field_replacement.mkdir()
    exp.paths.field_replacement_path(2).touch()

    assert exp.field_replacement_stops() == set()
    assert exp.plan_segment(0).last == 3


def test_new_stop_inside_for_replacement_files(tmp_path: Path):
    exp = make_replacing_experiment(tmp_path / "exp")
    plan = exp.plan_segment(0)  # 0-3
    exp.paths.field_replacement_path(2).touch()

    assert segments.new_stop_inside(
        plan, set(), None, field_replacement_stops=exp.field_replacement_stops()
    ) == (1, segments.STOP_FIELD_REPLACEMENT)


# Cycling


def test_cycle_member_restart_replaces_and_rebalances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp = make_replacing_experiment(tmp_path / "exp")
    monkeypatch.setattr(exp, "prepare_member_boundaries", lambda *_: None)
    exp.cfg.field_replacement.variables = ["BACKG_OH", "THM"]
    member_i = 1
    exp.paths.member_path(member_i).mkdir(parents=True)
    end = exp.cycles[0].end

    state, _ = make_state(np.random.default_rng(0))
    source = (
        exp.paths.scratch_restart_path(0, member_i)
        / f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
    )
    source.parent.mkdir(parents=True)
    write_restart(source, state, hypsometric_opt=2)
    with netCDF4.Dataset(source, "r+") as ds:
        ds.createDimension("DateStrLen", 19)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        update_bc.write_wrf_time(times, 0, end)
        ds.createVariable("BACKG_OH", "f8", ("Time", "z", "y", "x"))[:] = 0
        ds.createVariable("THM_1", "f8", ("Time", "z", "y", "x"))[:] = ds["THM_2"][:]
        p_before = ds["P"][:]

    shape = (1, NZ, NY, NX)
    with netCDF4.Dataset(exp.paths.field_replacement_path(1), "w") as ds:
        ds.createDimension("member", exp.cfg.assimilation.n_members)
        ds.createDimension("Time", 1)
        for dim, size in (("z", NZ), ("y", NY), ("x", NX)):
            ds.createDimension(dim, size)
        ds.createVariable("BACKG_OH", "f4", ("Time", "z", "y", "x"))[:] = np.ones(shape)
        thm = ds.createVariable("THM", "f8", ("member", "Time", "z", "y", "x"))
        thm[:] = np.stack(
            [state["THM"][None] + m for m in range(exp.cfg.assimilation.n_members)]
        )

    exp.cycle_member(member_i, use_forecast=True)

    initial_state = exp.member_initial_state(member_i, 1)
    with netCDF4.Dataset(initial_state) as ds:
        assert np.all(ds["BACKG_OH"][:] == 1)
        np.testing.assert_allclose(ds["THM_2"][0], state["THM"] + member_i)
        np.testing.assert_allclose(ds["THM_1"][0], state["THM"] + member_i)
        assert "BACKG_OH, THM" in ds.replaced_fields
        # Warmer, so the pressure changed
        assert not np.allclose(ds["P"][:], p_before)

    # The original in scratch is untouched, so cycling again starts over
    with netCDF4.Dataset(source) as ds:
        assert np.all(ds["BACKG_OH"][:] == 0)


def test_replace_fields_for_the_first_cycle(tmp_path: Path):
    exp = make_replacing_experiment(tmp_path / "exp")
    start = exp.cycles[0].start
    for member_i in range(2):
        exp.paths.member_path(member_i).mkdir(parents=True)
        make_target(exp.member_initial_state(member_i, 0), start, {"BACKG_OH": 0})
    make_source(
        exp.paths.field_replacement_path(0),
        {"BACKG_OH": np.full(SHAPE, 3.0)},
        time=start,
        n_members=exp.cfg.assimilation.n_members,
    )

    assert exp.replace_fields_current_cycle(1) == ["BACKG_OH"]

    with netCDF4.Dataset(exp.member_initial_state(1, 0)) as ds:
        assert np.all(ds["BACKG_OH"][:] == 3)
    with netCDF4.Dataset(exp.member_initial_state(0, 0)) as ds:
        assert np.all(ds["BACKG_OH"][:] == 0)
