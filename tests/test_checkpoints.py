import datetime as dt
from pathlib import Path

import netCDF4
import numpy as np

from wrf_ensembly import checkpoints

T0 = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)


def name(t: dt.datetime) -> str:
    return f"wrfrst_d01_{t:%Y-%m-%d_%H:%M:%S}"


def write_restart(path: Path, t: dt.datetime) -> None:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(f"{t:%Y-%m-%d_%H:%M:%S}".encode(), dtype="S1")


def test_list_checkpoints_sorted_by_time(tmp_path: Path):
    for h in (12, 6, 18):
        (tmp_path / name(T0 + dt.timedelta(hours=h))).touch()
    (tmp_path / "wrfrst_d01_garbage").touch()
    (tmp_path / "wrfout_d01_2021-01-01_06:00:00").touch()

    found = checkpoints.list_checkpoints(tmp_path)

    assert [t.hour for t, _ in found] == [6, 12, 18]


def test_prune_keeps_newest(tmp_path: Path):
    for h in range(0, 30, 6):
        (tmp_path / name(T0 + dt.timedelta(hours=h))).touch()

    deleted = checkpoints.prune(tmp_path, 2)

    assert len(deleted) == 3
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        name(T0 + dt.timedelta(hours=18)),
        name(T0 + dt.timedelta(hours=24)),
    ]
    assert checkpoints.prune(tmp_path, 2) == []


def test_is_complete(tmp_path: Path):
    t = T0 + dt.timedelta(hours=6)
    good = tmp_path / name(t)
    write_restart(good, t)
    assert checkpoints.is_complete(good, t)
    assert not checkpoints.is_complete(good, T0)

    truncated = tmp_path / name(T0)
    truncated.write_bytes(good.read_bytes()[:100])
    assert not checkpoints.is_complete(truncated, T0)

    assert not checkpoints.is_complete(tmp_path / "missing", T0)


def test_written_by_run():
    rsl = (
        "Timing for main ...\n"
        "Timing for Writing restart for domain        1:   15.06 elapsed seconds\n"
        "Timing for Writing wrfout ...\n"
        "Timing for Writing restart for domain        1:   14.20 elapsed seconds\n"
    )
    start = T0 + dt.timedelta(hours=6)

    written = checkpoints.written_by_run(rsl, start, dt.timedelta(hours=12))

    assert written == {T0 + dt.timedelta(hours=18), T0 + dt.timedelta(hours=30)}
    assert checkpoints.written_by_run("", start, dt.timedelta(hours=12)) == set()


def test_confirm_adds_to_confirmed_times(tmp_path: Path):
    assert checkpoints.confirmed_times(tmp_path) == set()

    checkpoints.confirm(tmp_path, {T0})
    checkpoints.confirm(tmp_path, {T0 + dt.timedelta(hours=6)})

    assert checkpoints.confirmed_times(tmp_path) == {T0, T0 + dt.timedelta(hours=6)}


def test_unreadable_confirmations_confirm_nothing(tmp_path: Path):
    (tmp_path / checkpoints.CONFIRMED_FILE).write_text("{")

    assert checkpoints.confirmed_times(tmp_path) == set()


def test_watcher_confirms_and_prunes(tmp_path: Path):
    rsl = tmp_path / "rsl.out.0000"
    restarts = tmp_path / "restart"
    restarts.mkdir()
    interval = dt.timedelta(hours=6)

    with checkpoints.CheckpointWatcher(restarts, rsl, T0, interval, keep=2, poll_s=0.01):
        for k in range(1, 5):
            (restarts / name(T0 + k * interval)).touch()
            with open(rsl, "a") as f:
                f.write("Timing for Writing restart for domain 1: 1.0 elapsed seconds\n")
        # The fifth is still being written
        (restarts / name(T0 + 5 * interval)).touch()

    assert [t for t, _ in checkpoints.list_checkpoints(restarts)] == [
        T0 + 4 * interval,
        T0 + 5 * interval,
    ]
    assert checkpoints.confirmed_times(restarts) == {T0 + k * interval for k in range(1, 5)}


def test_watcher_without_pruning(tmp_path: Path):
    restarts = tmp_path / "restart"
    restarts.mkdir()
    for k in range(4):
        (restarts / name(T0 + dt.timedelta(hours=k))).touch()

    with checkpoints.CheckpointWatcher(
        restarts, tmp_path / "rsl.out.0000", T0, dt.timedelta(hours=1), keep=None
    ):
        pass

    assert len(checkpoints.list_checkpoints(restarts)) == 4
