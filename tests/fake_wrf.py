"""
A stand-in for wrf.exe in tests: reads namelist.input from the current directory and
writes what WRF would, empty wrfouts every history_interval and restart files (with
`Times`) every restart_interval, logging both to rsl.out.0000 like WRF does.

With FAKE_WRF_DIE_AFTER_HOURS set, it "dies" after that many simulated hours, leaving a
half-written restart file at that time if one is due.
"""

import datetime as dt
import os
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrf_ensembly import fortran_namelists  # noqa: E402


def namelist_time(tc: dict, prefix: str) -> dt.datetime:
    return dt.datetime(
        tc[f"{prefix}_year"],
        tc[f"{prefix}_month"],
        tc[f"{prefix}_day"],
        tc[f"{prefix}_hour"],
        tc.get(f"{prefix}_minute", 0),
        tc.get(f"{prefix}_second", 0),
    )


def out_path(template: str, t: dt.datetime) -> Path:
    return Path(
        template.replace("<domain>", "01").replace("<date>", f"{t:%Y-%m-%d_%H:%M:%S}")
    )


def write_restart(path: Path, t: dt.datetime, complete: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not complete:
        path.write_bytes(b"CDF\x01 half written")
        return
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(f"{t:%Y-%m-%d_%H:%M:%S}".encode(), dtype="S1")


def main():
    tc = fortran_namelists.read("namelist.input")["time_control"]
    start, end = namelist_time(tc, "start"), namelist_time(tc, "end")
    history = dt.timedelta(minutes=tc["history_interval"])
    restart_every = dt.timedelta(minutes=tc["restart_interval"])
    die_after = os.environ.get("FAKE_WRF_DIE_AFTER_HOURS")
    die_at = start + dt.timedelta(hours=float(die_after)) if die_after else None

    rsl = open("rsl.out.0000", "w")
    rsl.write(f"start {start:%Y-%m-%d_%H:%M:%S} restart {tc['restart']}\n")
    if tc["restart"]:
        initial = Path(f"wrfrst_d01_{start:%Y-%m-%d_%H:%M:%S}")
        if not initial.exists():
            rsl.write(f"missing {initial}\n")
            sys.exit(1)

    t = start
    while t < end:
        t += min(history, restart_every)
        if die_at is not None and t >= die_at:
            if (t - start) % restart_every == dt.timedelta(0):
                write_restart(out_path(tc["rst_outname"], t), t, complete=False)
            rsl.close()
            sys.exit(9)
        if (t - start) % history == dt.timedelta(0):
            out = out_path(tc["history_outname"], t)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.touch()
            rsl.write(f"Timing for Writing {out} for domain 1: 1.0 elapsed seconds\n")
        if (t - start) % restart_every == dt.timedelta(0):
            write_restart(out_path(tc["rst_outname"], t), t)
            rsl.write("Timing for Writing restart for domain        1: 1.0 elapsed seconds\n")
        rsl.flush()

    rsl.write("wrf: SUCCESS COMPLETE WRF\n")
    rsl.close()


if __name__ == "__main__":
    main()
