"""
Stand-in for the ICON executable in tests.

Reads `icon_master.namelist` and the atmosphere namelist from the current directory
and writes what ICON would: one file per output step of every `output_nml` stream, and
`finish.status`. The first guess stream gets a small but complete set of variables, so
that the analysis and cycling code can work on its files.

Set FAKE_ICON_FAIL to make it fail like ICON does (message, no finish.status).
"""

import datetime as dt
import os
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

NCELLS = 8
NEDGES = 12
NLEV = 4


def groups(text: str, name: str) -> list[dict[str, str]]:
    found = []
    for block in re.findall(rf"&{name}\n(.*?)\n/", text, flags=re.S):
        group = {}
        for line in block.splitlines():
            key, _, value = line.partition("=")
            group[key.strip()] = value.strip()
        found.append(group)
    return found


def text(value: str) -> str:
    return value.strip("'")


def date(value: str) -> dt.datetime:
    return dt.datetime.strptime(text(value), "%Y-%m-%dT%H:%M:%SZ")


def duration(value: str) -> dt.timedelta:
    match = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", text(value))
    assert match is not None, value
    days, hours, minutes = (int(x) if x else 0 for x in match.groups())
    return dt.timedelta(days=days, hours=hours, minutes=minutes)


def write_file(path: Path, time: dt.datetime, start: dt.datetime, first_guess: bool):
    seed = int(time.timestamp()) % 1000 + sum(map(ord, str(Path.cwd().name)))
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as ds:
        ds.uuidOfHGrid = "fake-grid"
        ds.createDimension("time", None)
        ds.createDimension("height", NLEV)
        ds.createDimension("height_2", NLEV + 1)
        cells = "ncells"
        if first_guess:
            # With variables on edges, ICON calls the cell dimension ncells_2
            ds.createDimension("ncells", NEDGES)
            cells = "ncells_2"
        ds.createDimension(cells, NCELLS)
        var = ds.createVariable("time", "f8", ("time",))
        var.units = f"minutes since {start:%Y-%m-%d %H:%M:%S}"
        var.calendar = "proleptic_gregorian"
        var[:] = [(time - start).total_seconds() / 60]

        names = ["temp", "pres", "u", "v", "qv"]
        if first_guess:
            names += ["rho", "theta_v", "dust"]
            ds.createVariable("vn", "f4", ("time", "height", "ncells"))[:] = 1.0
            ds.createVariable("w", "f4", ("time", "height_2", cells))[:] = 0.0
            z_ifc = np.linspace(20000, 0, NLEV + 1)[:, None] * np.ones(NCELLS)
            ds.createVariable("z_ifc", "f4", ("height_2", cells))[:] = z_ifc
        for name in names:
            var = ds.createVariable(name, "f4", ("time", "height", cells))
            var.CDI_grid_type = "unstructured"
            var[:] = rng.random((1, NLEV, NCELLS))
        ds.createVariable("pres_sfc", "f4", ("time", cells))[:] = 1e5


def main():
    master = Path("icon_master.namelist").read_text()
    time_control = groups(master, "master_time_control_nml")[0]
    start = date(time_control["experimentStartDate"])
    stop = date(time_control["experimentStopDate"])
    atmo_file = text(groups(master, "master_model_nml")[0]["model_namelist_filename"])
    atmo = Path(atmo_file).read_text()

    initicon = groups(atmo, "initicon_nml")[0]
    if initicon["init_mode"] == "1":
        for key in ("dwdfg_filename",) + (
            ("dwdana_filename",) if initicon["lread_ana"] == ".true." else ()
        ):
            if not Path(text(initicon[key])).exists():
                print(f"FINISH PE: 0 fake_icon: {text(initicon[key])} not found")
                sys.exit(1)
    elif not Path(text(initicon["ifs2icon_filename"])).exists():
        print("FINISH PE: 0 fake_icon: no IFS initial conditions")
        sys.exit(1)

    if os.environ.get("FAKE_ICON_FAIL"):
        print("FINISH PE: 3 mo_fake:fail: asked to fail")
        sys.exit(1)

    for stream in groups(atmo, "output_nml"):
        prefix = text(stream["output_filename"])
        time = date(stream["output_start"])
        end = min(date(stream["output_end"]), stop)
        step = duration(stream["output_interval"])
        while time <= end:
            path = Path(f"{prefix}_{time:%Y%m%dT%H%M%SZ}.nc")
            write_file(path, time, start, first_guess=Path(prefix).name == "fg")
            time += step

    Path("finish.status").write_text(" OK\n")


if __name__ == "__main__":
    main()
