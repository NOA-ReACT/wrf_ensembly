"""
Script for creating the observation group for EarthCARE ATL EBD files
"""

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cftime
import click
import netCDF4
import numpy as np
import pandas as pd
import tomli_w


def process_ec_atl_ebd_file(nc_path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Get the start/end time of an EarthCARE ATL EBD file
    """

    with netCDF4.Dataset(nc_path) as nc:  # type: ignore
        time_var = nc["/ScienceData/time"]
        time = cftime.num2pydate(time_var[:], time_var.units)
        time = pd.to_datetime(time)
    start, end = time.min(), time.max()

    return start, end


@click.command()
@click.argument("dir_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(writable=True, path_type=Path))
@click.option(
    "--glob", "-g", default="*.h5", help="Glob pattern filter files/directories"
)
def create_earthcare_atl_ebd_obsgroup(dir_path: Path, output_path: Path, glob: str):
    """
    Creates the observation group file for the given directory of EarthCARE ATL EBD files
    """

    # Ensure the output path is writable
    output_path.touch(exist_ok=True)

    # Sample file name: ECA_EXAE_ATL_EBD_2A_20250401T073529Z_20250402T014558Z_04782F.h5
    matcher = re.compile(r"ECA_EXAE_ATL_EBD_2A_.*.h5")

    potential_files = [
        p for p in dir_path.rglob(glob) if p.is_file() and matcher.match(p.name)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_ec_atl_ebd_file, potential_files))

    # Create the structure for the TOML file
    files = []
    for nc_path, result in zip(potential_files, results):
        if result is None:
            print(f"File {nc_path} has invalid dates!")
            continue
        start, end = result
        start = start.strftime("%Y%m%dT%H%M%S")
        end = end.strftime("%Y%m%dT%H%M%S")
        files.append(dict(path=str(nc_path.resolve()), start_date=start, end_date=end))

    if not files:
        click.echo("No files found!")
        return

    # Sort files by start date, easier to browse the file this way
    files = sorted(files, key=lambda f: f["start_date"])

    obsgroup = {
        "kind": "EC_ATL_EBD",
        "converter": "/path/to/ec_atl_ebd_converter",
        "files": files,
    }
    with open(output_path, "wb") as f:
        tomli_w.dump(obsgroup, f)
    click.echo(f"Wrote info about {len(files)} files to {output_path}")


if __name__ == "__main__":
    create_earthcare_atl_ebd_obsgroup()
