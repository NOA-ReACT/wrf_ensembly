import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import netCDF4
import numpy as np
import pandas as pd
import tomli_w


def parse_spex_date(
    utc_date: np.ndarray, fraction_of_day: np.ndarray
) -> pd.DatetimeIndex:
    """
    SPEX dates are provided in a UTC date as an integer and a float representing the fraction of day.
    This function converts them to pandas timestamps.
    """

    utcdate_str = np.array([str(int(date)) for date in utc_date])
    date = pd.to_datetime(utcdate_str, format="%Y%m%d")
    date = date + pd.to_timedelta(fraction_of_day, unit="D")

    return date


def process_spex_file(nc_path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Get the start/end time of a SPEX file
    """

    with netCDF4.Dataset(nc_path) as nc:  # type: ignore
        try:
            utcdate = nc["/geolocation_data/utc_date"][:].flatten()
            fracday = nc["/geolocation_data/fracday"][:].flatten()

            # Mask out fill values for utcdate (29990101)
            mask = utcdate != 29990101

            timestamps = parse_spex_date(utcdate[mask], fracday[mask])
        except pd.errors.OutOfBoundsDatetime as ex:
            print(f"File {nc_path} has invalid dates!" + str(ex))
            return None
        start, end = timestamps.min(), timestamps.max()

    return start, end


@click.command()
@click.argument("dir_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(writable=True, path_type=Path))
@click.option(
    "--glob", "-g", default="*.nc", help="Glob pattern filter files/directories"
)
def create_remotap_spexone_obsgroup(dir_path: Path, output_path: Path, glob: str):
    """
    Creates the observation group file for the given directory of AEOLUS L2B DBL files
    """

    # Ensure the output path is writable
    output_path.touch(exist_ok=True)

    # Sample file name:  PACE_SPEXONE.20240226T013118.L2.AER_OCEAN_REMOTAP.nc
    matcher = re.compile(r"PACE_SPEXONE.*AER_(?:LAND|OCEAN)_REMOTAP\.nc")

    potential_files = [
        p for p in dir_path.rglob(glob) if p.is_file() and matcher.match(p.name)
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_spex_file, potential_files))

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
        "kind": "RemoTAP-SPEXone",
        "converter": "/path/to/remotap_spexone_converter",
        "files": files,
    }
    with open(output_path, "wb") as f:
        tomli_w.dump(obsgroup, f)
    click.echo(f"Wrote info about {len(files)} files to {output_path}")


if __name__ == "__main__":
    create_remotap_spexone_obsgroup()
