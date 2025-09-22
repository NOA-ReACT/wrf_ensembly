"""Some utility commands for performing basic operations on observation files, such as dump, info, etc."""

import json
from pathlib import Path

import click
import pandas as pd
from rich.progress import track
from rich.table import Table

from wrf_ensembly import external
from wrf_ensembly.console import console
from wrf_ensembly.observations import dart as observations_dart
from wrf_ensembly.observations import io as obs_io


@click.command()
@click.argument("file_list", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def join_files(file_list: list[Path], output_path: Path):
    """Given a list of WRF-ensembly observation files, join them into a single file."""

    dfs = []
    for file_path in track(file_list, description="Reading files..."):
        df = obs_io.read_obs(file_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    obs_io.write_obs(combined_df, output_path)
    print(
        f"Successfully joined {len(file_list)} files into {output_path}, total observations: {len(combined_df)}"
    )


@click.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--as-json", is_flag=True, help="Output in JSON format", default=False)
def dump_info(file_path: Path, as_json: bool):
    """
    Print basic information about a WRF-Ensembly observation file.
    - Number of observations
    - Instruments present and their counts
    - z_types present and their counts
    - Quantities present and their counts
    - Time range of observations
    - Geographical range (min/max latitude and longitude)
    """

    df = obs_io.read_obs(file_path)
    info = {
        "file_path": str(file_path),
        "num_observations": len(df),
        "instruments": df["instrument"].value_counts().to_dict(),
        "z_types": df["z_type"].value_counts().to_dict(),
        "quantities": df["quantity"].value_counts().to_dict(),
        "time_range": {
            "start": df["time"].min().isoformat(),
            "end": df["time"].max().isoformat(),
        },
        "geographical_range": {
            "latitude": {
                "min": df["latitude"].min(),
                "max": df["latitude"].max(),
            },
            "longitude": {
                "min": df["longitude"].min(),
                "max": df["longitude"].max(),
            },
        },
    }

    if as_json:
        print(json.dumps(info, indent=2))
        return 0

    # Pretty print using rich
    console.print(f"[bold]Observation File Info: {file_path}[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")
    table.add_row("Number of Observations", str(info["num_observations"]))
    table.add_row(
        "Instruments", ", ".join(f"{k} ({v})" for k, v in info["instruments"].items())
    )
    table.add_row(
        "Z Types", ", ".join(f"{k} ({v})" for k, v in info["z_types"].items())
    )
    table.add_row(
        "Quantities", ", ".join(f"{k} ({v})" for k, v in info["quantities"].items())
    )
    table.add_row(
        "Time Range", f"{info['time_range']['start']} to {info['time_range']['end']}"
    )
    table.add_row(
        "Latitude Range",
        f"{info['geographical_range']['latitude']['min']} to {info['geographical_range']['latitude']['max']}",
    )
    table.add_row(
        "Longitude Range",
        f"{info['geographical_range']['longitude']['min']} to {info['geographical_range']['longitude']['max']}",
    )
    console.print(table)


@click.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--min-lat", type=float, required=False, help="Minimum latitude")
@click.option("--max-lat", type=float, required=False, help="Maximum latitude")
@click.option("--min-lon", type=float, required=False, help="Minimum longitude")
@click.option("--max-lon", type=float, required=False, help="Maximum longitude")
@click.option(
    "--min-time", type=str, required=False, help="Minimum timestamp (ISO format)"
)
@click.option(
    "--max-time", type=str, required=False, help="Maximum timestamp (ISO format)"
)
def filter_obs(
    file_path: Path,
    output_path: Path,
    min_lat: float | None,
    max_lat: float | None,
    min_lon: float | None,
    max_lon: float | None,
    min_time: str | None,
    max_time: str | None,
):
    """Filters an observation file based on spatial and temporal criteria."""

    df = obs_io.read_obs(file_path)
    original_len = len(df)

    if min_lat is not None:
        if min_lat < -90 or min_lat > 90:
            raise ValueError("min_lat must be between -90 and 90")
        df = df[df["latitude"] >= min_lat]
    if max_lat is not None:
        if max_lat < -90 or max_lat > 90:
            raise ValueError("max_lat must be between -90 and 90")
        df = df[df["latitude"] <= max_lat]
    if min_lon is not None:
        if min_lon < -180 or min_lon > 180:
            raise ValueError("min_lon must be between -180 and 180")
        df = df[df["longitude"] >= min_lon]
    if max_lon is not None:
        if max_lon < -180 or max_lon > 180:
            raise ValueError("max_lon must be between -180 and 180")
        df = df[df["longitude"] <= max_lon]
    if min_time is not None:
        try:
            min_time_parsed = pd.to_datetime(min_time)
        except Exception as e:
            raise ValueError(f"Invalid min_time format: {e}")
        df = df[df["time"] >= min_time_parsed]
    if max_time is not None:
        try:
            max_time_parsed = pd.to_datetime(max_time)
        except Exception as e:
            raise ValueError(f"Invalid max_time format: {e}")
        df = df[df["time"] <= max_time_parsed]

    print(
        f"Filtered observations from {original_len} to {len(df)} based on spatial criteria."
    )
    obs_io.write_obs(df, output_path)


@click.command()
@click.option(
    "--dart-path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.argument("obs_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
def to_obs_seq(obs_file: Path, output_file: Path, dart_path: Path):
    """
    Convert a WRF-Ensembly observation file to DART obs_seq format.

    You must have built the `wrf_ensembly` observation converter in DART for this to work, check the `DART/observations/obs_converters/wrf_ensembly` directory.

    Args:
        obs_file: Path to the input WRF-Ensembly observation file.
        output_file: Path where the output DART obs_seq file will be saved.
        dart_path: Path to the root of the DART installation.

    Example usage:
        wrf-ensembly-obs convert-to-obs-seq --dart-path /path/to/DART /path/to/input.parquet /path/to/output.obs_seq
    """

    process = observations_dart.convert_to_dart_obs_seq(
        dart_path=dart_path,
        observations=obs_io.read_obs(obs_file),
        output_location=output_file,
    )
    result = external.run(process)
    print("Converter output:")
    print(result.output)

    if result.returncode != 0:
        print("Error converting to DART obs_seq!")
    else:
        print(f"Successfully converted to DART obs_seq: {output_file}")
