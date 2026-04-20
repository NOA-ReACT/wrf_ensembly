"""Converter for AEOLUS L2B files to WRF-Ensembly Observation format."""

from pathlib import Path

import click
import numpy as np
import pandas as pd

try:
    import coda
    _DEPS_ERR: Exception | None = None
except Exception as _e:
    coda = None  # type: ignore[assignment]
    _DEPS_ERR = _e


def _require_deps() -> None:
    if _DEPS_ERR is not None:
        raise click.ClickException(
            "The aeolus-l2b converter requires 'coda' (stcorp/coda).\n"
            f"Loading it failed with {type(_DEPS_ERR).__name__}: {_DEPS_ERR}\n"
            "Install from: https://github.com/stcorp/coda"
        )

from wrf_ensembly.observations import io as obs_io

# Variable mappings for Mie retrievals
mie_variables = {
    "wind_result_id": ("mie_geolocation", -1, "wind_result_id"),
    "z": ("mie_geolocation", -1, "windresult_geolocation", "altitude_vcog"),
    "alt_bottom": ("mie_geolocation", -1, "windresult_geolocation", "altitude_bottom"),
    "alt_top": ("mie_geolocation", -1, "windresult_geolocation", "altitude_top"),
    "azimuth_a": ("mie_geolocation", -1, "windresult_geolocation", "los_azimuth"),
    "y": ("mie_geolocation", -1, "windresult_geolocation", "latitude_cog"),
    "x": ("mie_geolocation", -1, "windresult_geolocation", "longitude_cog"),
    "t": ("mie_geolocation", -1, "windresult_geolocation", "datetime_cog"),
    "t_start": ("mie_geolocation", -1, "windresult_geolocation", "datetime_start"),
    "t_stop": ("mie_geolocation", -1, "windresult_geolocation", "datetime_stop"),
    "obs": ("mie_hloswind", -1, "windresult", "mie_wind_velocity"),
    "obs_err": ("mie_wind_prod_conf_data", -1, "mie_wind_qc", "hlos_error_estimate"),
    "qc_pass": ("mie_hloswind", -1, "windresult", "validity_flag"),
    "observation_type": ("mie_hloswind", -1, "windresult", "observation_type"),
}

# Variable mappings for Rayleigh retrievals
rayleigh_variables = {
    "wind_result_id": ("rayleigh_geolocation", -1, "wind_result_id"),
    "z": ("rayleigh_geolocation", -1, "windresult_geolocation", "altitude_vcog"),
    "alt_bottom": (
        "rayleigh_geolocation",
        -1,
        "windresult_geolocation",
        "altitude_bottom",
    ),
    "alt_top": (
        "rayleigh_geolocation",
        -1,
        "windresult_geolocation",
        "altitude_top",
    ),
    "azimuth_a": (
        "rayleigh_geolocation",
        -1,
        "windresult_geolocation",
        "los_azimuth",
    ),
    "y": ("rayleigh_geolocation", -1, "windresult_geolocation", "latitude_cog"),
    "x": (
        "rayleigh_geolocation",
        -1,
        "windresult_geolocation",
        "longitude_cog",
    ),
    "t": ("rayleigh_geolocation", -1, "windresult_geolocation", "datetime_cog"),
    "t_start": (
        "rayleigh_geolocation",
        -1,
        "windresult_geolocation",
        "datetime_start",
    ),
    "t_stop": (
        "rayleigh_geolocation",
        -1,
        "windresult_geolocation",
        "datetime_stop",
    ),
    "obs": ("rayleigh_hloswind", -1, "windresult", "rayleigh_wind_velocity"),
    "obs_err": (
        "rayleigh_wind_prod_conf_data",
        -1,
        "rayleigh_wind_qc",
        "hlos_error_estimate",
    ),
    "qc_pass": ("rayleigh_hloswind", -1, "windresult", "validity_flag"),
    "observation_type": ("rayleigh_hloswind", -1, "windresult", "observation_type"),
}

_AEOLUS_EPOCH = pd.Timestamp("2000-01-01", tz="UTC")


def _decode_aeolus_time(seconds: pd.Series) -> pd.Series:
    """Convert Aeolus seconds-since-2000 to timezone-aware UTC timestamps."""
    return _AEOLUS_EPOCH + pd.to_timedelta(seconds, unit="seconds")


def convert_aeolus_l2b(
    path: Path,
    include_mie: bool = True,
    include_rayleigh: bool = True,
) -> pd.DataFrame | None:
    """Convert an AEOLUS L2B file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AEOLUS L2B file.
        include_mie: Whether to include Mie wind retrievals.
        include_rayleigh: Whether to include Rayleigh wind retrievals.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid data.
    """
    _require_deps()
    cf = coda.open(str(path))
    if cf.product_type != "ALD_U_N_2B":
        cf.close()
        raise ValueError("Input file is not an AEOLUS DBL L2B file.")

    retrieval_types = {}
    if include_mie:
        retrieval_types["mie"] = mie_variables
    if include_rayleigh:
        retrieval_types["rayleigh"] = rayleigh_variables

    # Read all variables we need from the DBL file, per retrieval type
    datasets = []
    for name, variables in retrieval_types.items():
        data = {}
        for k, v in variables.items():
            try:
                data[k] = coda.fetch(cf, *v)
            except coda.CodacError:
                data[k] = np.array([])

        # Skip if no data available
        if len(data.get("obs", [])) == 0:
            continue

        shape = next(iter(data.values())).shape
        data["retrieval_type"] = np.full(shape, name)

        datasets.append(pd.DataFrame(data))

    cf.close()

    if not datasets:
        return None

    for k in datasets[0].keys():
        if not all(k in df for df in datasets):
            raise ValueError(f"Variable {k} is not present in all datasets.")

    # Merge the retrieval types in one long array for each variable
    data = {}
    for k in datasets[0].keys():
        data[k] = np.concatenate([df[k] for df in datasets])

    lengths = [len(data[k]) for k in data]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Variables have different lengths: {', '.join(f'{k}: {ll}' for k, ll in zip(data, lengths))}"
        )

    df = pd.DataFrame(data)

    if df.empty:
        return None

    # QC flags:
    # 0 = good (valid flag + correct observation type for retrieval method)
    # 1 = invalid validity flag
    # 2 = wrong observation type for retrieval method
    qc_flags = np.zeros(len(df), dtype=int)

    # Mark observations with invalid validity flags
    # For `validity_flag`, 1 means valid, 0 means invalid
    invalid_validity = df["qc_pass"] != 1
    qc_flags[invalid_validity] = 1

    # observation_type: 0=invalid, 1=cloudy, 2=clear
    # Rayleigh: keep only clear (2); Mie: keep only cloudy (1)
    wrong_obs_type = (df["retrieval_type"] == "rayleigh") & (
        df["observation_type"] != 2
    ) | (df["retrieval_type"] == "mie") & (df["observation_type"] != 1)
    qc_flags[wrong_obs_type] = 2

    df["wrf_ensembly_qc"] = qc_flags

    # Scale HLOS from cm/s to m/s
    df["obs"] = df["obs"] / 100.0
    df["obs_err"] = df["obs_err"] / 100.0

    # Decode all time fields to UTC timestamps
    df["t"] = _decode_aeolus_time(df["t"])
    df["t_start"] = _decode_aeolus_time(df["t_start"])
    df["t_stop"] = _decode_aeolus_time(df["t_stop"])

    # Longitude from [0, 360] to [-180, 180]
    df["x"] = ((df["x"] + 180) % 360) - 180

    obs_df = pd.DataFrame()

    obs_df["time"] = df["t"]
    obs_df["longitude"] = df["x"]
    obs_df["latitude"] = df["y"]
    obs_df["z"] = df["z"]
    obs_df["z_type"] = "height"
    obs_df["value"] = df["obs"]
    obs_df["value_uncertainty"] = df["obs_err"]
    obs_df["qc_flag"] = df["wrf_ensembly_qc"]
    obs_df["instrument"] = df["retrieval_type"].map(
        {"mie": "AEOLUS_L2B_MIE", "rayleigh": "AEOLUS_L2B_RAYLEIGH"}
    )
    obs_df["quantity"] = "HLOS_WIND"

    # Fixed error estimates from Rennie et al. (2023), doi:10.5194/amt-16-2691-2023
    obs_df.loc[df["retrieval_type"] == "rayleigh", "value_uncertainty"] = 4.5
    obs_df.loc[df["retrieval_type"] == "mie", "value_uncertainty"] = 2.5

    obs_df["orig_coords"] = obs_df.apply(
        lambda row: {
            "indices": np.array((row.name,), dtype=int),
            "shape": np.array((len(obs_df),), dtype=int),
            "names": np.array(("measurement",), dtype=object),
        },
        axis=1,
    )

    obs_df["orig_filename"] = path.name

    obs_df["metadata"] = df.apply(
        lambda row: {
            "retrieval_type": row["retrieval_type"],
            "wind_result_id": int(row["wind_result_id"]),
            "azimuth": float(row["azimuth_a"]),
            "observation_type": int(row["observation_type"]),
            # These four variables can be used as 2D boundaries for plotting each wind result
            "time_start": row["t_start"],
            "time_stop": row["t_stop"],
            "alt_bottom": float(row["alt_bottom"]),
            "alt_top": float(row["alt_top"]),
        },
        axis=1,
    )

    obs_df = obs_df[obs_io.REQUIRED_COLUMNS]
    obs_io.validate_schema(obs_df)

    return obs_df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--mie/--no-mie", default=True, help="Include Mie wind retrievals")
@click.option(
    "--rayleigh/--no-rayleigh", default=True, help="Include Rayleigh wind retrievals"
)
def aeolus_l2b(input_path: Path, output_path: Path, mie: bool, rayleigh: bool):
    """Convert AEOLUS L2B file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the AEOLUS L2B file
    OUTPUT_PATH: Path where to save the converted observations (parquet)
    """
    _require_deps()
    print(f"Converting AEOLUS L2B file: {input_path}")
    print(f"Output path: {output_path}")

    converted_df = convert_aeolus_l2b(
        input_path, include_mie=mie, include_rayleigh=rayleigh
    )
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    obs_io.write_obs(converted_df, output_path)

    total_obs = len(converted_df)
    good_obs = len(converted_df[converted_df["qc_flag"] == 0])
    bad_obs = total_obs - good_obs

    print(f"Successfully converted {total_obs} observations:")
    print(f"  - {good_obs} good quality observations (QC=0)")
    print(f"  - {bad_obs} flagged observations (QC>0)")
    for instrument in sorted(converted_df["instrument"].unique()):
        subset = converted_df[converted_df["instrument"] == instrument]
        good = len(subset[subset["qc_flag"] == 0])
        print(f"    [{instrument}] {len(subset)} total, {good} good")
    print(f"Saved to: {output_path}")
