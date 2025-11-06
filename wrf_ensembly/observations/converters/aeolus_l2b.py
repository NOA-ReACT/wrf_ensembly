"""Converter for AEOLUS L2B files to WRF-Ensembly Observation format."""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from wrf_ensembly.observations import io as obs_io

try:
    import coda

    HAS_CODA = True
except (ImportError, OSError):
    HAS_CODA = False
    coda = None


# Variable mappings for Mie retrievals
mie_variables = {
    "wind_result_id": ("mie_geolocation", -1, "wind_result_id"),
    "z": ("mie_geolocation", -1, "windresult_geolocation", "altitude_vcog"),
    "azimuth_a": ("mie_geolocation", -1, "windresult_geolocation", "los_azimuth"),
    "y": ("mie_geolocation", -1, "windresult_geolocation", "latitude_cog"),
    "x": ("mie_geolocation", -1, "windresult_geolocation", "longitude_cog"),
    "t": ("mie_geolocation", -1, "windresult_geolocation", "datetime_cog"),
    "obs": ("mie_hloswind", -1, "windresult", "mie_wind_velocity"),
    "obs_err": ("mie_wind_prod_conf_data", -1, "mie_wind_qc", "hlos_error_estimate"),
    "qc_pass": ("mie_hloswind", -1, "windresult", "validity_flag"),
    "observation_type": ("mie_hloswind", -1, "windresult", "observation_type"),
}

# Variable mappings for Rayleigh retrievals
rayleigh_variables = {
    "wind_result_id": ("rayleigh_geolocation", -1, "wind_result_id"),
    "z": ("rayleigh_geolocation", -1, "windresult_geolocation", "altitude_vcog"),
    "azimuth_a": ("rayleigh_geolocation", -1, "windresult_geolocation", "los_azimuth"),
    "y": ("rayleigh_geolocation", -1, "windresult_geolocation", "latitude_cog"),
    "x": (
        "rayleigh_geolocation",
        -1,
        "windresult_geolocation",
        "longitude_cog",
    ),
    "t": ("rayleigh_geolocation", -1, "windresult_geolocation", "datetime_cog"),
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


def convert_aeolus_l2b(path: Path) -> pd.DataFrame | None:
    """Convert an AEOLUS L2B file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AEOLUS L2B file.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid data.

    Raises:
        ImportError: If the coda library is not installed.
    """
    if not HAS_CODA:
        raise ImportError(
            "The 'coda' library is required for reading AEOLUS L2B files. "
            "Please install it using: pip install coda"
        )

    # Open the CODA file
    cf = coda.open(str(path))  # type: ignore[union-attr]
    if cf.product_type != "ALD_U_N_2B":
        cf.close()
        raise ValueError("Input file is not an AEOLUS DBL L2B file.")

    # Read all variables we need from the DBL file, per retrieval type
    datasets = []
    for name, variables in dict(mie=mie_variables, rayleigh=rayleigh_variables).items():
        data = {}
        for k, v in variables.items():
            try:
                data[k] = coda.fetch(cf, *v)  # type: ignore[union-attr]
            except coda.CodacError:  # type: ignore[union-attr]
                # If we can't fetch a variable, fill with NaN or appropriate default
                if k in ["qc_pass", "observation_type"]:
                    data[k] = np.array([])
                else:
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

    # Ensure same variables in all datasets
    for k in datasets[0].keys():
        if not all(k in df for df in datasets):
            raise ValueError(f"Variable {k} is not present in all datasets.")

    # Merge the retrieval types in one long array for each variable
    data = {}
    for k in datasets[0].keys():
        data[k] = np.concatenate([df[k] for df in datasets])

    # Check that all variables have the same length
    lengths = [len(data[k]) for k in data]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Variables have different lengths: {', '.join(f'{k}: {ll}' for k, ll in zip(data, lengths))}"
        )

    df = pd.DataFrame(data)

    if df.empty:
        return None

    # Create QC flags based on data quality:
    # QC = 0: Good quality (validity flag = 1 and correct observation type)
    # QC = 1: Invalid validity flag
    # QC = 2: Wrong observation type for retrieval method

    # Start with all observations as potentially good
    qc_flags = np.zeros(len(df), dtype=int)

    # Mark observations with invalid validity flags
    invalid_validity = df["qc_pass"] != 1
    qc_flags[invalid_validity] = 1

    # Mark observations with wrong observation type for retrieval method
    # Rayleigh should have observation type 2 (Rayleigh cloudy)
    # Mie should have observation type 1 (Mie clear)
    wrong_obs_type = ~(
        ((df["retrieval_type"] == "rayleigh") & (df["observation_type"] == 2))
        | ((df["retrieval_type"] == "mie") & (df["observation_type"] == 1))
    )
    qc_flags[wrong_obs_type] = 2

    # Store QC flags for later use
    df["wrf_ensembly_qc"] = qc_flags

    # Scale HLOS to m/s (from cm/s)
    df["obs"] = df["obs"] / 100.0
    df["obs_err"] = df["obs_err"] / 100.0

    # Convert date from 'seconds since 2000' to pd.Timestamp
    df["t"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["t"], unit="seconds")
    # Ensure UTC timezone
    df["t"] = df["t"].dt.tz_localize("UTC")

    # Convert longitude from [0, 360] to [-180, 180]
    df["x"] = ((df["x"] + 180) % 360) - 180

    # Create the WRF-Ensembly observation format DataFrame
    obs_df = pd.DataFrame()

    # Map columns to WRF-Ensembly format
    obs_df["time"] = df["t"]
    obs_df["longitude"] = df["x"]
    obs_df["latitude"] = df["y"]
    obs_df["z"] = df["z"]
    obs_df["z_type"] = "height"  # Altitude is in meters above geoid
    obs_df["value"] = df["obs"]
    obs_df["value_uncertainty"] = df["obs_err"]
    obs_df["qc_flag"] = df["wrf_ensembly_qc"]  # Use the QC flags we computed
    obs_df["instrument"] = "AEOLUS_L2B"
    obs_df["quantity"] = "HLOS_WIND"  # Horizontal Line of Sight wind

    # Set rayleight error to 4.5 m/s and mie error to 2.5 m/s
    # https://doi.org/10.5194/amt-16-2691-2023
    obs_df.loc[df["retrieval_type"] == "rayleigh", "value_uncertainty"] = 4.5
    obs_df.loc[df["retrieval_type"] == "mie", "value_uncertainty"] = 2.5

    # Create orig_coords for traceability
    obs_df["orig_coords"] = obs_df.apply(
        lambda row: {
            "indices": np.array((row.name,), dtype=int),
            "shape": np.array((len(obs_df),), dtype=int),
            "names": np.array(("measurement",), dtype=object),
        },
        axis=1,
    )

    obs_df["orig_filename"] = path.name

    # Add metadata with retrieval type and wind result ID
    obs_df["metadata"] = df.apply(
        lambda row: {
            "retrieval_type": row["retrieval_type"],
            "wind_result_id": int(row["wind_result_id"]),
            "azimuth": float(row["azimuth_a"]),
            "observation_type": int(row["observation_type"]),
        },
        axis=1,
    )

    # Sort columns as defined in the schema and validate
    obs_df = obs_df[obs_io.REQUIRED_COLUMNS]
    obs_io.validate_schema(obs_df)

    return obs_df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def aeolus_l2b(input_path: Path, output_path: Path):
    """Convert AEOLUS L2B file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the AEOLUS L2B file
    OUTPUT_PATH: Path where to save the converted observations (will be saved as parquet)
    """
    if not HAS_CODA:
        raise click.ClickException(
            "The 'coda' library is required for reading AEOLUS L2B files. "
            "Please install it using: pip install coda"
        )

    print(f"Converting AEOLUS L2B file: {input_path}")
    print(f"Output path: {output_path}")

    # Convert the data
    converted_df = convert_aeolus_l2b(input_path)
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    # Report statistics
    total_obs = len(converted_df)
    good_obs = len(converted_df[converted_df["qc_flag"] == 0])
    bad_obs = total_obs - good_obs

    print(f"Successfully converted {total_obs} observations:")
    print(f"  - {good_obs} good quality observations (QC=0)")
    print(f"  - {bad_obs} flagged observations (QC>0)")
    print(f"Saved to: {output_path}")
