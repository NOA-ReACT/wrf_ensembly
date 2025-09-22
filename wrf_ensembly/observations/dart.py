"""Functions about converting from and to DART obs_seq files"""

from pathlib import Path

import pandas as pd

from wrf_ensembly.external import ExternalProcess

OBS_TYPE_TABLE = {
    "AOD_500nm": "AIRSENSE_AOD",
    "AOD_550nm": "AIRSENSE_AOD",
}


def convert_to_dart_obs_seq(
    dart_path: Path, observations: pd.DataFrame, output_location: Path
) -> ExternalProcess:
    """
    Converts the given observations DataFrame to a DART obs_seq file at the given location.
    It will use the `wrf_ensembly` observation converter we wrote into DART for this purpose.

    The external command is not actually executed, but an ExternalProcess object is returned so
    you can run it when convenient (e.g. in parallel with other commands).
    """

    # Locate the converter
    converter = (
        dart_path
        / "observations"
        / "obs_converters"
        / "wrf_ensembly"
        / "work"
        / "convert_from_wrf_ensembly"
    )
    if not converter.exists() or not converter.is_file():
        raise FileNotFoundError(
            f"Could not find DART converter at {converter}. Is it compiled? Try {converter.parent / 'quickbuild.sh'}"
        )
    if not output_location.parent.is_dir():
        raise NotADirectoryError(
            f"Output location parent {output_location.parent} is not a directory"
        )

    # The converter uses stdin for the input and a command line argument for the output, so we must
    # convert the dataframe to a CSV string. The column order is:
    # obs_type, longitude, latitude, vert, year, month, day, hour, minute, second, obs_value, obs_error, obs_meta
    observations = observations.copy()
    observations["obs_type"] = (
        observations["quantity"].map(OBS_TYPE_TABLE).fillna(observations["quantity"])
    )
    observations["year"] = observations["time"].dt.year
    observations["month"] = observations["time"].dt.month
    observations["day"] = observations["time"].dt.day
    observations["hour"] = observations["time"].dt.hour
    observations["minute"] = observations["time"].dt.minute
    observations["second"] = observations["time"].dt.second
    observations["vert"] = observations["z"]
    observations["obs_value"] = observations["value"]
    observations["obs_error"] = observations["value_uncertainty"]
    observations["obs_meta"] = observations["metadata"].apply(lambda x: str(x))
    observations = observations[
        [
            "obs_type",
            "longitude",
            "latitude",
            "vert",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "obs_value",
            "obs_error",
            "obs_meta",
        ]
    ]

    # Longitude must be in [0, 360) range for DART
    observations["longitude"] = (observations["longitude"] + 360) % 360

    # Delete rows where AOD is NaN
    observations = observations.dropna(subset=["obs_value"])

    csv_data = observations.to_csv(index=False)

    return ExternalProcess(
        command=[converter.resolve(), "-", output_location.resolve()],
        stdin=csv_data,
        cwd=converter.parent.resolve(),
    )


# TODO Add functions to read from obs_seq files back to DataFrames
