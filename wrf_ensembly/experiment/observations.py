from dataclasses import dataclass
from pathlib import Path
import uuid

import duckdb
import pandas as pd
from wrf_ensembly.config import Config
from wrf_ensembly.cycling import CycleInformation
from wrf_ensembly import external, observations as obs, wrf
from wrf_ensembly.experiment.paths import ExperimentPaths


@dataclass
class ObservationFileMetadata:
    path: Path
    instrument: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp


class ExperimentObservations:
    """Manages observation files for a WRF-Ensembly experiment."""

    def __init__(
        self, config: Config, cycles: list[CycleInformation], paths: ExperimentPaths
    ):
        self.cfg = config
        self.cycles = cycles
        self.paths = paths

    def get_available_observation_files(self) -> list[ObservationFileMetadata]:
        """
        Scans the observation directory and returns metadata about available observation files.

        The files are stored at: `obs/INSTRUMENT/START_END_UUID_INSTRUMENT.parquet`
        """

        obs_files = []
        for instr_dir in self.paths.obs.iterdir():
            if not instr_dir.is_dir():
                continue
            instrument = instr_dir.name
            for file_path in instr_dir.glob("*.parquet"):
                try:
                    parts = file_path.stem.split("_")
                    # Format: START_END_UUID_INSTRUMENT
                    if len(parts) >= 4:
                        start_str, end_str = parts[0], parts[1]
                        # parts[2] is UUID (ignored for metadata)
                        # parts[3] should be instrument name
                    else:
                        raise ValueError(f"Unexpected file format: {file_path.stem}")

                    start_time = pd.to_datetime(
                        start_str, format="%Y%m%d%H%M"
                    ).tz_localize("UTC")
                    end_time = pd.to_datetime(end_str, format="%Y%m%d%H%M").tz_localize(
                        "UTC"
                    )
                    obs_files.append(
                        ObservationFileMetadata(
                            path=file_path,
                            instrument=instrument,
                            start_time=start_time,
                            end_time=end_time,
                        )
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not parse observation file name {file_path}: {e}"
                    )
        return obs_files

    def add_observation_file(self, file_path: Path):
        """
        Adds an observation file to the experiment:

        - Ensures it is in the WRF-Ensembly observation format
        - Trims it temporally and spatially according to the experiment configuration
        - If it contains multiple instruments, splits it into separate files per instrument
        - Saves it in the experiment's observation directory (obs/INSTRUMENT/UUID_INSTRUMENT.parquet)

        Each file gets a unique UUID to ensure thread-safe parallel processing.

        This function assumes that preprocessing is completed (needs a wrfinput file to get the spatial bounds).
        """

        df = obs.io.read_obs(file_path)
        original_len = len(df.index)

        # Find wrfinput file
        if not self.cfg.data.per_member_meteorology:
            wrfinput_path = self.paths.data_icbc / "wrfinput_d01_cycle_0"
        else:
            wrfinput_path = self.paths.data_icbc / "member_00" / "wrfinput_d01_cycle_0"
        if not wrfinput_path.exists():
            raise FileNotFoundError(
                f"wrfinput file not found at {wrfinput_path}, cannot trim observations spatially"
            )

        # Trim file into experiment time and space bounds
        transformer = wrf.get_wrf_proj_transformer(self.cfg.domain_control)
        start_time, end_time = wrf.get_temporal_domain_bounds(self.cycles)
        x_min, x_max, y_min, y_max = wrf.get_spatial_domain_bounds(wrfinput_path)

        df = obs.utils.project_locations_to_wrf(df, transformer)
        df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
        df = df[
            (df["x"] >= x_min)
            & (df["x"] <= x_max)
            & (df["y"] >= y_min)
            & (df["y"] <= y_max)
        ]

        trimmed_len = len(df.index)
        print(
            f"Trimmed observation file {file_path}: {original_len} -> {trimmed_len} observations"
        )
        if trimmed_len == 0:
            print(
                f"No observations left after trimming, skipping file {file_path.name}."
            )
            return

        # If required, split into separate files per instrument
        instruments = df["instrument"].unique()
        dataframes = {}
        for instr in instruments:
            df_instr = df[df["instrument"] == instr].copy()
            dataframes[instr] = df_instr

        # Write output files
        for instr, df in dataframes.items():
            output_dir = self.paths.obs / instr
            output_dir.mkdir(parents=True, exist_ok=True)

            # Use full UUID for guaranteed thread-safe unique naming
            file_uuid = uuid.uuid4().hex
            output_path = output_dir / f"{file_uuid}_{instr}.parquet"

            obs.io.write_obs(df, output_path)
            print(f"Wrote {len(df.index)} observations to {output_path}")

    def get_observations_for_cycle(
        self, cycle: CycleInformation
    ) -> pd.DataFrame | None:
        """
        Retrieves observation data for a specific cycle and set of instruments.

        Args:
            cycle: The cycle information to filter observations. The assimilation window from `cfg.assimilation.half_window_length_minutes` is applied.
        """

        instruments = self.cfg.observations.instruments_to_assimilate
        if instruments is not None:
            print(f"Filtering observations to instruments: {instruments}")

        start_time = cycle.start - pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )
        end_time = cycle.end + pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )

        # Query the observations with duck db, find only files that overlap with the time window and instrument list
        con = duckdb.connect()
        query = f"SELECT * FROM read_parquet('{self.paths.obs}/**/*.parquet') WHERE time >= '{start_time}' AND time <= '{end_time}'"
        observations = con.execute(query).fetchdf()
        if instruments is not None:
            observations = observations[observations["instrument"].isin(instruments)]

        return observations if not observations.empty else None

    def convert_cycle_to_dart(self, cycle: CycleInformation):
        """Converts the observations for a given cycle to DART obs_seq format."""

        df = self.get_observations_for_cycle(cycle)
        if df is None or df.empty:
            print(f"No observations for cycle {cycle.index}, skipping DART conversion")
            return

        output_path = self.paths.obs / f"obs_seq.{cycle.index:03d}"
        dart_process = obs.dart.convert_to_dart_obs_seq(
            dart_path=self.cfg.directories.dart_root,
            observations=df,
            output_location=output_path,
        )
        print(f"Converting observations for cycle {cycle.index} to DART obs_seq...")
        result = external.run(dart_process)
        if result.returncode != 0:
            print(result.output)
            raise RuntimeError(
                f"DART conversion failed for cycle {cycle.index} with return code {result.returncode}"
            )
        print(f"Wrote DART obs_seq file to {output_path}")
