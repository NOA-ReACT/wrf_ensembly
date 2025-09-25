from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from wrf_ensembly import external
from wrf_ensembly import observations as obs
from wrf_ensembly import wrf
from wrf_ensembly.config import Config
from wrf_ensembly.console import logger
from wrf_ensembly.cycling import CycleInformation
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

    def _get_duckdb(self, read_only: bool) -> duckdb.DuckDBPyConnection:
        if read_only:
            return duckdb.connect(
                database=str(self.paths.obs_db),
                read_only=True,
            )

        # When also writing, ensure the observation table exists
        con = duckdb.connect(
            database=str(self.paths.obs_db),
            read_only=False,
        )
        con.execute("""
                    CREATE TABLE IF NOT EXISTS observations (
                        instrument STRING NOT NULL,
                        quantity STRING NOT NULL,
                        time TIMESTAMP NOT NULL,
                        longitude DOUBLE NOT NULL,
                        latitude DOUBLE NOT NULL,
                        x DOUBLE NOT NULL,
                        y DOUBLE NOT NULL,
                        z DOUBLE NOT NULL,
                        z_type STRING NOT NULL,
                        value DOUBLE,
                        value_uncertainty DOUBLE,
                        qc_flag INT NOT NULL,
                        orig_coords STRUCT(
                            indices INT[], shape INT[], names STRING[]
                        ) NOT NULL,
                        orig_filename STRING NOT NULL,
                        metadata STRING,
                        downsampling_info STRUCT(
                            method STRING,
                            obs_count INT,
                            time_spread_seconds DOUBLE,
                            spatial_spread_meters DOUBLE
                        )
            )""")
        return con

    def get_available_quantities(self) -> list[dict]:
        """Returns all combinations of instrument and quantity available in the database."""

        with self._get_duckdb(read_only=True) as con:
            result = con.execute("""
                SELECT instrument, quantity, COUNT(*) as count
                FROM observations
                GROUP BY instrument, quantity
                ORDER BY count DESC
            """).fetch_df()
        return result.to_dict(orient="records")

    def get_available_observations_overview(
        self,
    ) -> list[dict]:
        """Returns a dataframe with observation filenames and their row counts, min time and max time."""

        with self._get_duckdb(read_only=True) as con:
            result = con.execute("""
                SELECT
                    orig_filename as filename,
                    instrument as instrument,
                    MIN(time) as start_time,
                    MAX(time) as end_time,
                    COUNT(*) as count
                FROM observations
                GROUP BY orig_filename, instrument
                ORDER BY instrument, start_time
            """).fetch_df()
        return result.to_dict(orient="records")

    def delete_observation_file(self, filename: str) -> int:
        """
        Removes all observations from a specific file from the database.

        Args:
            filename: The name of the file to remove (not the full path)

        Returns:
            The number of observations removed.
        """

        with self._get_duckdb(read_only=False) as con:
            result = con.execute(
                f"DELETE FROM observations WHERE orig_filename = '{filename}'"
            )
            return result.rowcount

    def trim_observation_file(
        self, input_path: Path, output_path: Path
    ) -> tuple[str, int, int]:
        """
        Trims an observation file temporally and spatially according to the experiment configuration.
        The file is only created if there are observations left after trimming.

        This function assumes that preprocessing is completed (needs a wrfinput file to get the spatial bounds).

        Args:
            input_path: Path to the input observation file
            output_path: Path where the trimmed observation file will be saved

        Returns:
            The input file name, the number of observations in the original file, and the
            number of observations in the trimmed file.
        """

        filename = input_path.name
        df = obs.io.read_obs(input_path)
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
        df["in_domain"] = (
            (df["x"] >= x_min)
            & (df["x"] <= x_max)
            & (df["y"] >= y_min)
            & (df["y"] <= y_max)
        )

        # For spatial filtering, we must make sure that the final array has no NaNs after
        # reshaping into the original shape.
        # Thus, we gotta check if after grouping by the original coordinates (keeping
        # the fastest-changing dimension out), there are any groups with at least one
        # observation inside the domain. Only the rest can be thrown out.
        # This proceedure must be done for each orig_filename,quantity pair separately
        per_quantity_dfs = []
        for _, df_subset in df.groupby(["orig_filename", "quantity"]):
            if df_subset.empty:
                continue

            # Find the fastest-changing dimension (the one with the smallest size in 'shape')
            orig_coords = df_subset["orig_coords"].iloc[0]
            shape = orig_coords["shape"]
            smalled_dim_index = shape.argmin()

            # Create a column with the groups, excluding the fastest-changing dimension
            df_subset["group_key"] = df_subset["orig_coords"].apply(
                lambda x: tuple(
                    idx for i, idx in enumerate(x["indices"]) if i != smalled_dim_index
                )
            )

            # Find groups with at least one observation inside the domain
            valid_groups = df_subset[df_subset["in_domain"]].groupby("group_key").size()
            valid_group_keys = valid_groups[valid_groups > 0].index

            # Keep only observations in valid groups
            df_subset = df_subset[df_subset["group_key"].isin(valid_group_keys)]

            # Set all outside-domain observations to NaN
            df_subset.loc[~df_subset["in_domain"], "value"] = pd.NA
            df_subset.loc[~df_subset["in_domain"], "value_uncertainty"] = pd.NA

            per_quantity_dfs.append(df_subset)

        if not per_quantity_dfs:
            trimmed_len = 0
        else:
            df = pd.concat(per_quantity_dfs, ignore_index=True)
            df = df.drop(columns=["in_domain", "group_key"])
            trimmed_len = len(df.index)

        # Save the trimmed observations to the output file
        if trimmed_len > 0:
            obs.io.write_obs(df, output_path)

        return filename, original_len, trimmed_len

    def add_observation_file(self, input_path: Path) -> int:
        """
        Adds an observation file to the experiment DuckDB database.

        This function assumes the file is already in the WRF-Ensembly observation format
        and has been trimmed as needed.

        Args:
            input_path: Path to the observation file to add to the database

        Returns:
            The number of observations added to the database.
        """

        df = obs.io.read_obs(input_path)
        if df.empty:
            return 0

        # Grab a connection to the database and save the observations
        with self._get_duckdb(read_only=False) as con:
            # First, if observations from this file already exist, remove them
            con.execute(
                f"DELETE FROM observations WHERE orig_filename = '{input_path.name}'"
            )

            con.register("df_view", df)
            con.execute("""
                INSERT INTO observations (
                    instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                    value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata
                )
                SELECT
                    instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                    value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata
                FROM df_view
            """)

        return len(df.index)

    def apply_superorbing(self) -> None:
        """
        Applies the configured downsampling (super-orbing) methods to all observations in the database.

        It will clear any old downsampled observations and replace them with the new ones.

        The `observations.superorbing` configuration dictionary controls how to downsample each instrument. The keys are the `instrument.quantity` pairs, e.g. `radiosonde.temperature`, and the values are how to downsample, check the `SuperorbingConfig` dataclass in `config.py` for details.
        """

        if not self.cfg.observations.superorbing:
            logger.info("No superorbing configuration found, skipping downsampling.")
            return

        with self._get_duckdb(read_only=False) as con:
            # Clean up any old downsampled observations
            res = con.execute(
                "DELETE FROM observations WHERE downsampling_info IS NOT NULL"
            )
            if res.rowcount > 0:
                logger.info(f"Removed {res.rowcount} old downsampled observations.")

            for key, superorb_config in self.cfg.observations.superorbing.items():
                try:
                    instrument, quantity = key.split(".")
                except ValueError:
                    raise ValueError(
                        f"Invalid superorbing key '{key}', must be in the format 'instrument.quantity'"
                    )

                logger.info(f"Applying superorbing for {instrument}.{quantity}'")

                df = con.execute(
                    f"SELECT * FROM observations WHERE instrument = '{instrument}' AND quantity = '{quantity}'"
                ).fetchdf()
                if df.empty:
                    logger.warning(
                        f"No observations found for {instrument}.{quantity}, skipping superorbing."
                    )
                    continue

                df_superobed = obs.superorbing.superorb_dbscan(df, superorb_config)

                # Insert the new superobed observations
                con.register("df_superobed_view", df_superobed)
                con.execute("""
                    INSERT INTO observations (
                        instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                        value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata, downsampling_info
                    )
                    SELECT
                        instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                        value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata, downsampling_info
                    FROM df_superobed_view
                """)

                logger.info(
                    f"Superorbing complete for {instrument}.{quantity}, reduced from {len(df)} to {len(df_superobed)} observations."
                )

    def get_observations_for_cycle(
        self, cycle: CycleInformation
    ) -> pd.DataFrame | None:
        """
        Retrieves observation data for a specific cycle and set of instruments.

        Args:
            cycle: The cycle information to filter observations. The assimilation window from `cfg.assimilation.half_window_length_minutes` is applied.
        """

        instruments = self.cfg.observations.instruments_to_assimilate

        start_time = cycle.start - pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )
        end_time = cycle.end + pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )

        # Query the observations with duck db, find only files that overlap with the time window and instrument list
        with self._get_duckdb(read_only=True) as con:
            # If there are no super-orbed observations, we will use the original ones.
            superorbed_count = con.execute("""
                SELECT COUNT(*) FROM observations WHERE downsampling_info IS NOT NULL
            """).fetchone()
            superorbed_count = superorbed_count[0] if superorbed_count else 0
            superorbed_available = superorbed_count > 0

            if superorbed_available:
                logger.info("Using super-orbed observations for assimilation.")
                observations = observations = con.execute(
                    f"SELECT * FROM observations WHERE time >= '{start_time}' AND time <= '{end_time}' WHERE downsampling_info IS NOT NULL"
                ).fetchdf()
            else:
                logger.info(
                    "No super-orbed observations found, using original observations."
                )
                observations = con.execute(
                    f"SELECT * FROM observations WHERE time >= '{start_time}' AND time <= '{end_time}' AND downsampling_info IS NULL"
                ).fetchdf()

        if instruments is not None:
            observations = observations[observations["instrument"].isin(instruments)]

        return observations if not observations.empty else None

    def convert_cycle_to_dart(self, cycle: CycleInformation):
        """Converts the observations for a given cycle to DART obs_seq format."""

        df = self.get_observations_for_cycle(cycle)
        if df is None or df.empty:
            logger.info(
                f"No observations for cycle {cycle.index}, skipping DART conversion"
            )
            return

        output_path = self.paths.obs / f"obs_seq.{cycle.index:03d}"
        dart_process = obs.dart.convert_to_dart_obs_seq(
            dart_path=self.cfg.directories.dart_root,
            observations=df,
            output_location=output_path,
        )
        logger.info(
            f"Converting observations for cycle {cycle.index} to DART obs_seq..."
        )
        result = external.run(dart_process)
        if result.returncode != 0:
            logger.error(result.output)
            raise RuntimeError(
                f"DART conversion failed for cycle {cycle.index} with return code {result.returncode}"
            )
        logger.info(f"Wrote DART obs_seq file to {output_path}")
