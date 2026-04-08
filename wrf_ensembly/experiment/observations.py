import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

from wrf_ensembly import external, wrf
from wrf_ensembly import observations as obs
from wrf_ensembly.config import Config
from wrf_ensembly.console import logger
from wrf_ensembly.cycling import CycleInformation
from wrf_ensembly.experiment.paths import ExperimentPaths
from wrf_ensembly.superobs import grid_bin


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
            con = duckdb.connect(
                database=str(self.paths.obs_db),
                read_only=True,
            )
            con.execute("SET TimeZone='UTC';")
            return con

        # When also writing, ensure the observation table exists
        con = duckdb.connect(
            database=str(self.paths.obs_db),
            read_only=False,
        )
        con.execute("SET TimeZone='UTC';")
        con.execute(
            """
                    CREATE TABLE IF NOT EXISTS observations (
                        instrument STRING NOT NULL,
                        quantity STRING NOT NULL,
                        time TIMESTAMPTZ NOT NULL,
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
                        metadata JSON,
                        cluster_id STRING,
                        model_forecast DOUBLE,
                        model_analysis DOUBLE,
                        used_in_da BOOLEAN NOT NULL DEFAULT FALSE,
                        da_cycle INT
            )"""
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_time ON observations (time)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_filename ON observations (orig_filename)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_instrument_time ON observations (instrument, time)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_da ON observations (da_cycle, used_in_da)"
        )
        return con

    def get_available_quantities(self) -> list[dict]:
        """Returns all combinations of instrument and quantity available in the database."""

        with self._get_duckdb(read_only=True) as con:
            result = con.execute(
                """
                SELECT instrument, quantity, COUNT(*) as count
                FROM observations
                GROUP BY instrument, quantity
                ORDER BY count DESC
            """
            ).fetch_df()
        return result.to_dict(orient="records")

    def get_available_observations_overview(
        self,
    ) -> list[dict]:
        """Returns a dataframe with observation filenames and their row counts, min time and max time."""

        with self._get_duckdb(read_only=True) as con:
            result = con.execute(
                """
                SELECT
                    orig_filename as filename,
                    instrument as instrument,
                    MIN(time AT TIME ZONE 'UTC') as start_time,
                    MAX(time AT TIME ZONE 'UTC') as end_time,
                    COUNT(*) as count,
                    COUNT(model_forecast) as model_values
                FROM observations
                GROUP BY orig_filename, instrument
                ORDER BY instrument, start_time
            """
            ).fetch_df()
        return result.to_dict(orient="records")

    def get_observations_by_filename(self, filename: str) -> pd.DataFrame | None:
        """
        Retrieves all observations from the database that match the given orig_filename.

        Args:
            filename: The original filename to filter on (not a full path).

        Returns:
            A DataFrame of matching observations, or None if no observations were found.
        """

        with self._get_duckdb(read_only=True) as con:
            result = con.execute(
                "SELECT *, time AT TIME ZONE 'UTC' FROM observations WHERE orig_filename = ?",
                [filename],
            ).fetchdf()

        if result.empty:
            return None

        if result["time"].dt.tz is None:
            result["time"] = result["time"].dt.tz_localize("UTC")

        return result

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
            wrfinput_path = (
                self.paths.data_icbc / "member_00" / "wrfinput_d01_member_00_cycle_0"
            )
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

        # For each pair of instrument-quantity inside the data, check if there is a superob config definition
        groups = []

        df["instrument_quantity"] = df["instrument"] + "." + df["quantity"]
        for iq in df["instrument_quantity"].unique():
            group = df.loc[df["instrument_quantity"] == iq]

            if iq in self.cfg.observations.superobs:
                superob_options = self.cfg.observations.superobs[iq]

                before_n = len(group)
                group = grid_bin(
                    group, superob_options.hoz_bin_sizes, superob_options.vert_bin_sizes
                )
                after_n = len(group)

                print(
                    f"{iq}: Generated {after_n} superobs from {before_n} observations."
                )

            groups.append(group)

        # Reconcatenate everything into one dataframe
        df = pd.concat(groups)

        # Convert orig_coords to a pyarrow-backed column so DuckDB sees it as STRUCT
        # (plain Python dicts get inferred as MAP, which can't be cast to STRUCT)
        oc_type = pa.struct(
            [
                ("indices", pa.list_(pa.int32())),
                ("shape", pa.list_(pa.int32())),
                ("names", pa.list_(pa.string())),
            ]
        )
        oc_array = pa.array(df["orig_coords"].tolist(), type=oc_type)
        df["orig_coords"] = pd.ArrowDtype(oc_type).__from_arrow__(oc_array)

        # Ensure time is in UTC
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")

        # Mark observations that fall within a DA assimilation window
        da_instruments = self.cfg.observations.instruments_to_assimilate
        half_window = self.cfg.assimilation.half_window_length_minutes

        df["used_in_da"] = False
        df["da_cycle"] = pd.NA

        for cycle in self.cycles:
            start_time = pd.to_datetime(cycle.end) - pd.Timedelta(minutes=half_window)
            end_time = pd.to_datetime(cycle.end) + pd.Timedelta(minutes=half_window)

            in_window = (df["time"] >= start_time) & (df["time"] <= end_time)
            if da_instruments is not None:
                is_da_instrument = df["instrument"].isin(da_instruments)
                df.loc[in_window & is_da_instrument, "used_in_da"] = True
                df.loc[in_window & is_da_instrument, "da_cycle"] = cycle.index
            else:
                df.loc[in_window, "used_in_da"] = True
                df.loc[in_window, "da_cycle"] = cycle.index

        # Grab a connection to the database and save the observations
        with self._get_duckdb(read_only=False) as con:
            # First, if observations from this file already exist, remove them
            con.execute(
                f"DELETE FROM observations WHERE orig_filename = '{input_path.name}'"
            )

            con.register("df_view", df)
            con.execute(
                """
                INSERT INTO observations (
                    instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                    value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata,
                    used_in_da, da_cycle
                )
                SELECT
                    instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                    value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata,
                    used_in_da, da_cycle
                FROM df_view
            """
            )

        return len(df.index)

    def update_model_source_values(self, df: pd.DataFrame, source: str) -> int:
        """
        Update model_forecast or model_analysis for observations using their rowid.

        The DataFrame must contain 'rowid' and 'model_value' columns.
        All existing entries for the target column are reset to NULL before
        applying the new values.

        Args:
            df: DataFrame with 'rowid' and 'model_value' columns.
            source: Either 'forecast' or 'analysis', selects the target column.

        Returns:
            The number of rows updated.
        """
        col = f"model_{source}"

        with self._get_duckdb(read_only=False) as con:
            con.execute(f"UPDATE observations SET {col} = NULL")

            update_df = df[["rowid", "model_value"]].copy()
            con.register("model_values_view", update_df)
            con.execute(
                f"""
                UPDATE observations
                SET {col} = mv.model_value
                FROM model_values_view mv
                WHERE observations.rowid = mv.rowid
                """
            )

        return len(update_df)

    def get_model_interpolated(
        self,
        start_date: dt.datetime | None = None,
        end_date: dt.datetime | None = None,
    ) -> pd.DataFrame | None:
        """
        Retrieves all observations that have a model_value set.

        Args:
            start_date: If set, only return observations at or after this time.
            end_date: If set, only return observations before or at this time.

        Returns:
            DataFrame of observations with model_value, or None if none exist.
        """

        query = "SELECT *, time AT TIME ZONE 'UTC' FROM observations WHERE model_forecast IS NOT NULL"
        params = []
        if start_date is not None:
            query += " AND time >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND time <= ?"
            params.append(end_date)

        with self._get_duckdb(read_only=True) as con:
            result = con.execute(query, params).fetchdf()

        if result.empty:
            return None

        if result["time"].dt.tz is None:
            result["time"] = result["time"].dt.tz_localize("UTC")

        return result

    def get_cycle_summary(self, cycles: list[CycleInformation]) -> pd.DataFrame:
        """
        Returns a summary of observations per cycle: total count and how many are to be assimilated.

        Args:
            cycles: List of cycles to summarize.

        Returns:
            DataFrame with columns: cycle_index, total, to_assimilate
        """
        rows = []
        with self._get_duckdb(read_only=True) as con:
            for cycle in cycles:
                result = con.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE used_in_da AND da_cycle = ?) AS to_assimilate
                    FROM observations
                    WHERE time >= ? AND time <= ?
                    """,
                    [cycle.index, str(cycle.start), str(cycle.end)],
                ).fetchone()
                total, to_assimilate = result if result else (0, 0)
                rows.append(
                    {
                        "cycle_index": cycle.index,
                        "total": int(total),
                        "to_assimilate": int(to_assimilate),
                    }
                )
        return pd.DataFrame(rows, columns=["cycle_index", "total", "to_assimilate"])

    def get_cycle_file_info(self, cycle: CycleInformation) -> pd.DataFrame:
        """
        Returns per-file observation statistics for a cycle's time window (start to end).

        Each row represents one (orig_filename, instrument) combination and includes:
        - total: total observations in the cycle time window
        - assimilated: observations marked as used_in_da for this cycle

        Args:
            cycle: The cycle whose start/end times define the time window.
        """

        with self._get_duckdb(read_only=True) as con:
            result = con.execute(
                """
                SELECT
                    orig_filename,
                    instrument,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE used_in_da AND da_cycle = ?) AS assimilated
                FROM observations
                WHERE time >= ? AND time <= ?
                GROUP BY orig_filename, instrument
                ORDER BY instrument, orig_filename
                """,
                [cycle.index, str(cycle.start), str(cycle.end)],
            ).fetchdf()
        return result

    def get_observations_for_cycle(
        self, cycle: CycleInformation
    ) -> pd.DataFrame | None:
        """
        Retrieves observation data for a specific cycle and set of instruments.
        Only returns observations that match a cycle's assimilation window and are from
        instruments in the `cfg.observations.instruments_to_assimilate` list.

        The error inflation factor from `cfg.observations.error_inflation_factor` is applied
        to the observation uncertainties.

        Args:
            cycle: The cycle information to filter observations. The assimilation window from `cfg.assimilation.half_window_length_minutes` is applied.
        """

        instruments = self.cfg.observations.instruments_to_assimilate

        start_time = cycle.end - pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )
        end_time = cycle.end + pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )

        # Query the observations with duck db, find only files that overlap with the time window and instrument list
        with self._get_duckdb(read_only=True) as con:
            if instruments is not None:
                placeholders = ", ".join("?" * len(instruments))
                observations = con.execute(
                    f"SELECT *, time AT TIME ZONE 'UTC' FROM observations WHERE time >= ? AND time <= ? AND instrument IN ({placeholders})",
                    [start_time, end_time, *instruments],
                ).fetchdf()
            else:
                observations = con.execute(
                    "SELECT *, time AT TIME ZONE 'UTC' FROM observations WHERE time >= ? AND time <= ?",
                    [start_time, end_time],
                ).fetchdf()

        if observations.empty:
            return None

        # Ensure observations time is timezone-aware in UTC
        if observations["time"].dt.tz is None:
            observations["time"] = observations["time"].dt.tz_localize("UTC")

        # Sometimes superobs might be a bit outside the time window, so filter again
        observations = observations[
            (observations["time"] >= start_time) & (observations["time"] <= end_time)
        ]

        # Drop anything that has zero variance, as that will break DART
        zero_variance = (observations["value_uncertainty"] <= 0) | (
            observations["value_uncertainty"].isna()
        )
        observations.loc[zero_variance, "qc_flag"] = 99  # Mark as bad quality

        # Ensure observation error is never negative
        negative_error = observations["value_uncertainty"] < 0
        observations.loc[negative_error, "value_uncertainty"] = 0

        logger.info(f"Got {len(observations)} observations for cycle {cycle.index}")

        # Apply error inflation if configured
        if not observations.empty and self.cfg.observations.error_inflation_factor:

            def inflate_error(row):
                key = f"{row['instrument']}.{row['quantity']}"
                factor = self.cfg.observations.error_inflation_factor.get(key, 1.0)
                return row["value_uncertainty"] * factor

            observations["value_uncertainty"] = observations.apply(
                inflate_error, axis=1
            )

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
