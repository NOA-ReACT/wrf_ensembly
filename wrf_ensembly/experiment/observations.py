import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from wrf_ensembly import external, wrf
from wrf_ensembly import observations as obs
from wrf_ensembly.config import Config
from wrf_ensembly.console import logger
from wrf_ensembly.cycling import CycleInformation, cycles_to_dataframe
from wrf_ensembly.experiment.paths import ExperimentPaths
from wrf_ensembly.superobs import grid_bin, stride_thin, time_bin


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
                        model_forecast DOUBLE,
                        model_analysis DOUBLE,
                        model_forecast_spread DOUBLE,
                        model_analysis_spread DOUBLE
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_obs_time ON observations (time)")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_filename ON observations (orig_filename)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_instrument_time ON observations (instrument, time)"
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
                "DELETE FROM observations WHERE orig_filename = ?", [filename]
            )
            return result.rowcount

    def trim_observation_file(
        self,
        input_path: Path,
        output_path: Path,
        ignore_instrument_quantity_pairs: list[str] | None = None,
    ) -> tuple[str, int, int]:
        """
        Trims an observation file temporally and spatially according to the experiment configuration.
        The file is only created if there are observations left after trimming.

        This function assumes that preprocessing is completed (needs a wrfinput file to get the spatial bounds).

        Args:
            input_path: Path to the input observation file
            output_path: Path where the trimmed observation file will be saved
            ignore_instrument_quantity_pairs: Optional list of instrument-quantity pairs to ignore (e.g. ["instrument1.quantity1", "instrument2.quantity2"])

        Returns:
            The input file name, the number of observations in the original file, and the
            number of observations in the trimmed file.
        """

        filename = input_path.name
        df = obs.io.read_obs(input_path)
        original_len = len(df.index)

        if ignore_instrument_quantity_pairs:
            for pair in ignore_instrument_quantity_pairs:
                instrument, quantity = pair.split(".")
                df = df[
                    ~((df["instrument"] == instrument) & (df["quantity"] == quantity))
                ]

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
            indices_array = np.array([oc["indices"] for oc in df_subset["orig_coords"]])
            mask = np.ones(indices_array.shape[1], dtype=bool)
            mask[smalled_dim_index] = False
            df_subset["group_key"] = [tuple(row) for row in indices_array[:, mask]]

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

        # Apply superobs only to instrument-quantity pairs that have a config definition
        df["instrument_quantity"] = df["instrument"] + "." + df["quantity"]
        superob_keys = set(df["instrument_quantity"].unique()) & set(
            self.cfg.observations.superobs.keys()
        )

        if superob_keys:
            needs_superobs = df["instrument_quantity"].isin(superob_keys)
            groups = [df[~needs_superobs]]

            for iq in superob_keys:
                group = df.loc[df["instrument_quantity"] == iq]
                superob_options = self.cfg.observations.superobs[iq]

                before_n = len(group)
                group = grid_bin(
                    group,
                    superob_options.hoz_bin_sizes,
                    superob_options.vert_bin_sizes,
                    superob_options.reduce_instrument_error,
                )
                print(
                    f"{iq}: Generated {len(group)} superobs from {before_n} observations."
                )
                groups.append(group)

            df = pd.concat(groups)

        # Validate mutual exclusivity of spatial superobbing and temporal binning
        overlap = set(self.cfg.observations.superobs) & set(
            self.cfg.observations.temporal_binning
        )
        if overlap:
            raise ValueError(
                f"These instrument-quantity pairs appear in both 'superobs' and "
                f"'temporal_binning' — they are mutually exclusive: {overlap}"
            )

        # Apply temporal binning (alternative to grid_bin, not in sequence)
        temporal_keys = set(df["instrument_quantity"].unique()) & set(
            self.cfg.observations.temporal_binning.keys()
        )
        if temporal_keys:
            needs_temporal = df["instrument_quantity"].isin(temporal_keys)
            parts = [df[~needs_temporal]]
            for iq in temporal_keys:
                group = df.loc[df["instrument_quantity"] == iq]
                tb_cfg = self.cfg.observations.temporal_binning[iq]
                before_n = len(group)
                group = time_bin(
                    group,
                    tb_cfg.bin_minutes,
                    tb_cfg.offset_minutes,
                    tb_cfg.reduce_instrument_error,
                )
                print(
                    f"{iq}: Temporal binning: {len(group)} bins from {before_n} obs "
                    f"({tb_cfg.bin_minutes} min windows, offset {tb_cfg.offset_minutes} min)."
                )
                parts.append(group)
            # Normalise value_uncertainty to float64 across all parts before
            # concat — AERONET and similar instruments leave it as None/object,
            # which triggers a FutureWarning when mixed with float64 partitions.
            parts = [
                p.assign(
                    value_uncertainty=pd.to_numeric(
                        p["value_uncertainty"], errors="coerce"
                    )
                )
                for p in parts
            ]
            df = pd.concat(parts)

        # Apply stride thinning (after superobbing)
        thinning_keys = set(df["instrument_quantity"].unique()) & set(
            self.cfg.observations.thinning.keys()
        )
        if thinning_keys:
            parts = [df[~df["instrument_quantity"].isin(thinning_keys)]]
            for iq in thinning_keys:
                group = df.loc[df["instrument_quantity"] == iq].copy()
                thin_opts = self.cfg.observations.thinning[iq]
                before_good = (group["qc_flag"] == 0).sum()
                group = stride_thin(group, thin_opts.keep_every_n)
                after_good = (group["qc_flag"] == 0).sum()
                print(
                    f"{iq}: Thinned to {after_good} DA obs from {before_good} good-QC obs "
                    f"(keep_every_n={thin_opts.keep_every_n})."
                )
                parts.append(group)
            df = pd.concat(parts)

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

        # Grab a connection to the database and save the observations
        with self._get_duckdb(read_only=False) as con:
            # First, if observations from this file already exist, remove them
            con.execute(
                "DELETE FROM observations WHERE orig_filename = ?", [input_path.name]
            )

            con.register("df_view", df)
            con.execute(
                """
                INSERT INTO observations (
                    instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                    value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata
                )
                SELECT
                    instrument, quantity, time, longitude, latitude, x, y, z, z_type,
                    value, value_uncertainty, qc_flag, orig_coords, orig_filename, metadata
                FROM df_view
            """
            )

        return len(df.index)

    def update_model_source_values(
        self, df: pd.DataFrame, source: str, clear: bool = True
    ) -> int:
        """
        Update model_forecast or model_analysis for observations using their rowid.

        The DataFrame must contain 'rowid' and 'model_value' columns.

        Args:
            df: DataFrame with 'rowid' and 'model_value' columns.
            source: Either 'forecast' or 'analysis', selects the target column.
            clear: If True, reset all existing entries for the target column to
                NULL before applying the new values. Callers that write the
                result in multiple chunks must pass ``clear=True`` only for the
                first chunk; otherwise each chunk would wipe the previous ones
                (the whole column is cleared, but each call only writes its own
                rowids).

        Returns:
            The number of rows updated.
        """
        col = f"model_{source}"

        with self._get_duckdb(read_only=False) as con:
            update_df = df[["rowid", "model_value"]].copy()
            con.register("model_values_view", update_df)

            # Clear old values and apply new ones atomically, so a failure in
            # the populating UPDATE cannot leave the column fully NULLed.
            con.execute("BEGIN TRANSACTION")
            try:
                if clear:
                    con.execute(
                        f"UPDATE observations SET {col} = NULL WHERE {col} IS NOT NULL"
                    )
                con.execute(
                    f"""
                    UPDATE observations
                    SET {col} = mv.model_value
                    FROM model_values_view mv
                    WHERE observations.rowid = mv.rowid
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

        return len(update_df)

    def update_model_source_spread_values(
        self, df: pd.DataFrame, source: str, clear: bool = True
    ) -> int:
        """
        Update model_forecast_spread or model_analysis_spread for observations using their rowid.

        The DataFrame must contain 'rowid' and 'model_value' columns.

        Args:
            df: DataFrame with 'rowid' and 'model_value' columns.
            source: Either 'forecast' or 'analysis', selects the target column.
            clear: If True, reset all existing entries for the target column to
                NULL before applying the new values. Callers that write the
                result in multiple chunks must pass ``clear=True`` only for the
                first chunk; otherwise each chunk would wipe the previous ones.

        Returns:
            The number of rows updated.
        """
        col = f"model_{source}_spread"

        with self._get_duckdb(read_only=False) as con:
            update_df = df[["rowid", "model_value"]].copy()
            con.register("model_spread_view", update_df)

            con.execute("BEGIN TRANSACTION")
            try:
                if clear:
                    con.execute(
                        f"UPDATE observations SET {col} = NULL WHERE {col} IS NOT NULL"
                    )
                con.execute(
                    f"""
                    UPDATE observations
                    SET {col} = mv.model_value
                    FROM model_spread_view mv
                    WHERE observations.rowid = mv.rowid
                    """
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

        return len(update_df)

    def get_member_values(
        self,
        instrument: str,
        quantity: str,
        source: str = "forecast",
    ) -> pd.DataFrame | None:
        """
        Read per-member model interpolation results from parquet.

        Requires interpolate-model-per-member to have been run first. Returns a
        long-form DataFrame with columns: rowid, instrument, quantity, time, member, value.

        Args:
            source: Either 'forecast' or 'analysis'.
            instrument: Filter to this instrument.
            quantity: Filter to this quantity.

        Returns:
            DataFrame of per-member values, or None if the parquet does not exist
            or no rows match the filters. The returned DataFrame will have columns time,
            model_forecast, x, y, z, latitude, longitude, and value (obs).
        """
        parquet_path = self.paths.data / "validation" / f"model_member_{source}.parquet"
        if not parquet_path.exists():
            return None

        params: list = [str(parquet_path), instrument, quantity]

        with self._get_duckdb(read_only=True) as con:
            result = con.execute(
                # sql
                """
                with pm as (select * from read_parquet(?) where instrument = ? and quantity = ?)
                select
                    pm.member,
                    pm.time,
                    pm.value as "model_forecast",
                    obs.x,
                    obs.y,
                    obs.z,
                    obs.latitude,
                    obs.longitude,
                    obs.value from pm
                left join observations obs on (obs.ROWID = pm.rowid)
                """,
                params,
            ).fetchdf()

        if result.empty:
            return None

        if "time" in result.columns and result["time"].dt.tz is None:
            result["time"] = result["time"].dt.tz_localize("UTC")

        return result

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

    def get_model_interpolated_pairs(
        self,
        start_date: dt.datetime | None = None,
        end_date: dt.datetime | None = None,
    ) -> list[tuple[str, str]]:
        """
        Returns distinct (instrument, quantity) pairs that have model_forecast values.

        Args:
            start_date: If set, only consider observations at or after this time.
            end_date: If set, only consider observations before or at this time.

        Returns:
            List of (instrument, quantity) tuples.
        """
        query = "SELECT DISTINCT instrument, quantity FROM observations WHERE model_forecast IS NOT NULL"
        params = []
        if start_date is not None:
            query += " AND time >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND time <= ?"
            params.append(end_date)

        with self._get_duckdb(read_only=True) as con:
            result = con.execute(query, params).fetchdf()

        return list(result.itertuples(index=False, name=None))

    def get_model_interpolated_for_pair(
        self,
        instrument: str,
        quantity: str,
        qc_flags: list[int] | None = None,
        start_date: dt.datetime | None = None,
        end_date: dt.datetime | None = None,
        metadata_filters: list[tuple[str, str, str]] | None = None,
    ) -> pd.DataFrame | None:
        """
        Retrieves observations for a specific instrument/quantity pair that have model_forecast set.

        Only fetches the columns needed for first-departures analysis, avoiding loading
        large metadata columns like orig_coords or metadata JSON.

        Args:
            instrument: Instrument name to filter by.
            quantity: Quantity name to filter by.
            qc_flags: If set, only return observations with qc_flag in this list.
            start_date: If set, only return observations at or after this time.
            end_date: If set, only return observations before or at this time.
            metadata_filters: If set, a list of (key, op, value) tuples filtering on the
                metadata JSON column, where op is '=' or '!='. Comparisons are textual.
                Observations whose metadata lacks the key are always kept (the filter only
                removes rows that have the key and fail the comparison).

        Returns:
            DataFrame of observations, or None if none exist.
        """
        # sql
        query = """
            SELECT
                instrument, quantity,
                time AT TIME ZONE 'UTC' AS time,
                latitude, longitude,
                value, value_uncertainty,
                model_forecast, model_analysis,
                model_forecast_spread, model_analysis_spread,
                qc_flag
            FROM observations
            WHERE model_forecast IS NOT NULL
              AND instrument = ?
              AND quantity = ?
        """
        params: list = [instrument, quantity]

        if qc_flags is not None:
            placeholders = ", ".join("?" * len(qc_flags))
            query += f" AND qc_flag IN ({placeholders})"
            params.extend(qc_flags)
        if start_date is not None:
            query += " AND time >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND time <= ?"
            params.append(end_date)
        # Metadata JSON filters. Observations whose metadata lacks the key are kept
        # (NULL extraction), so this only drops rows that have the key and fail the match.
        for key, op, value in metadata_filters or []:
            query += (
                " AND (json_extract_string(metadata, ?) IS NULL"
                f"      OR json_extract_string(metadata, ?) {op} ?)"
            )
            params.extend([key, key, value])

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
        cycles_df = cycles_to_dataframe(cycles)
        half_window_td = pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )
        cycles_df["window_start"] = cycles_df["end_time"] - half_window_td
        cycles_df["window_end"] = cycles_df["end_time"] + half_window_td

        instruments = self.cfg.observations.instruments_to_assimilate

        with self._get_duckdb(read_only=True) as con:
            con.register("cycles_view", cycles_df)

            if instruments is not None:
                placeholders = ", ".join("?" * len(instruments))
                result = con.execute(
                    f"""
                    SELECT
                        cw.cycle_index,
                        COUNT(o.time) AS total,
                        COUNT(o.time) FILTER (
                            WHERE o.time >= cw.window_start
                              AND o.time <= cw.window_end
                              AND o.instrument IN ({placeholders})
                        ) AS to_assimilate
                    FROM cycles_view cw
                    LEFT JOIN observations o ON o.time >= cw.start_time AND o.time <= cw.end_time
                    GROUP BY cw.cycle_index
                    ORDER BY cw.cycle_index
                    """,
                    [*instruments],
                ).fetchdf()
            else:
                result = con.execute(
                    """
                    SELECT
                        cw.cycle_index,
                        COUNT(o.time) AS total,
                        COUNT(o.time) FILTER (
                            WHERE o.time >= cw.window_start
                              AND o.time <= cw.window_end
                        ) AS to_assimilate
                    FROM cycles_view cw
                    LEFT JOIN observations o ON o.time >= cw.start_time AND o.time <= cw.end_time
                    GROUP BY cw.cycle_index
                    ORDER BY cw.cycle_index
                    """
                ).fetchdf()

        result["total"] = result["total"].astype(int)
        result["to_assimilate"] = result["to_assimilate"].astype(int)
        return result

    def get_cycle_file_info(self, cycle: CycleInformation) -> pd.DataFrame:
        """
        Returns per-file observation statistics for a cycle's time window (start to end).

        Each row represents one (orig_filename, instrument) combination and includes:
        - total: total observations in the cycle time window
        - assimilated: observations within the assimilation window around cycle end

        Args:
            cycle: The cycle whose start/end times define the time window.
        """

        window_start = cycle.end - pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )
        window_end = cycle.end + pd.Timedelta(
            minutes=self.cfg.assimilation.half_window_length_minutes
        )
        instruments = self.cfg.observations.instruments_to_assimilate

        with self._get_duckdb(read_only=True) as con:
            if instruments is not None:
                placeholders = ", ".join("?" * len(instruments))
                result = con.execute(
                    f"""
                    SELECT
                        orig_filename,
                        instrument,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (
                            WHERE time >= ? AND time <= ?
                              AND instrument IN ({placeholders})
                        ) AS assimilated
                    FROM observations
                    WHERE time >= ? AND time <= ?
                    GROUP BY orig_filename, instrument
                    ORDER BY instrument, orig_filename
                    """,
                    [
                        str(window_start),
                        str(window_end),
                        *instruments,
                        str(cycle.start),
                        str(cycle.end),
                    ],
                ).fetchdf()
            else:
                result = con.execute(
                    """
                    SELECT
                        orig_filename,
                        instrument,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (
                            WHERE time >= ? AND time <= ?
                        ) AS assimilated
                    FROM observations
                    WHERE time >= ? AND time <= ?
                    GROUP BY orig_filename, instrument
                    ORDER BY instrument, orig_filename
                    """,
                    [
                        str(window_start),
                        str(window_end),
                        str(cycle.start),
                        str(cycle.end),
                    ],
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
            factors_df = pd.DataFrame(
                [
                    {
                        "instrument": k.split(".")[0],
                        "quantity": k.split(".")[1],
                        "factor": v,
                    }
                    for k, v in self.cfg.observations.error_inflation_factor.items()
                ]
            )
            observations = observations.merge(
                factors_df, on=["instrument", "quantity"], how="left"
            )
            observations["factor"] = observations["factor"].fillna(1.0)
            observations["value_uncertainty"] *= observations["factor"]
            observations = observations.drop(columns=["factor"])

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
