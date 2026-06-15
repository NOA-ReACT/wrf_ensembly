"""Per-member model interpolation to observation locations and times."""

import glob
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from wrf_ensembly.console import console, logger
from wrf_ensembly.experiment.experiment import Experiment
from wrf_ensembly.validation.model_interpolation import ModelInterpolation


class PerMemberModelInterpolation:
    """Interpolates each ensemble member's model output to observation locations.

    Requires per-member ensemble files produced when ``postprocess.keep_per_member``
    is true. Results are written to
    ``data/validation/model_member_{forecast,analysis}.parquet`` in long form:
    one row per (observation, member).
    """

    exp: Experiment
    _base: ModelInterpolation

    def __init__(self, experiment: Experiment):
        self.exp = experiment
        self._base = ModelInterpolation(experiment)

    def run(self) -> int:
        """Run per-member interpolation for forecast and analysis sources.

        Returns:
            Total number of rows written across all sources, or 0 on failure.
        """
        needed_combos = self._base.determine_needed_combos()
        if not needed_combos:
            logger.info("No valid instrument/quantity combos found, cannot proceed!")
            return 0

        needed_vars: set[str] = {"z", "air_pressure", "geopotential_height"}
        for combo in needed_combos:
            needed_vars.update(combo["wrf_vars"])

        output_dir = self.exp.paths.data / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        for source in ("forecast", "analysis"):
            n = self._run_source(source, needed_combos, needed_vars, output_dir)
            total_rows += n

        return total_rows

    def _run_source(
        self,
        source: str,
        needed_combos: list[dict],
        needed_vars: set[str],
        output_dir: Path,
    ) -> int:
        if source == "forecast":
            glob_pattern = (
                f"{self.exp.paths.data_forecasts}/cycle_**/{source}_ensemble_cycle_*.nc"
            )
        else:
            glob_pattern = (
                f"{self.exp.paths.data_analysis}/cycle_**/{source}_ensemble_cycle_*.nc"
            )

        matched = glob.glob(glob_pattern)
        if not matched:
            logger.warning(
                f"No {source} ensemble files matched {glob_pattern}. "
                "Is postprocess.keep_per_member = true in config?"
            )
            return 0

        logger.info(f"Opening {len(matched)} {source} ensemble file(s)")
        # Forecast files overlap in time when a forecast extension is configured;
        # resolve the overlap into a unique time axis (preferring the extended or
        # analysis-driven forecast per config). Analysis files never overlap.
        if source == "forecast" and self.exp.cfg.time_control.forecast_extension > 0:
            ds_ensemble = self._base._open_forecast_resolved(sorted(matched))
        else:
            ds_ensemble = xr.open_mfdataset(
                glob_pattern,
                combine="by_coords",
                chunks={"time": 1},
                coords="minimal",
            )

        available = [v for v in needed_vars if v in ds_ensemble.data_vars]
        missing = [v for v in needed_vars if v not in ds_ensemble.data_vars]
        if missing:
            logger.warning(
                f"Variables not found in {source} ensemble files (may be filtered by "
                f"variables_to_keep_ensemble): {missing}"
            )
        ds_ensemble = ds_ensemble[available]

        if "member" not in ds_ensemble.dims:
            logger.error(
                f"Ensemble file for '{source}' has no 'member' dimension. "
                "Was it produced by the postprocess pipeline?"
            )
            return 0

        # Check for duplicate time coordinates (mirrors _open_files validation)
        t_vals = ds_ensemble["t"].values
        unique, counts = np.unique(t_vals, return_counts=True)
        duplicates = unique[counts > 1]
        if len(duplicates) > 0:
            logger.error(
                f"Duplicate time values in {source} ensemble files: {duplicates}"
            )
            return 0

        n_members = int(ds_ensemble.dims["member"])
        cfg_members = self.exp.cfg.assimilation.n_members
        if n_members != cfg_members:
            logger.warning(
                f"Ensemble file has {n_members} members but config says {cfg_members}"
            )

        output_path = output_dir / f"model_member_{source}.parquet"
        writer: pq.ParquetWriter | None = None
        total_rows = 0

        member_progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[extra]}"),
            console=console,
        )
        member_task = member_progress.add_task(
            "Members", total=n_members, extra=source
        )

        try:
            with member_progress:
                for m in range(n_members):
                    logger.info(
                        f"Interpolating {source} member {m + 1}/{n_members}"
                    )
                    ds_m = ds_ensemble.isel(member=m)
                    member_idx = np.int32(m)

                    def _flush_member(batch_df: pd.DataFrame) -> None:
                        nonlocal writer, total_rows
                        if batch_df.empty:
                            return
                        batch_df["member"] = member_idx
                        enriched = self._enrich_with_metadata(batch_df)
                        table = pa.Table.from_pandas(enriched, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                str(output_path), table.schema
                            )
                        writer.write_table(table)
                        total_rows += len(enriched)

                    self._base.interpolate_all(
                        needed_combos,
                        ds_m,
                        source=source,
                        result_callback=_flush_member,
                        progress=member_progress,
                    )
                    member_progress.advance(member_task)
        finally:
            if writer is not None:
                writer.close()

        if total_rows == 0:
            logger.warning(f"No interpolation results for {source}")
            return 0

        n_obs = total_rows // n_members if n_members else total_rows
        logger.info(
            f"Wrote {total_rows} rows ({n_members} members × ~{n_obs} obs) to {output_path}"
        )
        return total_rows

    def _enrich_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join per-member results with the obs DB to add instrument, quantity, time."""
        with duckdb.connect(str(self.exp.paths.obs_db), read_only=True) as con:
            con.execute("SET TimeZone='UTC'")
            con.register("member_results", df)
            enriched = con.execute("""
                SELECT
                    r.rowid,
                    o.instrument,
                    o.quantity,
                    o.time AT TIME ZONE 'UTC' AS time,
                    r.member,
                    r.model_value AS value
                FROM member_results r
                JOIN observations o ON o.rowid = r.rowid
            """).fetchdf()

        if enriched["time"].dt.tz is None:
            enriched["time"] = enriched["time"].dt.tz_localize("UTC")

        return enriched
