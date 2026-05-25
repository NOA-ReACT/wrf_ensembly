"""Per-member model interpolation to observation locations and times."""

import glob
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.console import logger
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

        all_dfs: list[pd.DataFrame] = []
        for m in range(n_members):
            logger.info(f"Interpolating {source} member {m + 1}/{n_members}")
            ds_m = ds_ensemble.isel(member=m)
            df_m = self._base.interpolate_all(needed_combos, ds_m, source=source)
            if df_m.empty:
                continue
            df_m["member"] = np.int32(m)
            all_dfs.append(df_m)

        if not all_dfs:
            logger.warning(f"No interpolation results for {source}")
            return 0

        combined = pd.concat(all_dfs, ignore_index=True)
        enriched = self._enrich_with_metadata(combined)

        output_path = output_dir / f"model_member_{source}.parquet"
        enriched.to_parquet(output_path, index=False)

        n_rows = len(enriched)
        n_obs = n_rows // n_members if n_members else n_rows
        logger.info(
            f"Wrote {n_rows} rows ({n_members} members × ~{n_obs} obs) to {output_path}"
        )
        return n_rows

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
