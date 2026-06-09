"""Model interpolation to observation locations and times."""

import contextlib
import glob
import os
import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
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
from wrf_ensembly.observations import QUANTITY_REGISTRY, OperatorSpec

# Maximum number of result rows to accumulate before flushing via result_callback.
_FLUSH_ROWS = 2_000_000


class ModelInterpolation:
    """Interpolates model outputs to observation locations and times.

    This class handles the validation workflow of comparing model forecasts
    to observations by interpolating model data spatially and temporally to
    match observation points.

    Supports both simple direct-field lookups (via QuantitySpec.model_equivalent)
    and complex observation operators (via QuantitySpec.operator) that combine
    multiple model fields and observation metadata.
    """

    exp: Experiment

    def __init__(self, experiment: Experiment):
        """Initialize with an experiment.

        Args:
            experiment: The experiment to perform validation on
        """
        self.exp = experiment

    def run(self) -> int:
        """Run the model interpolation.

        Interpolates model forecasts and analyses to observation locations and
        stores the results in the experiment's DuckDB observations table as
        model_forecast and model_analysis columns.

        Returns:
            Total number of observations updated across both sources, or 0 on failure
        """

        needed_combos = self.determine_needed_combos()
        if not needed_combos:
            logger.info("No valid instrument/quantity combos found, cannot proceed!")
            return 0

        # Collect all WRF variable names we need (target vars + vertical coords)
        needed_vars: set[str] = {"z", "air_pressure", "geopotential_height"}
        for combo in needed_combos:
            needed_vars.update(combo["wrf_vars"])

        n_updated = 0

        # Vertical coordinate variables live in the mean files and must not
        # be taken from the sd files (which contain ensemble spread, not actual
        # coordinate values).
        vert_coord_vars = ["air_pressure", "geopotential_height", "z"]

        # TODO: Feeding per-field spread through a non-identity operator does not
        # yield the spread of the operator output. Proper handling requires either
        # linearizing the operator (Jacobian-based SD propagation) or computing SD
        # across per-member operator evaluations. Until then, skip spread for
        # operator-based combos and only compute spread where model_equivalent is
        # a direct field lookup.
        spread_combos = [c for c in needed_combos if not c["has_operator"]]
        skipped_for_spread = [c for c in needed_combos if c["has_operator"]]
        for c in skipped_for_spread:
            logger.warning(
                f"Skipping spread for {c['instrument']}/{c['quantity']}: "
                "operator-based quantities cannot be propagated by interpolating spread directly"
            )

        for source in ("forecast", "analysis"):
            logger.info(f"Interpolating mean with {source} files")
            ds_mean = self._open_files(source, list(needed_vars), file_type="mean")
            if ds_mean is None:
                logger.warning(
                    f"No {source} mean files found, skipping {source} interpolation"
                )
                continue

            def _flush_mean(batch: pd.DataFrame) -> None:
                nonlocal n_updated
                n_updated += self._save_output(batch, source)

            self.interpolate_all(
                needed_combos, ds_mean, source, result_callback=_flush_mean
            )

            if not spread_combos:
                continue

            logger.info(f"Interpolating spread with {source} files")
            sd_vars = [v for v in needed_vars if v not in vert_coord_vars]
            ds_sd = self._open_files(source, sd_vars, file_type="sd")
            if ds_sd is None:
                logger.warning(
                    f"No {source} sd files found, skipping {source} spread interpolation"
                )
                continue

            # Inject actual vertical coordinates from the mean dataset
            for v in vert_coord_vars:
                if v in ds_mean.data_vars:
                    ds_sd[v] = ds_mean[v]

            def _flush_spread(batch: pd.DataFrame) -> None:
                nonlocal n_updated
                n_updated += self._save_spread_output(batch, source)

            self.interpolate_all(
                spread_combos, ds_sd, source, result_callback=_flush_spread
            )

        return n_updated

    def _instrument_where_clause(self, prefix: str = "WHERE") -> tuple[str, list[str]]:
        """Build a parameterized SQL WHERE/AND clause for instrument filtering.

        Args:
            prefix: SQL keyword to use ('WHERE' or 'AND')

        Returns:
            A (clause, params) tuple. The clause contains '?' placeholders and
            the params list holds the instrument names; both are empty when no
            filter is configured.
        """
        if not self.exp.cfg.validation.instruments:
            return "", []
        instruments = list(self.exp.cfg.validation.instruments)
        placeholders = ", ".join("?" for _ in instruments)
        return f"{prefix} instrument IN ({placeholders})", instruments

    def determine_needed_combos(self) -> list[dict]:
        """Determine which instrument/quantity combinations need interpolation.

        Queries DuckDB for distinct combinations and resolves their WRF
        variable mappings and vertical coordinate types. Handles both simple
        model_equivalent lookups and operator-based quantities.

        Returns:
            List of dicts with keys: instrument, quantity, wrf_vars, z_type,
            has_operator
        """
        where_clause, where_params = self._instrument_where_clause()

        with self.exp.obs._get_duckdb(read_only=True) as con:
            combos = con.execute(
                f"SELECT DISTINCT instrument, quantity, z_type FROM observations {where_clause}",
                where_params,
            ).fetchall()

        needed_combos = []
        for instrument, quantity, z_type in combos:
            quantity_info = QUANTITY_REGISTRY.get(quantity)
            if quantity_info is None or not quantity_info.has_model_mapping:
                logger.warning(
                    f"Quantity {quantity} not found in observation mappings or has no model mapping, skipping"
                )
                continue

            needed_combos.append(
                {
                    "instrument": instrument,
                    "quantity": quantity,
                    "wrf_vars": quantity_info.required_wrf_vars,
                    "z_type": z_type,
                    "has_operator": quantity_info.operator is not None,
                }
            )

        logger.info(f"Found {len(needed_combos)} instrument/quantity combinations")
        for combo in needed_combos:
            op_tag = " [operator]" if combo["has_operator"] else ""
            logger.info(
                f"  {combo['instrument']}/{combo['quantity']} -> "
                f"{combo['wrf_vars']} (z_type={combo['z_type']}){op_tag}"
            )

        return needed_combos

    def _open_files(
        self, source: str, needed_vars: list[str], file_type: str = "mean"
    ) -> xr.Dataset | None:
        """Open forecast or analysis files as a single lazy xarray Dataset.

        Args:
            source: Either 'forecast' or 'analysis'
            needed_vars: List of WRF variable names to load
            file_type: Either 'mean' or 'sd', selects which ensemble statistic
                files to open

        Returns:
            Lazy xarray Dataset with only the needed variables, or None if no
            files are found

        Raises:
            ValueError: If duplicate coordinates are found
        """
        if source == "forecast":
            glob_pattern = f"{self.exp.paths.data_forecasts}/cycle_**/{source}_{file_type}_cycle_*.nc"
        else:
            glob_pattern = f"{self.exp.paths.data_analysis}/cycle_**/{source}_{file_type}_cycle_*.nc"

        logger.debug(f"Globbing {source}/{file_type} files: {glob_pattern}")
        if not glob.glob(glob_pattern):
            logger.debug(f"No files matched for {source}/{file_type}")
            return None

        ds = xr.open_mfdataset(
            glob_pattern,
            combine="by_coords",
            chunks={"time": 1},
            coords="minimal",
        )

        available = [v for v in needed_vars if v in ds.data_vars]
        missing = [v for v in needed_vars if v not in ds.data_vars]
        if missing:
            logger.warning(
                f"Variables not found in {source}/{file_type} files: {missing}"
            )
        ds = ds[available]

        # Check for duplicate coordinates
        for coord_name in ["t", "x", "y"]:
            if coord_name not in ds.coords:
                continue
            coord_vals = ds.coords[coord_name].values
            unique, counts = np.unique(coord_vals, return_counts=True)
            duplicates = unique[counts > 1]

            if len(duplicates) > 0:
                logger.error(
                    f"Duplicate values found in {source}/{file_type} coordinate '{coord_name}':"
                )
                for dup in duplicates:
                    indices = np.where(coord_vals == dup)[0]
                    logger.error(f"  Value: {dup}")
                    logger.error(
                        f"  Appears {len(indices)} times at indices: {indices}"
                    )

                raise ValueError(
                    f"Duplicate values found in {source}/{file_type} coordinate "
                    f"'{coord_name}': {duplicates}"
                )

        return ds

    def interpolate_all(
        self,
        needed_combos: list[dict],
        ds: xr.Dataset,
        source: str = "forecast",
        result_callback: Callable[[pd.DataFrame], None] | None = None,
        progress: Progress | None = None,
    ) -> pd.DataFrame:
        """Interpolate model forecasts to all observation combos.

        Operates in three phases:
        1. Gather: fetch all observation data from DuckDB and compute time
           bracket assignments for every combo. All DuckDB access is
           concentrated here, in the main thread.
        2. Execute: iterate over unique brackets (across all combos). A
           dedicated loader thread owns all access to the lazy `ds` and
           materializes each bracket's 1-2 timesteps via
           `compute(scheduler="synchronous")`. A bounded queue hands the
           materialized datasets off to a ThreadPoolExecutor that runs
           scipy interpolation in parallel (numpy/scipy release the GIL).
           Loading and processing overlap, and the main thread stays
           reactive enough to drain completions and refresh progress.
        3. Collect: concatenate all results.

        Args:
            needed_combos: List of combo dicts from determine_needed_combos
            ds: Lazy Dataset (forecast or analysis, mean or sd)
            source: Either 'forecast' or 'analysis'. For analysis, observation
                times are snapped to the nearest available time in the dataset
                since analysis files have only one timestep per cycle.
            result_callback: If provided, results are flushed through this
                callback in batches during processing to bound memory usage.
                When set, the returned DataFrame is empty.
            progress: If provided, the bracket progress task is added to this
                existing Progress instance instead of creating a new one.
                Useful for nesting under an outer progress bar.

        Returns:
            DataFrame with all observations and their model_value, or an
            empty DataFrame if result_callback was provided (results are
            delivered through the callback instead).
        """
        instrument_filter, instrument_params = self._instrument_where_clause(
            prefix="AND"
        )
        all_results = []
        accumulated_rows = 0

        ds_times = pd.DatetimeIndex(ds["t"].values)
        ds_times_values = ds_times.values
        n_times = len(ds_times)
        ds_t_min = ds_times_values[0]
        ds_t_max = ds_times_values[-1]
        # Observations farther than this from the dataset time range are treated
        # as invalid and their model values are set to NaN. Inside the range
        # (or within max_time_gap of an edge) we clamp/snap and interpolate.
        max_time_gap = np.timedelta64(1, "h")

        # --- Phase 1: Gather observation data and bracket assignments ---
        combo_work_list: list[dict] = []
        # bracket_key -> [(combo_idx, obs_indices_into_that_combo)]
        bracket_work: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

        for combo in needed_combos:
            instrument = combo["instrument"]
            quantity = combo["quantity"]
            quantity_info = QUANTITY_REGISTRY[quantity]
            tag = f"[{instrument}/{quantity}]"

            logger.info(
                f"{tag} Fetching (vars={combo['wrf_vars']}, z_type={combo['z_type']})"
            )

            metadata_sql = ""
            if (
                quantity_info.operator is not None
                and quantity_info.operator.required_metadata
            ):
                meta_extracts = ", ".join(
                    f"json_extract(metadata, '$.{key}')::DOUBLE AS meta_{key}"
                    for key in quantity_info.operator.required_metadata
                )
                metadata_sql = f", {meta_extracts}"

            with self.exp.obs._get_duckdb(read_only=True) as con:
                arrays = con.execute(
                    f"""
                    SELECT rowid, time, x, y, z, latitude, longitude, value
                        {metadata_sql}
                    FROM observations
                    WHERE instrument = ? AND quantity = ?
                    {instrument_filter}
                    """,
                    [instrument, quantity, *instrument_params],
                ).fetchnumpy()

            n_obs = len(arrays["x"])
            if n_obs == 0:
                logger.warning(f"{tag} No observations, skipping")
                continue

            logger.info(f"{tag} {n_obs} observations")

            times = pd.DatetimeIndex(arrays["time"]).tz_localize(None)
            orig_times = times.values

            # Compute effective observation times with snapping/clamping, and
            # flag obs that are too far outside the dataset time range as invalid.
            if source == "analysis":
                nearest_idx = ds_times.get_indexer(times, method="nearest")
                eff_times = ds_times_values[nearest_idx]
                invalid_time = np.abs(orig_times - eff_times) > max_time_gap
            else:
                eff_times = np.clip(orig_times, ds_t_min, ds_t_max)
                invalid_time = (orig_times < ds_t_min - max_time_gap) | (
                    orig_times > ds_t_max + max_time_gap
                )

            n_invalid_time = int(invalid_time.sum())
            if n_invalid_time > 0:
                logger.warning(
                    f"{tag} {n_invalid_time}/{n_obs} obs are >1h outside dataset time range; their model values will be NaN"
                )

            # Find bracketing timestep indices for each observation.
            # searchsorted('right') gives the index of the first dataset time
            # strictly greater than the obs time, so the bracket is
            # (right-1, right), clamped to valid range.
            right_idx = np.searchsorted(ds_times_values, eff_times, side="right")
            right_idx = np.clip(right_idx, 0, n_times - 1)
            left_idx = np.clip(right_idx - 1, 0, n_times - 1)

            # Group observations by their time bracket using a single scalar
            # key per (left, right) pair
            bracket_keys = left_idx * n_times + right_idx
            unique_keys = np.unique(bracket_keys)

            logger.info(f"{tag} {len(unique_keys)} time brackets")

            combo_idx = len(combo_work_list)
            combo_work_list.append(
                {
                    "combo": combo,
                    "arrays": arrays,
                    "eff_times": eff_times,
                    "invalid_time": invalid_time,
                    "quantity_info": quantity_info,
                }
            )

            for bkey in unique_keys:
                obs_idx = np.where(bracket_keys == bkey)[0]
                bracket_work[bkey].append((combo_idx, obs_idx))

        if not combo_work_list:
            return pd.DataFrame(columns=["rowid", "model_value"])

        # --- Phase 2: Producer/consumer pipeline ---
        # A dedicated loader thread owns all access to the lazy `ds` and
        # materializes 1-2 timesteps per bracket (scheduler="synchronous"
        # so dask doesn't spawn its own threads). Loaded brackets are
        # handed off on a bounded queue. The main thread submits them to
        # a ThreadPoolExecutor that runs scipy interpolation in parallel
        # (numpy/scipy release the GIL), and stays reactive enough to
        # drain completions and refresh progress while loading happens
        # concurrently.
        sorted_brackets = sorted(bracket_work.keys())
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        n_brackets = len(sorted_brackets)
        logger.info(
            f"Processing {n_brackets} unique brackets with up to {max_workers} workers"
        )

        # Bounded handoff: blocks the loader once we have enough materialized
        # timesteps in flight, keeping memory bounded.
        load_queue: queue.Queue = queue.Queue(maxsize=max_workers + 2)
        loader_error: list[BaseException] = []
        stop_loader = threading.Event()

        def _loader() -> None:
            try:
                for bkey in sorted_brackets:
                    if stop_loader.is_set():
                        break
                    bl = int(bkey // n_times)
                    br = int(bkey % n_times)
                    logger.debug(f"  Loading bracket t[{bl}]..t[{br}]")
                    ds_sub = ds.isel(t=slice(bl, br + 1)).compute(
                        scheduler="synchronous"
                    )
                    load_queue.put((bkey, ds_sub))
            except BaseException as exc:
                loader_error.append(exc)
            finally:
                load_queue.put(None)

        pending: dict = {}

        owns_progress = progress is None
        if owns_progress:
            progress = Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("{task.fields[extra]}"),
                console=console,
            )
        task_done = progress.add_task("Brackets", total=n_brackets, extra="")

        def _update_progress() -> None:
            progress.update(task_done, extra=f"[dim]{len(pending)} in queue[/dim]")

        progress_ctx = progress if owns_progress else contextlib.nullcontext()
        with progress_ctx, ThreadPoolExecutor(max_workers=max_workers) as executor:
            loader_thread = threading.Thread(
                target=_loader, name="bracket-loader", daemon=True
            )
            loader_thread.start()
            loader_done = False

            try:
                while not loader_done or pending:
                    # Submit every loaded bracket that's ready right now.
                    while not loader_done:
                        try:
                            item = load_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            loader_done = True
                            break
                        bkey, ds_sub = item
                        future = executor.submit(
                            self._process_bracket,
                            ds_sub,
                            bracket_work[bkey],
                            combo_work_list,
                            bkey,
                            n_times,
                        )
                        pending[future] = bkey
                        _update_progress()

                    if not pending:
                        # Loader still has work to deliver — block briefly
                        # on the queue rather than busy-waiting.
                        if not loader_done:
                            try:
                                item = load_queue.get(timeout=0.1)
                            except queue.Empty:
                                continue
                            if item is None:
                                loader_done = True
                                continue
                            bkey, ds_sub = item
                            future = executor.submit(
                                self._process_bracket,
                                ds_sub,
                                bracket_work[bkey],
                                combo_work_list,
                                bkey,
                                n_times,
                            )
                            pending[future] = bkey
                            _update_progress()
                        continue

                    # Wait briefly for processing to complete so we can
                    # also re-check the loader queue for new work.
                    done, _ = wait(
                        pending.keys(), return_when=FIRST_COMPLETED, timeout=0.1
                    )
                    for future in done:
                        pending.pop(future)
                        bracket_results = future.result()
                        all_results.extend(bracket_results)
                        accumulated_rows += sum(len(df) for df in bracket_results)
                        progress.advance(task_done)
                    if done:
                        _update_progress()

                    if result_callback and accumulated_rows >= _FLUSH_ROWS:
                        logger.debug(
                            f"Flushing {accumulated_rows} accumulated result rows"
                        )
                        result_callback(pd.concat(all_results, ignore_index=True))
                        all_results.clear()
                        accumulated_rows = 0
            except BaseException:
                stop_loader.set()
                # Drain the queue so the loader isn't stuck on .put().
                try:
                    while True:
                        load_queue.get_nowait()
                except queue.Empty:
                    pass
                raise
            finally:
                loader_thread.join()

            if loader_error:
                raise loader_error[0]

        # --- Phase 3: Collect ---
        if not all_results:
            return pd.DataFrame(columns=["rowid", "model_value"])
        final = pd.concat(all_results, ignore_index=True)
        if result_callback:
            result_callback(final)
            return pd.DataFrame(columns=["rowid", "model_value"])
        return final

    def _process_bracket(
        self,
        ds_sub: xr.Dataset,
        bracket_combos: list[tuple[int, np.ndarray]],
        combo_work_list: list[dict],
        bkey: int,
        n_times: int,
    ) -> list[pd.DataFrame]:
        """Process all combos for a single time bracket.

        Called from worker threads. Operates only on the in-memory
        (numpy-backed) ds_sub — no dask or DuckDB access here.

        Args:
            ds_sub: In-memory Dataset with 1-2 timesteps already materialized
            bracket_combos: List of (combo_idx, obs_indices) pairs sharing this bracket
            combo_work_list: Pre-fetched combo data from Phase 1
            bkey: Bracket key (encoded as left_idx * n_times + right_idx)
            n_times: Total number of timesteps in the dataset

        Returns:
            List of DataFrames, one per (combo, bracket) pair
        """
        bl = int(bkey // n_times)
        br = int(bkey % n_times)
        logger.debug(f"  Bracket t[{bl}]..t[{br}]: {len(bracket_combos)} combos")

        results = []
        for combo_idx, obs_idx in bracket_combos:
            cw = combo_work_list[combo_idx]
            combo = cw["combo"]
            arrays = cw["arrays"]
            eff_times = cw["eff_times"]
            invalid_time = cw["invalid_time"]
            quantity_info = cw["quantity_info"]
            z_type = combo["z_type"]

            # Slice only what we need for this bracket/combo instead of copying
            # every column in `arrays`.
            obs_t = eff_times[obs_idx]
            obs_x = arrays["x"][obs_idx]
            obs_y = arrays["y"][obs_idx]
            obs_z = arrays["z"][obs_idx]
            rowid = arrays["rowid"][obs_idx]

            if quantity_info.operator is not None:
                metadata = {
                    key: arrays[f"meta_{key}"][obs_idx]
                    for key in quantity_info.operator.required_metadata
                    if f"meta_{key}" in arrays
                }
                model_vals = self._apply_operator(
                    ds_sub,
                    quantity_info.operator,
                    z_type,
                    obs_t,
                    obs_x,
                    obs_y,
                    obs_z,
                    metadata,
                )
            else:
                model_vals = self._interpolate_simple(
                    ds_sub,
                    quantity_info.model_equivalent,
                    z_type,
                    obs_t,
                    obs_x,
                    obs_y,
                    obs_z,
                )

            # NaN out obs that fall outside the dataset time range.
            invalid_subset = invalid_time[obs_idx]
            if invalid_subset.any():
                model_vals = np.where(invalid_subset, np.nan, model_vals)

            results.append(
                pd.DataFrame(
                    {
                        "rowid": rowid,
                        "model_value": model_vals,
                    }
                )
            )

        return results

    def _interpolate_simple(
        self,
        ds: xr.Dataset,
        wrf_var: str,
        z_type: str,
        obs_t: np.ndarray,
        obs_x: np.ndarray,
        obs_y: np.ndarray,
        obs_z: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a single model field to observation locations.

        This is the original interpolation path for quantities with a direct
        model_equivalent mapping.

        Args:
            ds: In-memory Dataset
            wrf_var: WRF variable name
            z_type: Vertical coordinate type ("columnar", "surface", "height", "pressure")
            obs_t: Observation times (datetime64)
            obs_x: Observation x coordinates
            obs_y: Observation y coordinates
            obs_z: Vertical coordinate values for observations

        Returns:
            (n_obs,) array of interpolated model values
        """
        n_obs = len(obs_z)

        if wrf_var not in ds.data_vars:
            logger.warning(f"Variable {wrf_var} not in forecast data")
            return np.full(n_obs, np.nan)

        if z_type in ("columnar", "surface"):
            return self._hor_interp(ds, [wrf_var], obs_t, obs_x, obs_y)[wrf_var]
        elif z_type == "height":
            return self._interp_vertical(
                ds,
                wrf_var,
                "geopotential_height",
                obs_t,
                obs_x,
                obs_y,
                obs_z,
            )
        elif z_type == "pressure":
            return self._interp_vertical(
                ds,
                wrf_var,
                "air_pressure",
                obs_t,
                obs_x,
                obs_y,
                obs_z,
                flip=True,
            )
        else:
            logger.warning(f"Unknown z_type '{z_type}'")
            return np.full(n_obs, np.nan)

    def _apply_operator(
        self,
        ds: xr.Dataset,
        operator: OperatorSpec,
        z_type: str,
        obs_t: np.ndarray,
        obs_x: np.ndarray,
        obs_y: np.ndarray,
        obs_z: np.ndarray,
        metadata: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Interpolate all required model fields and apply the observation operator.

        For each field in the operator's required_model_fields:
        - dims=2: horizontal/temporal interpolation to a scalar per obs
        - dims=3: horizontal/temporal + vertical interpolation to a scalar per obs

        Note on NaN handling:
            operator.func may receive NaN-valued entries in `model_fields`
            (e.g. when z_type is unknown for 3D fields, the corresponding
            arrays are NaN-filled so other 2D fields can still be passed
            through). Metadata arrays may also contain NaN if the converter
            did not populate every observation. Operator implementations
            must tolerate per-observation NaN input and propagate NaN to
            their output for those rows rather than raising.

        Args:
            ds: In-memory Dataset
            operator: The OperatorSpec to apply
            z_type: Vertical coordinate type
            obs_t: Observation times (datetime64)
            obs_x: Observation x coordinates
            obs_y: Observation y coordinates
            obs_z: Vertical coordinate values for observations
            metadata: Dict mapping required metadata key -> array of values
                already sliced to this bracket's observations

        Returns:
            (n_obs,) array of operator-computed model-equivalent values
        """
        n_obs = len(obs_x)

        # Check that all required model fields are available
        missing_fields = [
            f.name for f in operator.required_model_fields if f.name not in ds.data_vars
        ]
        if missing_fields:
            logger.warning(
                f"Model fields {missing_fields} not in forecast data, operator cannot run"
            )
            return np.full(n_obs, np.nan)

        fields_2d = [f for f in operator.required_model_fields if f.dims == 2]
        fields_3d = [f for f in operator.required_model_fields if f.dims == 3]

        model_fields: dict[str, np.ndarray] = {}

        # Batch all 2D fields in a single _hor_interp call
        if fields_2d:
            names_2d = [f.name for f in fields_2d]
            interped = self._hor_interp(ds, names_2d, obs_t, obs_x, obs_y)
            for name in names_2d:
                model_fields[name] = interped[name]

        # Batch all 3D fields + vertical coord, then do per-field vertical interp
        if fields_3d:
            flip = False
            if z_type == "height":
                vert_coord = "geopotential_height"
                flip = False
            elif z_type == "pressure":
                vert_coord = "air_pressure"
                flip = True
            else:
                logger.warning(f"Unknown z_type '{z_type}' for 3D operator fields")
                for f in fields_3d:
                    model_fields[f.name] = np.full(n_obs, np.nan)
                vert_coord = None

            if vert_coord is not None:
                names_3d = [f.name for f in fields_3d]
                all_3d_vars = list(set(names_3d + [vert_coord]))
                interped = self._hor_interp(ds, all_3d_vars, obs_t, obs_x, obs_y)
                coord_profiles = interped[vert_coord]
                if flip:
                    coord_profiles = coord_profiles[:, ::-1]

                for name in names_3d:
                    profiles = interped[name]
                    if flip:
                        profiles = profiles[:, ::-1]
                    model_fields[name] = self._vertical_interp_1d(
                        profiles, coord_profiles, obs_z
                    )

        # Validate metadata presence and log NaN fraction
        for key in operator.required_metadata:
            if key not in metadata:
                logger.error(
                    f"Required metadata '{key}' not found in observation data. Check that the converter populates this field."
                )
                return np.full(n_obs, np.nan)
            vals = metadata[key]
            n_invalid = (
                np.sum(np.isnan(vals)) if np.issubdtype(vals.dtype, np.floating) else 0
            )
            if n_invalid > 0:
                logger.warning(
                    f"Metadata '{key}' has {n_invalid}/{n_obs} NaN values ({100 * n_invalid / n_obs:.1f}%)"
                )

        return operator.func(model_fields, metadata)

    def _interp_vertical(
        self,
        ds: xr.Dataset,
        wrf_var: str,
        vertical_coord_var: str,
        obs_t: np.ndarray,
        obs_x: np.ndarray,
        obs_y: np.ndarray,
        obs_z: np.ndarray,
        flip: bool = False,
    ) -> np.ndarray:
        """Interpolate a 3D variable vertically to observation altitudes/pressures.

        Performs horizontal/temporal interpolation first to get vertical
        profiles at each observation location, then does 1D vertical
        interpolation per observation.

        Args:
            ds: In-memory Dataset
            wrf_var: Name of the target variable
            vertical_coord_var: Name of the vertical coordinate variable
                (e.g., 'geopotential_height' for height, 'air_pressure' for
                pressure)
            obs_t: Observation times (datetime64)
            obs_x: Observation x coordinates
            obs_y: Observation y coordinates
            obs_z: Target vertical coordinate values (height or pressure)
            flip: If True, reverse the vertical axis before interpolation
                (needed for pressure, which decreases with altitude)

        Returns:
            Array of interpolated values, one per observation
        """
        interped = self._hor_interp(
            ds, [wrf_var, vertical_coord_var], obs_t, obs_x, obs_y
        )

        profiles = interped[wrf_var]  # (n_obs, n_levels)
        coord_profiles = interped[vertical_coord_var]  # (n_obs, n_levels)

        if flip:
            profiles = profiles[:, ::-1]
            coord_profiles = coord_profiles[:, ::-1]

        return self._vertical_interp_1d(profiles, coord_profiles, obs_z)

    @staticmethod
    def _hor_interp(
        ds: xr.Dataset,
        var_names: list[str],
        obs_t: np.ndarray,
        obs_x: np.ndarray,
        obs_y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Trilinear interpolation in (t, x, y) using pure numpy.

        Replaces xarray.interp() + scipy.RegularGridInterpolator for the
        horizontal/temporal dimensions. All heavy work is pure numpy (C-level
        operations that release the GIL), enabling true thread parallelism.

        For 2D variables (t, x, y) returns shape (n_obs,).
        For 3D variables (t, x, y, vert) returns shape (n_obs, n_levels) so
        the caller can do vertical interpolation afterwards.

        Args:
            ds: In-memory (numpy-backed) Dataset with dims t, x, y [, vert]
            var_names: Variables to interpolate
            obs_t: Observation times (datetime64, n_obs)
            obs_x: Observation x coordinates (n_obs,)
            obs_y: Observation y coordinates (n_obs,)

        Returns:
            Dict mapping var_name -> interpolated numpy array
        """
        t_c = ds["t"].values.astype("datetime64[ns]").astype(np.float64)
        x_c = ds["x"].values.astype(np.float64)
        y_c = ds["y"].values.astype(np.float64)
        obs_t_f = obs_t.astype("datetime64[ns]").astype(np.float64)

        def _bracket_weight(coords: np.ndarray, vals: np.ndarray):
            """Return (left_indices, weights) for linear interpolation.

            The single-coordinate branch returns `(0, 0)` for every point; this
            is paired with the `data = concatenate([data, data], axis=...)`
            padding below so that the unrolled `for dt/dx/dy in range(2)` loop
            sees `data[0]` and `data[1]` (both identical). Weights `(1-0)*1
            + 0*1 = 1` land entirely on the duplicated slice, giving the same
            result as "no interpolation" along that axis.
            """
            if len(coords) == 1:
                return np.zeros(len(vals), dtype=np.intp), np.zeros(len(vals))
            idx = np.searchsorted(coords, vals, side="right") - 1
            idx = np.clip(idx, 0, len(coords) - 2)
            dc = coords[idx + 1] - coords[idx]
            w = np.where(dc == 0, 0.0, (vals - coords[idx]) / dc)
            return idx, np.clip(w, 0.0, 1.0)

        it, wt = _bracket_weight(t_c, obs_t_f)
        ix, wx = _bracket_weight(x_c, obs_x.astype(np.float64))
        iy, wy = _bracket_weight(y_c, obs_y.astype(np.float64))

        # Pre-compute both weights for each dimension
        awt = (1.0 - wt, wt)
        awx = (1.0 - wx, wx)
        awy = (1.0 - wy, wy)

        result: dict[str, np.ndarray] = {}
        for name in var_names:
            var = ds[name]
            vert_dims = [d for d in var.dims if d not in ("t", "x", "y")]
            if len(vert_dims) > 1:
                raise ValueError(
                    f"Variable {name!r} has multiple non-(t,x,y) dims {vert_dims}; "
                    "only a single vertical dim is supported"
                )

            if vert_dims:
                z_dim = vert_dims[0]
                data = var.transpose(
                    "t", "x", "y", z_dim
                ).values  # (n_t, n_x, n_y, n_z)
                # Pad to 2 along each axis so idx+1 is always valid
                if data.shape[0] == 1:
                    data = np.concatenate([data, data], axis=0)
                if data.shape[1] == 1:
                    data = np.concatenate([data, data], axis=1)
                if data.shape[2] == 1:
                    data = np.concatenate([data, data], axis=2)
                n_z = data.shape[3]
                out = np.zeros((len(obs_t), n_z), dtype=np.float64)
                for dt in range(2):
                    for dx in range(2):
                        for dy in range(2):
                            w = (awt[dt] * awx[dx] * awy[dy])[:, np.newaxis]
                            out += w * data[it + dt, ix + dx, iy + dy]
            else:
                data = var.transpose("t", "x", "y").values  # (n_t, n_x, n_y)
                if data.shape[0] == 1:
                    data = np.concatenate([data, data], axis=0)
                if data.shape[1] == 1:
                    data = np.concatenate([data, data], axis=1)
                if data.shape[2] == 1:
                    data = np.concatenate([data, data], axis=2)
                out = np.zeros(len(obs_t), dtype=np.float64)
                for dt in range(2):
                    for dx in range(2):
                        for dy in range(2):
                            w = awt[dt] * awx[dx] * awy[dy]
                            out += w * data[it + dt, ix + dx, iy + dy]

            result[name] = out

        return result

    @staticmethod
    def _vertical_interp_1d(
        value_profiles: np.ndarray,
        coord_profiles: np.ndarray,
        obs_z: np.ndarray,
    ) -> np.ndarray:
        """1D vertical interpolation per observation.

        For each observation, interpolates along the vertical using the
        model's coordinate profile at that location/time.

        Uses a fast vectorized path when all profiles are fully valid
        (the common case for model output), falling back to a per-observation
        loop only for profiles containing NaNs.

        Args:
            value_profiles: (n_obs, n_levels) target variable profiles
            coord_profiles: (n_obs, n_levels) vertical coordinate profiles
            obs_z: (n_obs,) target vertical coordinate values

        Returns:
            (n_obs,) interpolated values
        """
        n_obs = len(obs_z)
        result = np.empty(n_obs, dtype=np.float64)

        # Split into all-valid (vectorizable) and has-NaN (needs loop)
        all_valid = np.isfinite(coord_profiles).all(axis=1) & np.isfinite(
            value_profiles
        ).all(axis=1)
        valid_idx = np.where(all_valid)[0]
        nan_idx = np.where(~all_valid)[0]

        # --- Vectorized path for all-valid profiles ---
        if len(valid_idx) > 0:
            c = coord_profiles[valid_idx]  # (n_valid, n_levels)
            v = value_profiles[valid_idx]  # (n_valid, n_levels)
            z = obs_z[valid_idx]  # (n_valid,)

            # Sort each profile by coordinate (ascending)
            sort_idx = np.argsort(c, axis=1)
            obs_range = np.arange(len(valid_idx))[:, None]
            c = c[obs_range, sort_idx]
            v = v[obs_range, sort_idx]

            # Find bracketing level indices via broadcasting
            # (n_valid, 1) vs (n_valid, n_levels) -> (n_valid, n_levels)
            n_levels = c.shape[1]
            ge_mask = c >= z[:, None]
            has_ge = ge_mask.any(axis=1)

            # Index of the first level >= obs_z (right bracket)
            idx_right = np.where(has_ge, ge_mask.argmax(axis=1), n_levels - 1)
            idx_left = np.maximum(idx_right - 1, 0)

            # Gather bracketing values
            row_idx = np.arange(len(valid_idx))
            c_left = c[row_idx, idx_left]
            c_right = c[row_idx, idx_right]
            v_left = v[row_idx, idx_left]
            v_right = v[row_idx, idx_right]

            # Linear interpolation weight, clamped to [0, 1] for boundary extrapolation
            # (matches np.interp behavior: clamp to boundary values)
            dc = c_right - c_left
            safe_dc = np.where(dc == 0, 1.0, dc)
            weight = np.clip((z - c_left) / safe_dc, 0.0, 1.0)
            result[valid_idx] = v_left + weight * (v_right - v_left)

        # --- Scalar fallback for profiles with NaNs ---
        for i in nan_idx:
            c = coord_profiles[i, :]
            v = value_profiles[i, :]

            valid = np.isfinite(c) & np.isfinite(v)
            if valid.sum() < 2:
                result[i] = np.nan
                continue

            c_valid = c[valid]
            v_valid = v[valid]
            sort_idx = np.argsort(c_valid)
            result[i] = np.interp(obs_z[i], c_valid[sort_idx], v_valid[sort_idx])

        return result

    def _save_output(self, obs: pd.DataFrame, source: str) -> int:
        """Save interpolated values to the DuckDB observations table.

        Args:
            obs: DataFrame with 'rowid' and 'model_value' columns
            source: Either 'forecast' or 'analysis'; selects the target column

        Returns:
            Number of rows updated
        """
        n_updated = self.exp.obs.update_model_source_values(obs, source)
        logger.info(
            f"Updated {n_updated} observations with {source} model values in DuckDB"
        )
        return n_updated

    def _save_spread_output(self, obs: pd.DataFrame, source: str) -> int:
        """Save interpolated spread values to the DuckDB observations table.

        Args:
            obs: DataFrame with 'rowid' and 'model_value' columns
            source: Either 'forecast' or 'analysis'; selects the target column

        Returns:
            Number of rows updated
        """
        n_updated = self.exp.obs.update_model_source_spread_values(obs, source)
        logger.info(
            f"Updated {n_updated} observations with {source} spread values in DuckDB"
        )
        return n_updated
