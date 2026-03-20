"""Model interpolation to observation locations and times."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.console import logger
from wrf_ensembly.experiment.experiment import Experiment
from wrf_ensembly.observations import QUANTITY_REGISTRY, OperatorSpec


class ModelInterpolation:
    """Interpolates model outputs to observation locations and times.

    This class handles the validation workflow of comparing model forecasts
    to observations by interpolating model data spatially and temporally to
    match observation points.

    Supports both simple direct-field lookups (via QuantitySpec.model_equivalent)
    and complex observation operators (via QuantitySpec.operator) that combine
    multiple model fields and observation metadata.
    """

    def __init__(self, experiment: Experiment):
        """Initialize with an experiment.

        Args:
            experiment: The experiment to perform validation on
        """
        self.exp = experiment

    def run(self) -> Path | None:
        """Run the model interpolation.

        Creates a parquet file containing observations with an additional
        'model_value' column representing the interpolated model value at
        each observation location and time.

        Returns:
            Path to the output model_interpolated.parquet file, None on failure
        """

        needed_combos = self._determine_needed_combos()
        if not needed_combos:
            logger.info("No valid instrument/quantity combos found, cannot proceed!")
            return None

        # Collect all WRF variable names we need (target vars + vertical coords)
        needed_vars: set[str] = {"z", "air_pressure", "geopotential_height"}
        for combo in needed_combos:
            needed_vars.update(combo["wrf_vars"])

        forecast_mean = self._open_forecasts(list(needed_vars))

        result_df = self._interpolate_all(needed_combos, forecast_mean)

        result_df = self._mark_da_usage(result_df)

        return self._save_output(result_df)

    def _instrument_where_clause(self, prefix: str = "WHERE") -> str:
        """Build a SQL WHERE/AND clause for instrument filtering.

        Args:
            prefix: SQL keyword to use ('WHERE' or 'AND')

        Returns:
            SQL clause string, or empty string if no filter is configured
        """
        if not self.exp.cfg.validation.instruments:
            return ""
        instruments_list = ", ".join(
            f"'{inst}'" for inst in self.exp.cfg.validation.instruments
        )
        return f"{prefix} instrument IN ({instruments_list})"

    def _determine_needed_combos(self) -> list[dict]:
        """Determine which instrument/quantity combinations need interpolation.

        Queries DuckDB for distinct combinations and resolves their WRF
        variable mappings and vertical coordinate types. Handles both simple
        model_equivalent lookups and operator-based quantities.

        Returns:
            List of dicts with keys: instrument, quantity, wrf_vars, z_type,
            has_operator
        """
        where_clause = self._instrument_where_clause()

        with self.exp.obs._get_duckdb(read_only=True) as con:
            combos = con.execute(
                f"SELECT DISTINCT instrument, quantity, z_type FROM observations {where_clause}"
            ).fetchall()

        needed_combos = []
        for instrument, quantity, z_type in combos:
            quantity_info = QUANTITY_REGISTRY.get(quantity)
            if quantity_info is None or not quantity_info.has_model_mapping:
                logger.warning(
                    f"Quantity {quantity} not found in observation mappings or "
                    "has no model mapping, skipping"
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

    def _open_forecasts(self, needed_vars: list[str]) -> xr.Dataset:
        """Open forecast mean files as a single lazy xarray Dataset.

        Args:
            needed_vars: List of WRF variable names to load

        Returns:
            Lazy xarray Dataset with only the needed variables

        Raises:
            ValueError: If duplicate coordinates are found
        """

        forecast_mean = xr.open_mfdataset(
            f"{self.exp.paths.data_forecasts}/cycle_**/forecast_mean_cycle_*.nc",
            combine="by_coords",
            chunks={"time": 1},
            coords="minimal",
        )

        available = [v for v in needed_vars if v in forecast_mean.data_vars]
        missing = [v for v in needed_vars if v not in forecast_mean.data_vars]
        if missing:
            logger.warning(f"Variables not found in forecast files: {missing}")
        forecast_mean = forecast_mean[available]

        # Check for duplicate coordinates
        for coord_name in ["t", "x", "y"]:
            if coord_name not in forecast_mean.coords:
                continue
            coord_vals = forecast_mean.coords[coord_name].values
            unique, counts = np.unique(coord_vals, return_counts=True)
            duplicates = unique[counts > 1]

            if len(duplicates) > 0:
                print(
                    f"Duplicate values found in forecast_mean coordinate '{coord_name}':"
                )
                for dup in duplicates:
                    indices = np.where(coord_vals == dup)[0]
                    print(f"  Value: {dup}")
                    print(f"  Appears {len(indices)} times at indices: {indices}")

                raise ValueError(
                    f"Duplicate values found in forecast_mean coordinate '{coord_name}': {duplicates}"
                )

        return forecast_mean

    def _interpolate_all(
        self, needed_combos: list[dict], forecast_mean: xr.Dataset
    ) -> pd.DataFrame:
        """Interpolate model forecasts to all observation combos.

        For each instrument/quantity combination, fetches observations from
        DuckDB, performs interpolation (simple or operator-based), and
        assembles results.

        Args:
            needed_combos: List of combo dicts from _determine_needed_combos
            forecast_mean: Lazy forecast Dataset

        Returns:
            DataFrame with all observations and their model_value
        """
        instrument_filter = self._instrument_where_clause(prefix="AND")
        all_results = []

        for combo in needed_combos:
            instrument = combo["instrument"]
            quantity = combo["quantity"]
            z_type = combo["z_type"]
            quantity_info = QUANTITY_REGISTRY[quantity]

            logger.info(
                f"Interpolating {instrument}/{quantity} "
                f"(vars={combo['wrf_vars']}, z_type={z_type})"
            )

            # Determine if we need to fetch metadata from DuckDB
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

            # Fetch numeric + geographic columns from DuckDB
            with self.exp.obs._get_duckdb(read_only=True) as con:
                arrays = con.execute(
                    f"""
                    SELECT time, x, y, z, latitude, longitude, value
                        {metadata_sql}
                    FROM observations
                    WHERE instrument = ? AND quantity = ?
                    {instrument_filter}
                    """,
                    [instrument, quantity],
                ).fetchnumpy()

            n_obs = len(arrays["x"])
            if n_obs == 0:
                logger.warning(f"No observations for {instrument}/{quantity}, skipping")
                continue

            logger.info(f"  {n_obs} observations")

            times = pd.DatetimeIndex(arrays["time"]).tz_localize(None)
            batch_t = xr.DataArray(times, dims="obs")
            batch_x = xr.DataArray(arrays["x"], dims="obs")
            batch_y = xr.DataArray(arrays["y"], dims="obs")

            if quantity_info.operator is not None:
                model_vals = self._apply_operator(
                    forecast_mean,
                    quantity_info.operator,
                    z_type,
                    batch_t,
                    batch_x,
                    batch_y,
                    arrays,
                )
            else:
                wrf_var = quantity_info.model_equivalent
                model_vals = self._interpolate_simple(
                    forecast_mean,
                    wrf_var,
                    z_type,
                    batch_t,
                    batch_x,
                    batch_y,
                    arrays["z"],
                )

            chunk_df = pd.DataFrame(
                {
                    "time": arrays["time"],
                    "x": arrays["x"],
                    "y": arrays["y"],
                    "z": arrays["z"],
                    "latitude": arrays["latitude"],
                    "longitude": arrays["longitude"],
                    "z_type": z_type,
                    "value": arrays["value"],
                    "model_value": model_vals,
                    "instrument": instrument,
                    "quantity": quantity,
                }
            )
            all_results.append(chunk_df)

        return pd.concat(all_results, ignore_index=True)

    def _interpolate_simple(
        self,
        forecast_mean: xr.Dataset,
        wrf_var: str,
        z_type: str,
        batch_t: xr.DataArray,
        batch_x: xr.DataArray,
        batch_y: xr.DataArray,
        obs_z: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a single model field to observation locations.

        This is the original interpolation path for quantities with a direct
        model_equivalent mapping.

        Args:
            forecast_mean: Lazy forecast Dataset
            wrf_var: WRF variable name
            z_type: Vertical coordinate type ("columnar", "surface", "height", "pressure")
            batch_t: Time coordinates for observations
            batch_x: X coordinates for observations
            batch_y: Y coordinates for observations
            obs_z: Vertical coordinate values for observations

        Returns:
            (n_obs,) array of interpolated model values
        """
        n_obs = len(obs_z)

        if wrf_var not in forecast_mean.data_vars:
            logger.warning(f"Variable {wrf_var} not in forecast data")
            return np.full(n_obs, np.nan)

        if z_type in ("columnar", "surface"):
            return forecast_mean[wrf_var].interp(t=batch_t, x=batch_x, y=batch_y).values
        elif z_type == "height":
            return self._interp_vertical(
                forecast_mean,
                wrf_var,
                "geopotential_height",
                batch_t,
                batch_x,
                batch_y,
                obs_z,
            )
        elif z_type == "pressure":
            return self._interp_vertical(
                forecast_mean,
                wrf_var,
                "air_pressure",
                batch_t,
                batch_x,
                batch_y,
                obs_z,
                flip=True,
            )
        else:
            logger.warning(f"Unknown z_type '{z_type}'")
            return np.full(n_obs, np.nan)

    def _apply_operator(
        self,
        forecast_mean: xr.Dataset,
        operator: OperatorSpec,
        z_type: str,
        batch_t: xr.DataArray,
        batch_x: xr.DataArray,
        batch_y: xr.DataArray,
        obs_arrays: dict,
    ) -> np.ndarray:
        """Interpolate all required model fields and apply the observation operator.

        For each field in the operator's required_model_fields:
        - dims=2: horizontal/temporal interpolation to a scalar per obs
        - dims=3: horizontal/temporal + vertical interpolation to a scalar per obs

        Metadata fields are extracted from the DuckDB query results (already
        present in obs_arrays as meta_<key> columns).

        Args:
            forecast_mean: Lazy forecast Dataset
            operator: The OperatorSpec to apply
            z_type: Vertical coordinate type
            batch_t: Time coordinates for observations
            batch_x: X coordinates for observations
            batch_y: Y coordinates for observations
            obs_arrays: Dict of arrays from the DuckDB query, including
                meta_<key> columns for required metadata

        Returns:
            (n_obs,) array of operator-computed model-equivalent values
        """
        n_obs = len(obs_arrays["x"])

        # Check that all required model fields are available
        missing_fields = [
            f.name
            for f in operator.required_model_fields
            if f.name not in forecast_mean.data_vars
        ]
        if missing_fields:
            logger.warning(
                f"Model fields {missing_fields} not in forecast data, "
                "operator cannot run"
            )
            return np.full(n_obs, np.nan)

        # Interpolate each required model field
        model_fields: dict[str, np.ndarray] = {}
        for field_spec in operator.required_model_fields:
            if field_spec.dims == 2:
                model_fields[field_spec.name] = (
                    forecast_mean[field_spec.name]
                    .interp(t=batch_t, x=batch_x, y=batch_y)
                    .values
                )
            elif field_spec.dims == 3:
                model_fields[field_spec.name] = self._interpolate_simple(
                    forecast_mean,
                    field_spec.name,
                    z_type,
                    batch_t,
                    batch_x,
                    batch_y,
                    obs_arrays["z"],
                )

        # Collect metadata arrays (extracted via SQL in _interpolate_all)
        metadata: dict[str, np.ndarray] = {}
        for key in operator.required_metadata:
            col_name = f"meta_{key}"
            if col_name not in obs_arrays:
                logger.error(
                    f"Required metadata '{key}' not found in observation data. "
                    "Check that the converter populates this field."
                )
                return np.full(n_obs, np.nan)

            vals = obs_arrays[col_name]
            n_invalid = (
                np.sum(np.isnan(vals)) if np.issubdtype(vals.dtype, np.floating) else 0
            )
            if n_invalid > 0:
                logger.warning(
                    f"Metadata '{key}' has {n_invalid}/{n_obs} NaN values "
                    f"({100 * n_invalid / n_obs:.1f}%)"
                )
            metadata[key] = vals

        return operator.func(model_fields, metadata)

    def _interp_vertical(
        self,
        forecast_mean: xr.Dataset,
        wrf_var: str,
        vertical_coord_var: str,
        batch_t: xr.DataArray,
        batch_x: xr.DataArray,
        batch_y: xr.DataArray,
        obs_z: np.ndarray,
        flip: bool = False,
    ) -> np.ndarray:
        """Interpolate a 3D variable vertically to observation altitudes/pressures.

        Performs horizontal/temporal interpolation first to get vertical
        profiles at each observation location, then does 1D vertical
        interpolation per observation.

        Args:
            forecast_mean: Lazy forecast Dataset
            wrf_var: Name of the target variable
            vertical_coord_var: Name of the vertical coordinate variable
                (e.g., 'geopotential_height' for height, 'air_pressure' for
                pressure)
            batch_t: Time coordinates for observations
            batch_x: X coordinates for observations
            batch_y: Y coordinates for observations
            obs_z: Target vertical coordinate values (height or pressure)
            flip: If True, reverse the vertical axis before interpolation
                (needed for pressure, which decreases with altitude)

        Returns:
            Array of interpolated values, one per observation
        """
        # Interpolate both the target variable and the vertical coordinate
        # horizontally/temporally in a single compute call to avoid
        # evaluating the dask graph twice.
        interp_ds = (
            forecast_mean[[wrf_var, vertical_coord_var]]
            .interp(t=batch_t, x=batch_x, y=batch_y)
            .compute()
        )

        profiles = interp_ds[wrf_var].values  # (n_obs, n_levels)
        coord_profiles = interp_ds[vertical_coord_var].values  # (n_obs, n_levels)

        if flip:
            profiles = profiles[:, ::-1]
            coord_profiles = coord_profiles[:, ::-1]

        return self._vertical_interp_1d(profiles, coord_profiles, obs_z)

    @staticmethod
    def _vertical_interp_1d(
        value_profiles: np.ndarray,
        coord_profiles: np.ndarray,
        obs_z: np.ndarray,
    ) -> np.ndarray:
        """1D vertical interpolation per observation.

        For each observation, interpolates along the vertical using the
        model's coordinate profile at that location/time.

        Args:
            value_profiles: (n_obs, n_levels) target variable profiles
            coord_profiles: (n_obs, n_levels) vertical coordinate profiles
            obs_z: (n_obs,) target vertical coordinate values

        Returns:
            (n_obs,) interpolated values
        """
        n_obs = len(obs_z)
        result = np.empty(n_obs, dtype=np.float64)

        for i in range(n_obs):
            c = coord_profiles[i, :]
            v = value_profiles[i, :]

            valid = np.isfinite(c) & np.isfinite(v)
            if valid.sum() < 2:
                result[i] = np.nan
                continue

            c_valid = c[valid]
            v_valid = v[valid]

            # np.interp requires monotonically increasing x
            sort_idx = np.argsort(c_valid)
            result[i] = np.interp(obs_z[i], c_valid[sort_idx], v_valid[sort_idx])

        return result

    def _mark_da_usage(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Mark which observations were used in data assimilation.

        An observation is marked as used if:
        - It falls within the DA window of a cycle
        - Its instrument is in the instruments_to_assimilate list

        Args:
            obs: Observations DataFrame

        Returns:
            DataFrame with 'used_in_da' and 'cycle' columns added
        """
        da_instruments = self.exp.cfg.observations.instruments_to_assimilate
        half_window = self.exp.cfg.assimilation.half_window_length_minutes

        obs["used_in_da"] = False
        obs["cycle"] = pd.NA

        for cycle in self.exp.cycles:
            start_time = pd.to_datetime(cycle.end) - pd.Timedelta(minutes=half_window)
            end_time = pd.to_datetime(cycle.end) + pd.Timedelta(minutes=half_window)

            in_window = (obs["time"] >= start_time.to_numpy()) & (
                obs["time"] <= end_time.to_numpy()
            )
            if da_instruments is not None:
                is_da_instrument = obs["instrument"].isin(da_instruments)
                obs.loc[in_window & is_da_instrument, "used_in_da"] = True
                obs.loc[in_window & is_da_instrument, "cycle"] = cycle.index
            else:
                obs.loc[in_window, "used_in_da"] = True
                obs.loc[in_window, "cycle"] = cycle.index

        return obs

    def _save_output(self, obs: pd.DataFrame) -> Path:
        """Save the observations with model values to a parquet file.

        Args:
            obs: Observations DataFrame with model_value column

        Returns:
            Path to the output file
        """
        output_path = self.exp.paths.data / "model_interpolated.parquet"
        obs.to_parquet(output_path)
        logger.info(f"Saved model interpolated data to {output_path}")

        return output_path
