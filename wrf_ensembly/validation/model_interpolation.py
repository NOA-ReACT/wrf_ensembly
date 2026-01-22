"""Model interpolation to observation locations and times."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.console import logger
from wrf_ensembly.experiment.experiment import Experiment
from wrf_ensembly.observations.mapping import QUANTITY_TO_WRF_VAR


class ModelInterpolation:
    """Interpolates model outputs to observation locations and times.

    This class handles the validation workflow of comparing model forecasts
    to observations by interpolating model data spatially and temporally to
    match observation points.
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
        obs = self._load_observations()
        obs = self._mark_da_usage(obs)
        needed_vars = self._determine_needed_vars(obs)

        if not needed_vars:
            logger.info("No WRF variables needed for interpolation, cannot proceed!")
            return None

        obs_ds = self._observations_to_dataset(obs)
        model_obs_df = self._interpolate_model_to_obs(obs_ds, needed_vars)
        obs = self._add_model_values(obs, model_obs_df)

        return self._save_output(obs)

    def _load_observations(self) -> pd.DataFrame:
        """Load observations from the experiment database.

        Filters out downsampled observations and applies instrument filters
        from the experiment configuration.

        Returns:
            DataFrame of observations
        """
        where_conditions = ["downsampling_info IS NULL"]

        if self.exp.cfg.validation.instruments:
            instruments_to_use = self.exp.cfg.validation.instruments
            logger.info(
                f"Filtering observations to only use instruments: {instruments_to_use}"
            )
            instruments_list = ", ".join(f"'{inst}'" for inst in instruments_to_use)
            where_conditions.append(f"instrument IN ({instruments_list})")

        where_clause = " AND ".join(where_conditions)
        obs = (
            self.exp.obs._get_duckdb(read_only=True)
            .execute(f"SELECT * FROM observations WHERE {where_clause}")
            .fetchdf()
        )

        return obs

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

            in_window = (obs["time"] >= start_time) & (obs["time"] <= end_time)
            if da_instruments is not None:
                is_da_instrument = obs["instrument"].isin(da_instruments)
                obs.loc[in_window & is_da_instrument, "used_in_da"] = True
                obs.loc[in_window & is_da_instrument, "cycle"] = cycle.index
            else:
                obs.loc[in_window, "used_in_da"] = True
                obs.loc[in_window, "cycle"] = cycle.index

        return obs

    def _determine_needed_vars(self, obs: pd.DataFrame) -> list[str]:
        """Determine which WRF variables are needed for the observations.

        Args:
            obs: Observations DataFrame

        Returns:
            List of WRF variable names needed
        """
        needed_vars = set()
        for quantity in obs["quantity"].unique():
            if quantity in QUANTITY_TO_WRF_VAR:
                needed_vars.add(QUANTITY_TO_WRF_VAR[quantity])
            else:
                logger.warning(
                    f"Quantity {quantity} not found in observation mappings, "
                    "cannot determine WRF variable"
                )

        needed_vars = list(needed_vars)
        logger.info(f"Need to interpolate WRF variables: {needed_vars}")

        return needed_vars

    def _observations_to_dataset(self, obs: pd.DataFrame) -> xr.Dataset:
        """Convert observations DataFrame to xarray Dataset for interpolation.

        Args:
            obs: Observations DataFrame

        Returns:
            xarray Dataset with time, x, y as coordinates
        """
        obs_ds = xr.Dataset.from_dataframe(obs).set_coords(["time", "x", "y"])
        # Convert Timestamps to DateTimeIndex without timezone info
        # xarray does not support tz-aware times
        obs_ds["time"] = (
            ("index",),
            pd.DatetimeIndex(obs_ds["time"].values).tz_localize(None),
        )
        logger.debug(f"Observation dataset:\n{obs_ds}")

        return obs_ds

    def _interpolate_model_to_obs(
        self, obs_ds: xr.Dataset, needed_vars: list[str]
    ) -> pd.DataFrame:
        """Interpolate model forecasts to observation locations and times.

        Args:
            obs_ds: Observations as xarray Dataset
            needed_vars: List of WRF variables to interpolate

        Returns:
            DataFrame with interpolated model values

        Raises:
            ValueError: If duplicate coordinates are found in forecast_mean
        """

        # Open all model output forecast mean files as a single xarray dataset
        # TODO: Allow mean/member selection
        forecast_mean = xr.open_mfdataset(
            f"{self.exp.paths.data_forecasts}/cycle_**/forecast_mean_cycle_*.nc",
            combine="by_coords",
            chunks={"time": 1},
            coords="minimal",
        )[needed_vars]

        # Check forecast_mean coordinates for duplicates
        for coord_name in ["t", "x", "y"]:
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

        # Interpolate the model data to observation locations and times
        model_obs = forecast_mean.interp(
            t=obs_ds["time"], x=obs_ds["x"], y=obs_ds["y"]
        ).compute()

        model_obs_df = model_obs.to_dataframe().reset_index()

        return model_obs_df

    def _add_model_values(
        self, obs: pd.DataFrame, model_obs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add interpolated model values to the observations DataFrame.

        For each observation, selects the appropriate WRF variable value
        based on the observation quantity.

        Args:
            obs: Original observations DataFrame
            model_obs_df: DataFrame with interpolated model values

        Returns:
            Observations DataFrame with 'model_value' column added
        """

        def get_column_that_matches_quantity(row):
            # Get the quantity from the original obs dataframe
            quantity = str(obs.loc[row["index"], "quantity"])
            var_name = QUANTITY_TO_WRF_VAR.get(quantity, None)
            if var_name is not None and var_name in row:
                return row[var_name]
            else:
                return pd.NA

        model_obs_df["model_value"] = model_obs_df.apply(
            get_column_that_matches_quantity, axis=1
        )

        # Add the model_value to the original dataframe
        obs["model_value"] = model_obs_df["model_value"]

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
