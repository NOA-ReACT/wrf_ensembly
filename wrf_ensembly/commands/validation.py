from pathlib import Path

import click
import duckdb
import pandas as pd
import xarray as xr

from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.experiment import experiment
from wrf_ensembly.console import logger
from wrf_ensembly.observations.mapping import QUANTITY_TO_WRF_VAR


@click.group(name="validation", cls=GroupWithStartEndPrint)
def validation_cli():
    """Commands related to validating an experiment, i.e. comparing model output to observations"""
    pass


@validation_cli.command()
@pass_experiment_path
def interpolate_model(experiment_path: Path):
    """
    Interpolate the model outputs to the observation locations and times.

    This will create a `validation` directory in the experiment directory, containing
    the results of the validation in parquet format.

    The output file will be in WRF ensembly observation file format but include additional columns:
    - model_value: The value from the model at the observation location and time
    - used_in_da: A boolean indicating if the observation was used in data assimilation
    - cycle: The index of the assimilation cycle the observation was used in (if any)
    """

    logger.setup("validation-interpolate-model", experiment_path)
    exp = experiment.Experiment(experiment_path)

    obs = (
        exp.obs._get_duckdb(read_only=True)
        .execute("SELECT * FROM observations WHERE downsampling_info IS NULL")
        .fetchdf()
    )

    # Mark observations used in DA:
    # - They must be within the DA window of a cycle (`assimilation::half_window_length_minutes`)
    # - They must be of an instrument that is assimilated (`observations::instruments_to_assimilate`)
    da_instruments = exp.cfg.observations.instruments_to_assimilate
    half_window = exp.cfg.assimilation.half_window_length_minutes

    obs["used_in_da"] = False
    obs["cycle"] = pd.NA

    for cycle in exp.cycles:
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

    # Gather which WRF variables are needed for the observations
    needed_vars = set()
    for quantity in obs["quantity"].unique():
        if quantity in QUANTITY_TO_WRF_VAR:
            needed_vars.add(QUANTITY_TO_WRF_VAR[quantity])
        else:
            logger.warning(
                f"Quantity {quantity} not found in observation mappings, cannot determine WRF variable"
            )
    needed_vars = list(needed_vars)
    logger.info(f"Need to interpolate WRF variables: {needed_vars}")

    # Open all model output forecast mean files as a single xarray dataset
    # TODO use mean?
    forecast_mean = xr.open_mfdataset(
        f"{exp.paths.data_forecasts}/cycle_**/forecast_member_00_cycle_*.nc",
        combine="by_coords",
        chunks={"time": 1},
        coords="minimal",
    )[needed_vars]

    # Interpolate the model data to the observation locations and times, convert back to dataframe
    obs_ds = xr.Dataset.from_dataframe(obs).set_coords(["time", "x", "y"])
    print(obs_ds)
    model_obs = forecast_mean.interp(
        t=obs_ds["time"], x=obs_ds["x"], y=obs_ds["y"]
    ).compute()

    model_obs_df = model_obs.to_dataframe().reset_index()

    # At this point, the dataframe contains one column per WRF variable, we want to keep
    # only the original quantity for each variable
    def get_column_that_matches_quantity(row):
        # We have to grab the quantity from the original obs dataframe
        quantity = str(obs.loc[row["index"], "quantity"])
        var_name = QUANTITY_TO_WRF_VAR.get(quantity, None)
        if var_name is not None and var_name in row:
            return row[var_name]
        else:
            return pd.NA

    model_obs_df["model_value"] = model_obs_df.apply(
        get_column_that_matches_quantity, axis=1
    )

    # Put the model_value in the original dataframe
    obs["model_value"] = model_obs_df["model_value"]

    output_path = exp.paths.data / "model_interpolated.parquet"
    obs.to_parquet(output_path)
