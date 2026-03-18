"""Definition of instruments and quantities"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Geometry(str, Enum):
    MAP_SWATH = "map_swath"
    PROFILE_CURTAIN = "profile_curtain"
    TIMESERIES = "timeseries"


@dataclass(frozen=True)
class AxisSpec:
    dim: str
    label: str
    coord: str


@dataclass(frozen=True)
class InstrumentSpec:
    label: str
    geometry: Geometry
    x: AxisSpec
    y: AxisSpec | None = None


@dataclass(frozen=True)
class QuantitySpec:
    label: str
    units: str
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    model_equivalent: str | None = None
    dart_quantity: str | None = None


INSTRUMENT_REGISTRY: dict[str, InstrumentSpec] = {
    "AEOLUS_L2A_MLE": InstrumentSpec(
        label="AEOLUS L2A MLE",
        geometry=Geometry.PROFILE_CURTAIN,
        x=AxisSpec(dim="profile", label="Profiles", coord="time"),
        y=AxisSpec(dim="height_bin", label="Height", coord="z"),
    ),
    "AEOLUS_L2A_SCA": InstrumentSpec(
        label="AEOLUS L2A SCA",
        geometry=Geometry.PROFILE_CURTAIN,
        x=AxisSpec(dim="profile", label="Profiles", coord="time"),
        y=AxisSpec(dim="height_bin", label="Height", coord="z"),
    ),
    "AEOLUS_L2A_AEL_PRO": InstrumentSpec(
        label="AEOLUS L2A AEL-PRO",
        geometry=Geometry.PROFILE_CURTAIN,
        x=AxisSpec(dim="profile", label="Profiles", coord="time"),
        y=AxisSpec(dim="height_bin", label="Height", coord="z"),
    ),
}


QUANTITY_REGISTRY: dict[str, QuantitySpec] = {
    "LIDAR_EXTINCTION_355nm": QuantitySpec(
        label="Lidar Extinction Coefficient @ 355nm",
        units="1/km",
        vmin=0,
        model_equivalent="EXT355",
        dart_quantity="LIDAR_EXTINCTION_355nm",
    )
}


def reshape_to_native(
    df: pd.DataFrame,
    field: str = "value",
) -> tuple[np.ndarray, tuple[int, ...], list[str]]:
    """
    Given a wrf-ensembly observation dataframe of ONE original file, folds back the data
    into the original shape.
    """

    # Sanity checking
    instrument_quantity = df["instrument"] + "." + df["quantity"]
    if len(instrument_quantity.unique()) != 1:
        raise ValueError("All rows must have the same instrument and quantity")
    if len(df["orig_filename"]) != 1:
        raise ValueError("All rows must have the same orig_filename")

    coords = df.iloc[0]["orig_coords"]
    shape = tuple(coords["shape"])
    names = coords["names"]

    arr = np.full(shape, np.nan)
    indices = df["orig_coords"].apply(lambda r: tuple(r["indices"]))
    arr[tuple(zip(*indices))] = df[field].values

    return arr, shape, names
