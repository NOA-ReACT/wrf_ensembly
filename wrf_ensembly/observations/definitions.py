"""Definition of instruments, quantities, and observation operators."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from wrf_ensembly.observations.operators.aeolus_hlos import hlos_wind_operator


class Geometry(str, Enum):
    MAP_SWATH = "map_swath"
    PROFILE_CURTAIN = "profile_curtain"
    AEOLUS_WINDRESULTS = "aeolus_windresults"
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
    plot_metadata_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelFieldSpec:
    """Specification for a model field required by an operator.

    Attributes:
        name: WRF variable name (e.g. "wind_east", "EXT550").
        dims: Dimensionality for interpolation.
            2 - interpolate horizontally/temporally to a scalar per obs.
            3 - interpolate horizontally/temporally, then vertically to the
                observation height/pressure, yielding a scalar per obs.
    """

    name: str
    dims: int = 2

    def __post_init__(self):
        if self.dims not in (2, 3):
            raise ValueError(f"dims must be 2 or 3, got {self.dims}")


@dataclass(frozen=True)
class OperatorSpec:
    """Defines a non-trivial observation operator.

    The operator function receives two dictionaries keyed by field/metadata
    name, with values as numpy arrays aligned with the observation count.
    It must return an (n_obs,) array of model-equivalent values.

    Attributes:
        func: Callable[[dict[str, ndarray], dict[str, ndarray]], ndarray]
            Signature: (model_fields, metadata) -> model_equivalent_values.
        required_model_fields: Model fields to interpolate before calling func.
        required_metadata: Keys to extract from the observation metadata JSON
            column before calling func.
        description: Human-readable description for logging/docs.
    """

    func: Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray]
    required_model_fields: tuple[ModelFieldSpec, ...]
    required_metadata: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class QuantitySpec:
    """Specification for a physical quantity.

    Either ``model_equivalent`` (simple string lookup) or ``operator``
    (complex multi-field operator) may be set, but not both.
    """

    label: str
    units: str
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    model_equivalent: str | None = None
    operator: OperatorSpec | None = None
    dart_quantity: str | None = None

    def __post_init__(self):
        if self.model_equivalent is not None and self.operator is not None:
            raise ValueError(
                "QuantitySpec cannot have both model_equivalent and operator set. "
                "Use model_equivalent for direct field lookups, operator for "
                "non-trivial transformations."
            )

    @property
    def has_model_mapping(self) -> bool:
        """Whether this quantity can be compared to model output."""
        return self.model_equivalent is not None or self.operator is not None

    @property
    def required_wrf_vars(self) -> list[str]:
        """All WRF variable names needed for model comparison."""
        if self.model_equivalent is not None:
            return [self.model_equivalent]
        if self.operator is not None:
            return [f.name for f in self.operator.required_model_fields]
        return []


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
    "AEOLUS_L2B_RAYLEIGH": InstrumentSpec(
        label="AEOLUS L2B HLOS Wind (Rayleigh)",
        geometry=Geometry.AEOLUS_WINDRESULTS,
        x=AxisSpec(dim="wind_result", label="Time", coord="time"),
        y=AxisSpec(dim="wind_result", label="Height [m]", coord="z"),
        plot_metadata_keys=("time_start", "time_stop", "alt_bottom", "alt_top"),
    ),
    "AEOLUS_L2B_MIE": InstrumentSpec(
        label="AEOLUS L2B HLOS Wind (Mie)",
        geometry=Geometry.AEOLUS_WINDRESULTS,
        x=AxisSpec(dim="wind_result", label="Time", coord="time"),
        y=AxisSpec(dim="wind_result", label="Height [m]", coord="z"),
        plot_metadata_keys=("time_start", "time_stop", "alt_bottom", "alt_top"),
    ),
}


QUANTITY_REGISTRY: dict[str, QuantitySpec] = {
    "LIDAR_EXTINCTION_355nm": QuantitySpec(
        label="Lidar Extinction Coefficient @ 355nm",
        units="1/m",
        vmin=0,
        model_equivalent="EXT355",
        dart_quantity="LIDAR_EXTINCTION_355nm",
    ),
    "HLOS_WIND": QuantitySpec(
        label="Horizontal Line-of-Sight Wind",
        units="m/s",
        cmap="RdBu_r",
        vmax=30,
        vmin=-30,
        operator=OperatorSpec(
            func=hlos_wind_operator,
            required_model_fields=(
                ModelFieldSpec("wind_east", dims=3),
                ModelFieldSpec("wind_north", dims=3),
            ),
            required_metadata=("azimuth",),
            description="HLOS wind projection using Aeolus azimuth (positive away from satellite)",
        ),
        dart_quantity="SAT_HLOS_WIND",
    ),
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
