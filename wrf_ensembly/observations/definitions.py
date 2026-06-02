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

    @property
    def needs_geo_axes(self) -> bool:
        """Whether this geometry requires a cartopy GeoAxes for plotting."""
        return self in (Geometry.MAP_SWATH,)


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

    display_units: str | None = None
    display_scale: float | None = None

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
    "MSG_SEVIRI": InstrumentSpec(
        label="MSG SEVIRI",
        geometry=Geometry.MAP_SWATH,
        x=AxisSpec(dim="x", label="X", coord="longitude"),
        y=AxisSpec(dim="y", label="Y", coord="latitude"),
    ),
    "GRASP_HARP2": InstrumentSpec(
        label="GRASP HARP2",
        geometry=Geometry.MAP_SWATH,
        x=AxisSpec(dim="x", label="X", coord="x"),
        y=AxisSpec(dim="y", label="Y", coord="y"),
    ),
}


QUANTITY_REGISTRY: dict[str, QuantitySpec] = {
    "LIDAR_EXTINCTION_355nm": QuantitySpec(
        label="Lidar Extinction Coefficient @ 355nm",
        units="1/m",
        vmin=0,
        model_equivalent="EXT355",
        dart_quantity="LIDAR_EXTINCTION_355nm",
        display_units="1/Mm",
        display_scale=1e6,
    ),
    "BT_WV62": QuantitySpec(
        label="Brightness Temperature WV 6.2 um",
        units="K",
        cmap="RdYlBu_r",
        vmin=200,
        vmax=260,
        model_equivalent="WV62",
    ),
    "BT_WV73": QuantitySpec(
        label="Brightness Temperature WV 7.3 um",
        units="K",
        cmap="RdYlBu_r",
        vmin=210,
        vmax=270,
        model_equivalent="WV73",
    ),
    "BT_IR87": QuantitySpec(
        label="Brightness Temperature IR 8.7 um",
        units="K",
        cmap="RdYlBu_r",
        vmin=200,
        vmax=310,
        model_equivalent="IR87",
    ),
    "BT_IR108": QuantitySpec(
        label="Brightness Temperature IR 10.8 um",
        units="K",
        cmap="RdYlBu_r",
        vmin=200,
        vmax=310,
        model_equivalent="IR108",
    ),
    "BT_IR120": QuantitySpec(
        label="Brightness Temperature IR 12.0 um",
        units="K",
        cmap="RdYlBu_r",
        vmin=200,
        vmax=310,
        model_equivalent="IR120",
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
    "AOD_355nm": QuantitySpec(
        label="Aerosol Optical Depth @ 355nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_355",
    ),
    "AOD_440nm": QuantitySpec(
        label="Aerosol Optical Depth @ 440nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_440",
    ),
    "AOD_Fine_440nm": QuantitySpec(
        label="Fine mode Aerosol Optical Depth @ 440nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_FINE_440",
    ),
    "AOD_Coarse_440nm": QuantitySpec(
        label="Coarse mode Aerosol Optical Depth @ 440nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_COARSE_440",
    ),
    "AOD_500nm": QuantitySpec(
        label="Aerosol Optical Depth @ 500nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_500",
    ),
    "AOD_550nm": QuantitySpec(
        label="Aerosol Optical Depth @ 550nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_550",
    ),
    "AOD_Fine_550nm": QuantitySpec(
        label="Fine mode Aerosol Optical Depth @ 550nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_FINE_550",
    ),
    "AOD_Coarse_550nm": QuantitySpec(
        label="Coarse mode Aerosol Optical Depth @ 550nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_COARSE_550",
    ),
    "AOD_665nm": QuantitySpec(
        label="Aerosol Optical Depth @ 665nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_665",
    ),
    "AOD_Fine_665nm": QuantitySpec(
        label="Fine mode Aerosol Optical Depth @ 665nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_FINE_665",
    ),
    "AOD_Coarse_665nm": QuantitySpec(
        label="Coarse mode Aerosol Optical Depth @ 665nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_COARSE_665",
    ),
    "AOD_870nm": QuantitySpec(
        label="Aerosol Optical Depth @ 870nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_870",
    ),
    "AOD_Fine_870nm": QuantitySpec(
        label="Fine mode Aerosol Optical Depth @ 870nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_FINE_870",
    ),
    "AOD_Coarse_870nm": QuantitySpec(
        label="Coarse mode Aerosol Optical Depth @ 870nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_DUST_COARSE_870",
    ),
    "AOD_1064nm": QuantitySpec(
        label="Aerosol Optical Depth @ 1064nm",
        units="",
        cmap="Oranges",
        vmin=0,
        vmax=2,
        model_equivalent="AOD_1064",
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
