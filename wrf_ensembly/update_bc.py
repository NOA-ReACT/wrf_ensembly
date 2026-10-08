"""
Updates the lateral boundary conditions (wrfbdy) to match modified initial conditions,
after cycling or applying perturbations.

Based on DART's `update_wrf_bc` (models/wrf/WRF_BC/update_wrf_bc.f90). For the boundary
record that contains the analysis time, the increment at the domain edges (analysis minus
the unmodified state) is added to the boundary value at the analysis time, and the
tendency is recomputed so that the value at the end of the record's interval stays what
real.exe produced. In effect the increment at the boundaries fades out linearly until the
next boundary time.

`update_wrf_bc` instead replaces the boundary value with the analysis itself. In the
relaxation zone the model state never exactly matches the boundary values, so that
changes the boundaries even when the state did not change, e.g. when cycling a forecast
without observations. Here an unchanged state leaves the file untouched. When the
unmodified state is the wrfinput the boundaries were made with (`wrfinput` cycling mode),
the two agree up to rounding, since the first boundary record holds that same state.

Unlike the Fortran program, the fields are coupled with column mass the way WRF v4 does it
(`couple` in dyn_em/module_big_step_utilities_em.F, as called by real.exe), using the
hybrid vertical coordinate coefficients (C1H/C2H for half levels, C1F/C2F for full levels).
`update_wrf_bc` couples with the total dry column mass only, which is only correct for the
terrain-following coordinate (`hybrid_opt = 0`). With that coordinate WRF sets C1H = C1F = 1
and C2H = C2F = 0, so the coupling here reduces to the Fortran one and both work.
"""

import datetime as dt
from collections.abc import Mapping
from pathlib import Path

import netCDF4
import numpy as np

SIDES = ("XS", "XE", "YS", "YE")
"""Boundary sides, west/east/south/north, as they appear in wrfbdy variable names"""

MOIST_VARIABLES = (
    "QVAPOR",
    "QCLOUD",
    "QRAIN",
    "QICE",
    "QSNOW",
    "QGRAUP",
    "QNICE",
)
"""Moisture fields that get updated, same list as `update_wrf_bc`. Their value at the end
of the boundary interval is clamped to be non-negative."""

STATE_VARIABLES = (
    "U",
    "V",
    "W",
    "PH",
    "THM",
    "MU",
    "MUB",
    "C1H",
    "C2H",
    "C1F",
    "C2F",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_MY",
)
"""Fields that must be present in the state given to `update_boundaries`, besides any
`MOIST_VARIABLES` that should be updated"""

THIS_BDY_TIME = "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
NEXT_BDY_TIME = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"

WRF_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def parse_wrf_times(var: "netCDF4.Variable[np.bytes_]") -> list[dt.datetime]:
    """Parses a WRF (Time, DateStrLen) character variable into timezone-aware datetimes"""

    return [
        dt.datetime.strptime(str(text), WRF_TIME_FORMAT).replace(tzinfo=dt.timezone.utc)
        for text in netCDF4.chartostring(var[:])
    ]


def write_wrf_time(
    var: "netCDF4.Variable[np.bytes_]", index: int, time: dt.datetime
) -> None:
    """Writes `time` at `index` of a WRF (Time, DateStrLen) character variable"""

    var[index] = np.frombuffer(time.strftime(WRF_TIME_FORMAT).encode(), dtype="S1")


def edges(field: np.ndarray, width: int) -> dict[str, np.ndarray]:
    """
    Cuts the outer `width` rows of a (..., south_north, west_east) field for each side of the
    domain, in the wrfbdy layout (bdy_width, ..., n). The east and north sides are counted
    inwards from the edge, so index 0 is always the outermost row.
    """

    return {
        "XS": np.moveaxis(field[..., :width], -1, 0),
        "XE": np.moveaxis(field[..., ::-1][..., :width], -1, 0),
        "YS": np.moveaxis(field[..., :width, :], -2, 0),
        "YE": np.moveaxis(field[..., ::-1, :][..., :width, :], -2, 0),
    }


def couple(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Multiplies the state with the dry air column mass, the form in which wrfbdy stores it.
    Follows WRF's `couple` subroutine: U and V use the staggered column mass and the map
    scale factor, W and PH use full level coefficients, everything else half levels.
    MU itself is stored uncoupled.

    Args:
        state: Fields from a wrfinput-like file without the Time dimension, see
               `STATE_VARIABLES`. Any `MOIST_VARIABLES` present are coupled as well.

    Returns:
        Coupled fields keyed by their wrfbdy name (THM is stored as `T` in wrfbdy).
    """

    mut = state["MU"].astype("f8") + state["MUB"]

    # Column mass on the U/V points. At the domain edges WRF uses the nearest mass point,
    # which is what edge padding gives us.
    padded = np.pad(mut, 1, mode="edge")
    muu = 0.5 * (padded[1:-1, :-1] + padded[1:-1, 1:])
    muv = 0.5 * (padded[:-1, 1:-1] + padded[1:, 1:-1])

    def half(mass: np.ndarray) -> np.ndarray:
        return state["C1H"][:, None, None] * mass + state["C2H"][:, None, None]

    def full(mass: np.ndarray) -> np.ndarray:
        return state["C1F"][:, None, None] * mass + state["C2F"][:, None, None]

    coupled = {
        "MU": state["MU"].astype("f8"),
        "U": state["U"] * half(muu) / state["MAPFAC_UY"],
        "V": state["V"] * half(muv) / state["MAPFAC_VX"],
        "W": state["W"] * full(mut) / state["MAPFAC_MY"],
        "PH": state["PH"] * full(mut),
        "T": state["THM"] * half(mut),
    }
    for name in MOIST_VARIABLES:
        if name in state:
            coupled[name] = state[name] * half(mut)

    return coupled


def update_boundaries(
    bdy: netCDF4.Dataset,
    state: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    analysis_time: dt.datetime,
) -> bool:
    """
    Updates an open wrfbdy file in place with the increment `state - reference` at the
    domain edges, at `analysis_time`.

    Only the boundary record that contains `analysis_time` is changed, so this works
    both with one wrfbdy per cycle and with one wrfbdy for the whole experiment.

    Args:
        bdy: wrfbdy file, opened for writing
        state: Modified fields at `analysis_time`, without the Time dimension. See `couple`.
        reference: The same fields before they were modified (the forecast, or real.exe's
                   wrfinput). Must contain the same moisture fields as `state`.
        analysis_time: The time of `state`.

    Returns:
        False if the increment is zero everywhere and the file was left untouched
    """

    bdy_times = parse_wrf_times(bdy["Times"])
    candidates = [i for i, t in enumerate(bdy_times) if t <= analysis_time]
    if not candidates:
        raise ValueError(
            f"Analysis time {analysis_time} is before the first boundary time {bdy_times[0]}"
        )
    itime = candidates[-1]

    this_time = parse_wrf_times(bdy[THIS_BDY_TIME])[itime]
    next_time = parse_wrf_times(bdy[NEXT_BDY_TIME])[itime]
    if not this_time <= analysis_time < next_time:
        raise ValueError(
            f"Analysis time {analysis_time} is outside boundary record {itime} "
            f"({this_time} -> {next_time})"
        )
    interval_old = (next_time - this_time).total_seconds()
    elapsed = (analysis_time - this_time).total_seconds()
    interval_new = (next_time - analysis_time).total_seconds()
    width = bdy.dimensions["bdy_width"].size

    coupled_state = couple(state)
    coupled_reference = couple(reference)
    if coupled_state.keys() != coupled_reference.keys():
        raise ValueError(
            f"The state has fields {sorted(coupled_state)}, "
            f"but the reference {sorted(coupled_reference)}"
        )
    increments = {
        name: edges(coupled_state[name] - coupled_reference[name], width)
        for name in coupled_state
    }
    if not any(
        np.any(side) for sides in increments.values() for side in sides.values()
    ):
        return False

    for name, sides in increments.items():
        for side, increment in sides.items():
            value_var = bdy[f"{name}_B{side}"]
            tend_var = bdy[f"{name}_BT{side}"]

            first_old = value_var[itime].astype("f8")
            tend_old = tend_var[itime].astype("f8")
            last = first_old + tend_old * interval_old
            first_new = first_old + tend_old * elapsed + increment
            if name in MOIST_VARIABLES:
                last = np.maximum(last, 0.0)
                first_new = np.maximum(first_new, 0.0)

            value_var[itime] = first_new
            tend_var[itime] = (last - first_new) / interval_new

    write_wrf_time(bdy[THIS_BDY_TIME], itime, analysis_time)
    return True


def read_state(ds: netCDF4.Dataset) -> tuple[dict[str, np.ndarray], dt.datetime]:
    """
    Reads the fields `update_boundaries` needs from a wrfinput, wrfout or WRF restart
    file. Restart files keep both time levels of the prognostic fields (`U_1`, `U_2`,
    ...), and `_2` is the current state.

    Returns:
        The fields without the Time dimension, and the file's time.
    """

    def name_in_file(name: str) -> str:
        return f"{name}_2" if f"{name}_2" in ds.variables else name

    names = [*STATE_VARIABLES, *(v for v in MOIST_VARIABLES if v in ds.variables)]
    state = {name: ds[name_in_file(name)][0].astype("f8") for name in names}
    time = parse_wrf_times(ds["Times"])[0]
    return state, time


def update_wrf_bc(state_path: Path, reference_path: Path, wrfbdy: Path) -> bool:
    """
    Updates the given `wrfbdy` file with the changes made to a wrfinput or restart file.
    Required if you have modified the initial state (analysis, perturbations).

    Args:
        state_path: The modified wrfinput or restart file.
        reference_path: The same file before it was modified.
        wrfbdy: The wrfbdy file to update. Will be mutated.

    Returns:
        False if nothing changed at the boundaries and the file was left untouched
    """

    with netCDF4.Dataset(state_path, "r") as ds:  # type: ignore
        ds.set_auto_mask(False)
        state, time = read_state(ds)
    with netCDF4.Dataset(reference_path, "r") as ds:  # type: ignore
        ds.set_auto_mask(False)
        reference, reference_time = read_state(ds)
    if reference_time != time:
        raise ValueError(
            f"{state_path} is at {time} but the reference {reference_path} at "
            f"{reference_time}"
        )

    with netCDF4.Dataset(wrfbdy, "r+") as bdy:  # type: ignore
        bdy.set_auto_mask(False)
        return update_boundaries(bdy, state, reference, time)
