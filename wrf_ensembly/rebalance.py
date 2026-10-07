"""
Rebalances a WRF restart file after its state was changed (analysis, perturbations).

On a cold start, WRF derives the perturbation pressure (P) and inverse density (AL) from
the dry column mass (MU), geopotential (PH), potential temperature (THM) and moisture
(start_em.F). On a restart it takes them from the file as they are, and the first time
step's pressure gradient uses them. When the analysis changes MU, PH, THM or QVAPOR in a
restart file, P and AL have to be recomputed, with the same equations WRF uses during
integration:

- AL and P: `calc_p_rho_phi` (dyn_em/module_big_step_utilities_em.F), nonhydrostatic,
  hypsometric_opt = 2
- ALT = AL + ALB
- P_HYD_W and P_HYD: `phy_prep` (same file), the moist hydrostatic pressure integrated
  down from the model top. WRF recomputes these every time step before the physics, so
  they are only rewritten to keep the file consistent.

The prognostic fields are read from their current time level (`_2`). Write the analysis
to both levels (`_1` and `_2`) before rebalancing.
"""

import netCDF4
import numpy as np

R_D = 287.0
"""Gas constant of dry air, J kg-1 K-1"""
R_V = 461.6
"""Gas constant of water vapour, J kg-1 K-1"""
CP = 7.0 * R_D / 2.0
CPOVCV = CP / (CP - R_D)
RVOVRD = R_V / R_D
P1000 = 1.0e5
"""Reference pressure (p0), Pa"""
T0 = 300.0
"""Base state potential temperature offset, K"""

MOIST_SPECIES = (
    "QVAPOR",
    "QCLOUD",
    "QRAIN",
    "QICE",
    "QSNOW",
    "QGRAUP",
    "QHAIL",
    "QICE2",
    "QICE3",
    "QICEC",
    "QICED",
    "QICEP",
)
"""Members of WRF's moist array (Registry); the hydrostatic pressure includes their mass"""

BALANCED_FIELDS = ("AL", "P", "ALT", "P_HYD", "P_HYD_W")
"""Fields `rebalance` recomputes"""


def compute_balanced_fields(
    state: dict[str, np.ndarray], p_top: float, use_theta_m: int
) -> dict[str, np.ndarray]:
    """
    Computes the fields WRF derives from the prognostic state.

    Args:
        state: Fields without the Time dimension. Prognostic: MU (y, x), PH (z_stag, y, x),
               THM (z, y, x) and the moist species present (z, y, x), with QVAPOR required.
               Base state and coordinate: MUB, PHB, ALB, PB, C1H, C2H, C3H, C4H, C3F, C4F, DNW.
        p_top: Model top pressure, Pa
        use_theta_m: WRF's use_theta_m, whether THM is moist (1) or dry (0) potential
                     temperature

    Returns:
        AL, P, ALT, P_HYD (z, y, x) and P_HYD_W (z_stag, y, x)
    """

    def column(name: str) -> np.ndarray:
        return state[name][:, np.newaxis, np.newaxis]

    mut = state["MU"] + state["MUB"]

    # Specific volume from the geopotential thickness of each layer (hypsometric_opt = 2)
    p_full = column("C3F") * mut + column("C4F") + p_top
    p_half = column("C3H") * mut + column("C4H") + p_top
    phi = state["PH"] + state["PHB"]
    al = (phi[1:] - phi[:-1]) / p_half / np.log(p_full[:-1] / p_full[1:]) - state["ALB"]

    # Pressure from the equation of state
    temp = R_D * (T0 + state["THM"]) / (P1000 * (al + state["ALB"]))
    if use_theta_m != 1:
        temp = temp * (1.0 + RVOVRD * state["QVAPOR"])
    p = P1000 * temp**CPOVCV - state["PB"]

    # Moist hydrostatic pressure, integrated down from the top. DNW is negative.
    qtot = sum(state[name] for name in MOIST_SPECIES if name in state)
    layer_weight = -(1.0 + qtot) * (column("C1H") * mut + column("C2H")) * column("DNW")
    p_hyd_w = np.empty_like(phi)
    p_hyd_w[-1] = p_top
    p_hyd_w[:-1] = p_top + np.cumsum(layer_weight[::-1], axis=0)[::-1]
    p_hyd = 0.5 * (p_hyd_w[:-1] + p_hyd_w[1:])

    return {
        "AL": al,
        "P": p,
        "ALT": al + state["ALB"],
        "P_HYD": p_hyd,
        "P_HYD_W": p_hyd_w,
    }


def rebalance(ds: netCDF4.Dataset) -> None:
    """
    Recomputes AL, P, ALT, P_HYD and P_HYD_W of an open WRF restart file from its current
    prognostic state, in place.

    Args:
        ds: WRF restart file, opened for writing
    """

    if ds.getncattr("HYPSOMETRIC_OPT") != 2:
        raise ValueError(
            f"Only hypsometric_opt = 2 is supported, the file has "
            f"{ds.getncattr('HYPSOMETRIC_OPT')}"
        )

    names = {
        "MU": "MU_2",
        "PH": "PH_2",
        "THM": "THM_2",
        **{
            name: name
            for name in (
                "MUB", "PHB", "ALB", "PB", "C1H", "C2H", "C3H", "C4H", "C3F", "C4F", "DNW"
            )
        },
        **{name: name for name in MOIST_SPECIES if name in ds.variables},
    }  # fmt: skip
    state = {name: ds[file_name][0].astype("f8") for name, file_name in names.items()}
    p_top = float(ds["P_TOP"][0])

    balanced = compute_balanced_fields(state, p_top, int(ds.getncattr("USE_THETA_M")))
    for name, field in balanced.items():
        ds[name][0] = field
