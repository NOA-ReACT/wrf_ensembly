from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wrf_ensembly import rebalance

NZ, NY, NX = 10, 3, 4
P_TOP = 5000.0


def make_state(rng: np.random.Generator) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    A column state with hybrid coordinate coefficients, whose geopotential is built from
    a known inverse density, so `compute_balanced_fields` should get it back.

    Returns:
        The state, and the inverse density perturbation (AL) it was built from
    """

    znw = np.linspace(1.0, 0.0, NZ + 1)
    znu = 0.5 * (znw[:-1] + znw[1:])
    c3f, c3h = znw**1.5, znu**1.5  # Hybrid-like: terrain following near the surface
    c4f = (znw - c3f) * (1.0e5 - P_TOP)
    c4h = (znu - c3h) * (1.0e5 - P_TOP)
    c1h = np.gradient(c3f, znw)[:-1]
    mub = 9.0e4 + 1000 * rng.random((NY, NX))
    mu = 300 * rng.standard_normal((NY, NX))
    alb = np.broadcast_to(
        np.linspace(0.85, 3.0, NZ)[:, None, None], (NZ, NY, NX)
    ).copy()
    al = 0.01 * rng.standard_normal((NZ, NY, NX))

    def column(a: np.ndarray) -> np.ndarray:
        return a[:, None, None]

    p_full = column(c3f) * (mu + mub) + column(c4f) + P_TOP
    p_half = column(c3h) * (mu + mub) + column(c4h) + P_TOP
    dphi = (al + alb) * p_half * np.log(p_full[:-1] / p_full[1:])
    phi = np.concatenate([np.zeros((1, NY, NX)), np.cumsum(dphi, axis=0)])
    phb = 0.9 * phi

    state = {
        "MU": mu,
        "MUB": mub,
        "PH": phi - phb,
        "PHB": phb,
        "ALB": alb,
        "PB": np.full((NZ, NY, NX), 5.0e4),
        "THM": 10 + 5 * rng.random((NZ, NY, NX)),
        "QVAPOR": 0.01 * rng.random((NZ, NY, NX)),
        "QCLOUD": 0.001 * rng.random((NZ, NY, NX)),
        "C1H": c1h,
        "C2H": (1 - c1h) * (1.0e5 - P_TOP),
        "C3H": c3h,
        "C4H": c4h,
        "C3F": c3f,
        "C4F": c4f,
        "DNW": np.diff(znw),
    }
    return state, al


def test_inverse_density_from_geopotential():
    state, al = make_state(np.random.default_rng(0))

    out = rebalance.compute_balanced_fields(state, P_TOP, use_theta_m=1)

    np.testing.assert_allclose(out["AL"], al, atol=1e-12)
    np.testing.assert_allclose(out["ALT"], al + state["ALB"], atol=1e-12)


def test_pressure_from_equation_of_state():
    state, al = make_state(np.random.default_rng(0))
    alpha = al + state["ALB"]

    moist = rebalance.compute_balanced_fields(state, P_TOP, use_theta_m=1)
    dry = rebalance.compute_balanced_fields(state, P_TOP, use_theta_m=0)

    # Moist theta (use_theta_m = 1) already includes the vapour, dry theta needs it added
    theta_m = rebalance.T0 + state["THM"]
    expected = (
        rebalance.P1000
        * (rebalance.R_D * theta_m / (rebalance.P1000 * alpha)) ** rebalance.CPOVCV
    )
    np.testing.assert_allclose(moist["P"] + state["PB"], expected, rtol=1e-12)
    qvf = 1 + rebalance.RVOVRD * state["QVAPOR"]
    np.testing.assert_allclose(
        dry["P"] + state["PB"], expected * qvf**rebalance.CPOVCV, rtol=1e-12
    )


def test_hydrostatic_pressure_integrates_the_moist_column():
    state, _ = make_state(np.random.default_rng(0))

    out = rebalance.compute_balanced_fields(state, P_TOP, use_theta_m=1)

    p_hyd_w = out["P_HYD_W"]
    np.testing.assert_allclose(p_hyd_w[-1], P_TOP)
    # The surface pressure is the dry column mass plus the water it carries
    mut = state["MU"] + state["MUB"]
    layer_mass = -(state["C1H"][:, None, None] * mut + state["C2H"][:, None, None])
    layer_mass = layer_mass * state["DNW"][:, None, None]
    water = ((state["QVAPOR"] + state["QCLOUD"]) * layer_mass).sum(axis=0)
    np.testing.assert_allclose(p_hyd_w[0], P_TOP + layer_mass.sum(axis=0) + water)
    np.testing.assert_allclose(out["P_HYD"], 0.5 * (p_hyd_w[:-1] + p_hyd_w[1:]))


def write_restart(
    path: Path, state: dict[str, np.ndarray], hypsometric_opt: int
) -> None:
    """A minimal restart file holding `state`, prognostic fields under their `_2` name"""

    with netCDF4.Dataset(path, "w") as ds:
        ds.HYPSOMETRIC_OPT = np.int32(hypsometric_opt)
        ds.USE_THETA_M = np.int32(1)
        ds.createDimension("Time", None)
        for dim, size in (("z", NZ), ("z_stag", NZ + 1), ("y", NY), ("x", NX)):
            ds.createDimension(dim, size)
        ds.createVariable("P_TOP", "f8", ("Time",))[0] = P_TOP

        def dims(field: np.ndarray) -> tuple[str, ...]:
            vertical = {NZ: "z", NZ + 1: "z_stag"}
            if field.ndim == 1:
                return ("Time", vertical[field.shape[0]])
            if field.ndim == 2:
                return ("Time", "y", "x")
            return ("Time", vertical[field.shape[0]], "y", "x")

        for name, field in state.items():
            file_name = f"{name}_2" if name in ("MU", "PH", "THM") else name
            ds.createVariable(file_name, "f8", dims(field))[0] = field
        for name in rebalance.BALANCED_FIELDS:
            shape = (NZ + 1, NY, NX) if name == "P_HYD_W" else (NZ, NY, NX)
            ds.createVariable(name, "f8", dims(np.zeros(shape)))[0] = np.zeros(shape)


def test_rebalance_updates_the_file(tmp_path: Path):
    state, al = make_state(np.random.default_rng(0))
    write_restart(tmp_path / "wrfrst", state, hypsometric_opt=2)

    with netCDF4.Dataset(tmp_path / "wrfrst", "r+") as ds:
        rebalance.rebalance(ds)

    expected = rebalance.compute_balanced_fields(state, P_TOP, use_theta_m=1)
    with netCDF4.Dataset(tmp_path / "wrfrst") as ds:
        np.testing.assert_allclose(ds["AL"][0], al, atol=1e-12)
        for name in rebalance.BALANCED_FIELDS:
            np.testing.assert_allclose(ds[name][0], expected[name])


def test_rebalance_rejects_other_hypsometric_options(tmp_path: Path):
    state, _ = make_state(np.random.default_rng(0))
    write_restart(tmp_path / "wrfrst", state, hypsometric_opt=1)

    with netCDF4.Dataset(tmp_path / "wrfrst", "r+") as ds:
        with pytest.raises(ValueError, match="hypsometric_opt = 2"):
            rebalance.rebalance(ds)
