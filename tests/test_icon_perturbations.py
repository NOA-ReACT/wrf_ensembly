"""
`[perturbations]` with ICON: fields on the triangular grid, applied to each member's
copy of the IFS initial conditions.
"""

from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr
from scipy.spatial import cKDTree
from test_icon_model import make_experiment, template_text
from test_icon_regrid import limited_area_grid

from wrf_ensembly import config
from wrf_ensembly.icon import model as icon_model
from wrf_ensembly.icon import namelists, perturbations
from wrf_ensembly.icon.regrid import IconGrid

NLEV = 5
PERTURB_T = """
[perturbations.variables.T]
operation = "add"
mean = 0.0
sd = 0.5
gaussian_sigma = 2.0
"""


@pytest.fixture
def grid(monkeypatch: pytest.MonkeyPatch) -> IconGrid:
    """A small limited-area grid, which is what any grid file gives"""

    grid = limited_area_grid()
    monkeypatch.setattr(IconGrid, "from_file", classmethod(lambda cls, path: grid))
    return grid


def write_initial_conditions(path: Path, ncells: int):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("lev", NLEV)
        ds.createDimension("cell", ncells)
        ds.createVariable("T", "f4", ("time", "lev", "cell"))[:] = 280.0
        ds.createVariable("QV", "f4", ("time", "lev", "cell"))[:] = 0.01


def perturbed_experiment(tmp_path: Path, grid: IconGrid, extra: str = PERTURB_T):
    """An experiment with real initial conditions, set up again to pick them up"""

    exp = make_experiment(tmp_path / "exp", extra=extra, members=3)
    assert exp.cfg.icon is not None
    write_initial_conditions(exp.cfg.icon.initial_conditions, grid.ncells)
    for member_i in range(3):
        icon_model.setup_member(exp, member_i)
    return exp


def test_fields_are_smooth_and_differ_between_members(grid: IconGrid):
    np.random.seed(1)
    fields = perturbations.generate_perturbation_ensemble(
        3,
        (1, NLEV, grid.ncells),
        mean=1.0,
        sd=0.2,
        sigma=2.0,
        grid_file=Path("grid.nc"),
    )

    assert fields.shape == (3, 1, NLEV, grid.ncells)
    for member in fields:
        assert member.mean() == pytest.approx(1.0)
        assert member.std() == pytest.approx(0.2)

    # Neighbouring cells get nearly the same value, members and distant cells do not
    xyz = np.stack(
        [
            np.cos(grid.clat) * np.cos(grid.clon),
            np.cos(grid.clat) * np.sin(grid.clon),
            np.sin(grid.clat),
        ],
        axis=1,
    )
    _, neighbour = cKDTree(xyz).query(xyz, k=[2])
    level = fields[0, 0, 2]
    assert np.corrcoef(level, level[neighbour[:, 0]])[0, 1] > 0.8
    assert abs(np.corrcoef(level, fields[1, 0, 2])[0, 1]) < 0.5

    np.random.seed(1)
    again = perturbations.generate_perturbation_ensemble(
        3,
        (1, NLEV, grid.ncells),
        mean=1.0,
        sd=0.2,
        sigma=2.0,
        grid_file=Path("grid.nc"),
    )
    assert np.array_equal(fields, again)


def test_limits_and_unsupported_options(grid: IconGrid):
    fields = perturbations.generate_perturbation_ensemble(
        2, (grid.ncells,), mean=1.0, sd=1.0, min_value=0.5, max_value=1.5,
        grid_file=Path("grid.nc"),
    )  # fmt: skip
    assert fields.min() == 0.5 and fields.max() == 1.5

    with pytest.raises(ValueError, match="boundary"):
        perturbations.generate_perturbation_ensemble(
            2, (grid.ncells,), 0.0, 1.0, boundary=2, grid_file=Path("grid.nc")
        )
    with pytest.raises(ValueError, match="cells"):
        perturbations.generate_perturbation_ensemble(
            2, (grid.ncells + 1,), 0.0, 1.0, grid_file=Path("grid.nc")
        )


def test_generate_and_apply(tmp_path: Path, grid: IconGrid):
    exp = perturbed_experiment(tmp_path, grid)
    assert exp.cfg.icon is not None

    exp.model.generate_perturbations(0)
    pert_file = exp.paths.data_diag / "perturbations" / "perts_cycle_0.nc"
    with xr.open_dataset(pert_file) as perts:
        assert perts["T"].dims == ("member", "time", "lev", "cell")
        fields = perts["T"].to_numpy()

    for member_i in range(3):
        exp.model.apply_perturbations(member_i)

    for member_i in range(3):
        initial = exp.paths.member_path(member_i) / namelists.IFS_LINK
        assert not initial.is_symlink()
        with netCDF4.Dataset(initial) as ds:
            np.testing.assert_allclose(ds["T"][:], 280.0 + fields[member_i], atol=1e-3)
            # Only what is in [perturbations] changes
            assert np.all(ds["QV"][:] == np.float32(0.01))

    # The shared file stays as it was
    with netCDF4.Dataset(exp.cfg.icon.initial_conditions) as ds:
        assert np.all(ds["T"][:] == 280.0)


def test_multiply(tmp_path: Path, grid: IconGrid):
    exp = perturbed_experiment(
        tmp_path,
        grid,
        '[perturbations.variables.QV]\noperation = "multiply"\nmean = 1.0\nsd = 0.1\n',
    )
    exp.model.generate_perturbations(0)
    exp.model.apply_perturbations(1)

    pert_file = exp.paths.data_diag / "perturbations" / "perts_cycle_0.nc"
    with (
        xr.open_dataset(pert_file) as perts,
        netCDF4.Dataset(exp.paths.member_path(1) / namelists.IFS_LINK) as ds,
    ):
        np.testing.assert_allclose(
            ds["QV"][:], 0.01 * perts["QV"].sel(member=1).to_numpy(), rtol=1e-3
        )


def test_only_the_first_cycle_is_perturbed(tmp_path: Path, grid: IconGrid):
    exp = perturbed_experiment(tmp_path, grid)
    exp.model.generate_perturbations(0)
    assert list(exp.model.generate_perturbations_for_cycles(range(1, 4), 1)) == []

    exp.current_cycle_i = 1
    exp.model.apply_perturbations(0)
    with netCDF4.Dataset(exp.paths.member_path(0) / namelists.IFS_LINK) as ds:
        assert np.all(ds["T"][:] == 280.0)


def test_apply_needs_a_copy_and_the_fields(tmp_path: Path, grid: IconGrid):
    exp = perturbed_experiment(tmp_path, grid)
    with pytest.raises(FileNotFoundError, match="generate-perturbations"):
        exp.model.apply_perturbations(0)

    # Set up before [perturbations] was added: the member has a link
    exp.model.generate_perturbations(0)
    initial = exp.paths.member_path(0) / namelists.IFS_LINK
    initial.unlink()
    initial.symlink_to(exp.cfg.icon.initial_conditions)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="ensemble setup"):
        exp.model.apply_perturbations(0)


def test_without_perturbations_members_share_the_file(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp")
    assert (exp.paths.member_path(0) / namelists.IFS_LINK).is_symlink()


@pytest.mark.parametrize(
    "setting,message",
    [
        ('kind = "parameter"', "parameter"),
        ("perturb_every_cycle = true", "perturb_every_cycle"),
        ("boundary = 3", "boundary"),
    ],
)
def test_config_check_rejects(setting, message):
    cfg = config.Config.from_toml(template_text() + PERTURB_T + setting + "\n")
    with pytest.raises(ValueError, match=message):
        cfg.check()


def test_config_check_accepts_first_cycle_perturbations():
    config.Config.from_toml(template_text() + PERTURB_T).check()
