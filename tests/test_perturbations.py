from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wrf_ensembly import config, perturbations

N_MEMBERS = 3


def make_config(
    cycling_mode: str, variables: dict[str, config.PerturbationVariableConfig]
) -> config.Config:
    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template, inject_environment=False)
    cfg.assimilation.n_members = N_MEMBERS
    cfg.assimilation.cycling_mode = cycling_mode  # type: ignore
    cfg.assimilation.cycled_variables = ["U", "V", "THM"]
    cfg.perturbations = config.PerturbationsConfig(variables=variables, seed=1)
    return cfg


def noise(every_cycle: bool = True) -> config.PerturbationVariableConfig:
    return config.PerturbationVariableConfig(
        operation="add", mean=0.0, sd=0.5, perturb_every_cycle=every_cycle
    )


def parameter() -> config.PerturbationVariableConfig:
    return config.PerturbationVariableConfig(
        operation="multiply", kind="parameter", sd=0.2
    )


def test_constant_field_every_cycle_is_read_as_a_parameter(
    caplog: pytest.LogCaptureFixture,
):
    cfg = config.PerturbationsConfig(
        variables={
            "DUST_EMIS_WEIGHT": config.PerturbationVariableConfig(
                operation="multiply",
                perturb_every_cycle=True,
                different_field_every_cycle=False,
            ),
            "THM": config.PerturbationVariableConfig(
                operation="add",
                perturb_every_cycle=True,
                different_field_every_cycle=True,
            ),
        }
    )

    weight, thm = cfg.variables["DUST_EMIS_WEIGHT"], cfg.variables["THM"]
    assert weight.kind == "parameter" and not weight.perturb_every_cycle
    assert thm.kind == "state" and thm.perturb_every_cycle
    assert 'reading this variable as kind = "parameter"' in caplog.text


@pytest.mark.parametrize(
    "settings", [{"perturb_every_cycle": True}, {"midcycle_taper_width": 5}]
)
def test_parameters_reject_per_cycle_settings(settings: dict):
    with pytest.raises(ValueError, match="constant field"):
        config.PerturbationsConfig(
            variables={
                "DUST_EMIS_WEIGHT": config.PerturbationVariableConfig(
                    operation="multiply", kind="parameter", **settings
                )
            }
        )


def test_applied_at_cycle():
    variables = {
        "THM": noise(),
        "U": noise(every_cycle=False),
        "DUST_EMIS_WEIGHT": parameter(),
        "V": parameter(),  # cycled, so carried over like in restart mode
    }

    wrfinput = make_config("wrfinput", variables)
    assert perturbations.applied_at_cycle(wrfinput, 0) == list(variables)
    assert perturbations.applied_at_cycle(wrfinput, 1) == ["THM", "DUST_EMIS_WEIGHT"]

    restart = make_config("restart", variables)
    assert perturbations.applied_at_cycle(restart, 0) == list(variables)
    assert perturbations.applied_at_cycle(restart, 3) == ["THM"]


def generate(cfg: config.Config, cycle_i: int, tmp_path: Path) -> Path:
    ic_path = tmp_path / "wrfinput_d01"
    if not ic_path.exists():
        xr.Dataset(
            {
                "THM": (("Time", "z", "y", "x"), np.zeros((1, 2, 6, 5))),
                "DUST_EMIS_WEIGHT": (("Time", "y", "x"), np.ones((1, 6, 5))),
            }
        ).to_netcdf(ic_path)

    perturbations.generate_perturbations_for_cycle(
        cycle_i=cycle_i,
        perturbations_cfg=cfg.perturbations,
        reapplied_parameters=perturbations.reapplied_parameters(cfg),
        n_members=N_MEMBERS,
        experiment_name="test",
        ic_path=ic_path,
        diag_dir=tmp_path,
    )
    return tmp_path / "perturbations" / f"perts_cycle_{cycle_i}.nc"


def test_restart_mode_only_perturbs_parameters_once(tmp_path: Path):
    cfg = make_config("restart", {"DUST_EMIS_WEIGHT": parameter()})

    assert generate(cfg, 0, tmp_path).exists()
    # A file from an earlier configuration is removed, not kept around
    stale = tmp_path / "perturbations" / "perts_cycle_1.nc"
    stale.symlink_to("perts_cycle_0.nc")
    assert not generate(cfg, 1, tmp_path).is_symlink()
    assert not stale.exists()


def test_wrfinput_mode_reapplies_the_first_cycles_parameter_field(tmp_path: Path):
    cfg = make_config("wrfinput", {"DUST_EMIS_WEIGHT": parameter()})

    generate(cfg, 0, tmp_path)
    second = generate(cfg, 1, tmp_path)
    assert second.is_symlink() and second.resolve().name == "perts_cycle_0.nc"


def test_noise_gets_a_new_field_next_to_the_reused_parameter(tmp_path: Path):
    cfg = make_config("wrfinput", {"THM": noise(), "DUST_EMIS_WEIGHT": parameter()})

    with xr.open_dataset(generate(cfg, 0, tmp_path)) as first:
        first = first.load()
    with xr.open_dataset(generate(cfg, 1, tmp_path)) as second:
        second = second.load()

    xr.testing.assert_equal(second["DUST_EMIS_WEIGHT"], first["DUST_EMIS_WEIGHT"])
    assert not np.allclose(second["THM"], first["THM"])
