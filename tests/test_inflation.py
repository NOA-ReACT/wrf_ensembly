from importlib import resources
from pathlib import Path

from wrf_ensembly import config
from wrf_ensembly.experiment.inflation import InflationConfig

NAMELIST = {"filter_nml": {"inf_flavor": [2, 0]}}


def make_inflation(tmp_path: Path, use_inflation: bool = True) -> InflationConfig:
    """An InflationConfig with prior inflation, rooted at a temporary directory"""

    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template)
    cfg.assimilation.use_inflation = use_inflation
    cfg.dart_namelist = NAMELIST

    data_dir = tmp_path / "data_inflation"
    work_dir = tmp_path / "dart_work"
    data_dir.mkdir()
    work_dir.mkdir()
    return InflationConfig.from_config(cfg, data_dir, work_dir)


def test_leftover_links_are_removed_when_not_restarting(tmp_path):
    inflation = make_inflation(tmp_path)

    # A link left behind by another experiment's filter run
    other = tmp_path / "other_experiment_output_priorinf_mean.nc"
    other.write_text("keep me")
    link = inflation.dart_work_dir / "input_priorinf_mean.nc"
    link.symlink_to(other)

    namelist = inflation.prepare_cycle(NAMELIST, 0)

    assert namelist["filter_nml"]["inf_initial_from_restart"] == [False, False]
    assert not link.is_symlink()
    assert other.read_text() == "keep me"


def test_leftover_links_are_removed_with_inflation_off(tmp_path):
    inflation = make_inflation(tmp_path, use_inflation=False)

    other = tmp_path / "other_experiment_output_postinf_sd.nc"
    other.write_text("keep me")
    link = inflation.dart_work_dir / "input_postinf_sd.nc"
    link.symlink_to(other)

    inflation.prepare_cycle(NAMELIST, 3)

    assert not link.is_symlink()
    assert other.read_text() == "keep me"


def test_regular_input_files_are_kept(tmp_path):
    inflation = make_inflation(tmp_path)

    manual = inflation.dart_work_dir / "input_priorinf_mean.nc"
    manual.write_text("put here by hand")

    inflation.prepare_cycle(NAMELIST, 0)

    assert manual.read_text() == "put here by hand"
