from importlib import resources
from pathlib import Path

from wrf_ensembly import config
from wrf_ensembly.experiment.inflation import InflationConfig
from wrf_ensembly.experiment.paths import ExperimentPaths
from wrf_ensembly.experiment.state_machine import StateTransition
from wrf_ensembly.experiment.state_store import ExperimentState

NAMELIST = {"filter_nml": {"inf_flavor": [2, 0]}}


def make_inflation(tmp_path: Path, use_inflation: bool = True) -> InflationConfig:
    """An InflationConfig with prior inflation, rooted at a temporary directory"""

    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template)
    cfg.assimilation.use_inflation = use_inflation
    cfg.dart_namelist = NAMELIST

    state = ExperimentState(ExperimentPaths(tmp_path, cfg), cfg.assimilation.n_members)
    state.initialize()

    data_dir = tmp_path / "data_inflation"
    work_dir = tmp_path / "dart_work"
    data_dir.mkdir()
    work_dir.mkdir()
    return InflationConfig.from_config(cfg, data_dir, work_dir, state)


def stash(inflation: InflationConfig, cycle_i: int, filter_complete: bool = True):
    """Pretend filter ran for a cycle and its inflation files were stashed"""

    for filename in inflation.active_files:
        (inflation.dart_work_dir / filename).write_text(f"cycle {cycle_i}")
    inflation.stash_restart_files(cycle_i)
    if filter_complete:
        inflation.state.set_marker(cycle_i, StateTransition.FILTER_COMPLETE.value)


def linked_cycle(inflation: InflationConfig) -> str:
    return (inflation.dart_work_dir / "input_priorinf_mean.nc").read_text()


def test_latest_filtered_cycle_is_used(tmp_path):
    inflation = make_inflation(tmp_path)
    stash(inflation, 0)
    stash(inflation, 2)

    namelist = inflation.prepare_cycle(NAMELIST, 4)

    assert namelist["filter_nml"]["inf_initial_from_restart"] == [True, False]
    assert namelist["filter_nml"]["inf_sd_initial_from_restart"] == [True, False]
    assert linked_cycle(inflation) == "cycle 2"


def test_files_from_before_a_reset_are_ignored(tmp_path):
    inflation = make_inflation(tmp_path)

    # The earlier run filtered at every cycle
    for cycle_i in range(7):
        stash(inflation, cycle_i)
    inflation.state.reset()

    # This run only filters at cycle 3, so cycle 7 must not take the old cycle 6 files
    stash(inflation, 3)
    inflation.prepare_cycle(NAMELIST, 7)
    assert linked_cycle(inflation) == "cycle 3"


def test_no_filtered_cycle_means_no_restart(tmp_path):
    inflation = make_inflation(tmp_path)
    stash(inflation, 0, filter_complete=False)

    namelist = inflation.prepare_cycle(NAMELIST, 1)

    assert namelist["filter_nml"]["inf_initial_from_restart"] == [False, False]
    assert not (inflation.dart_work_dir / "input_priorinf_mean.nc").exists()


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
