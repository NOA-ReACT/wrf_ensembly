from importlib import resources
from pathlib import Path

import pytest

from wrf_ensembly import config
from wrf_ensembly.experiment.paths import ExperimentPaths


def make_cfg(scratch_root: str, name: str = "panorama_ens30_mtg") -> config.Config:
    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template)
    cfg.metadata.name = name
    cfg.directories.scratch_root = Path(scratch_root)
    return cfg


def test_placeholder_is_replaced_by_the_experiment_name(tmp_path):
    paths = ExperimentPaths(tmp_path, make_cfg("/lus/scratch/grtg/{{exp}}"))
    assert paths.scratch == Path("/lus/scratch/grtg/panorama_ens30_mtg")
    assert paths.scratch_restart == Path("/lus/scratch/grtg/panorama_ens30_mtg/restart")


def test_relative_scratch_with_placeholder_stays_inside_the_experiment(tmp_path):
    paths = ExperimentPaths(tmp_path, make_cfg("scratch/{{exp}}"))
    assert paths.scratch == tmp_path / "scratch" / "panorama_ens30_mtg"


def test_scratch_without_placeholder_is_unchanged(tmp_path):
    paths = ExperimentPaths(tmp_path, make_cfg("/lus/scratch/grtg/fixed"))
    assert paths.scratch == Path("/lus/scratch/grtg/fixed")


@pytest.mark.parametrize("name", ["", "a/b", ".."])
def test_placeholder_needs_a_plain_experiment_name(tmp_path, name):
    with pytest.raises(ValueError, match="metadata.name"):
        ExperimentPaths(tmp_path, make_cfg("/lus/scratch/{{exp}}", name))
