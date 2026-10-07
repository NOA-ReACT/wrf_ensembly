from importlib import resources
from pathlib import Path

from wrf_ensembly import config, experiment


def make_experiment(path: Path, settings: str) -> experiment.Experiment:
    """An experiment from the default template (24 cycles, 20 members) in restart mode"""

    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    text = template.read_text()
    text = text.replace(
        "[assimilation]\n", f'[assimilation]\ncycling_mode = "restart"\n{settings}\n', 1
    )
    path.mkdir()
    (path / "config.toml").write_text(text)
    exp = experiment.Experiment(path)
    exp.paths.create_directories()
    exp.state.initialize()
    return exp


def write_restart_files(exp: experiment.Experiment, cycles: range) -> None:
    """Empty stand-ins for the restart files each member writes at the end of a cycle,
    plus one extra (as from a forecast extension)"""

    for cycle_i in cycles:
        cycle = exp.cycles[cycle_i]
        for member_i in range(2):
            directory = exp.paths.scratch_restart_path(cycle_i, member_i)
            directory.mkdir(parents=True)
            (directory / f"wrfrst_d01_{cycle.end:%Y-%m-%d_%H:%M:%S}").touch()
            (
                directory
                / f"wrfrst_d01_{exp.cycles[cycle_i + 1].end:%Y-%m-%d_%H:%M:%S}"
            ).touch()


def remaining(exp: experiment.Experiment) -> dict[int, int]:
    """Restart files left per cycle, member 0"""

    return {
        int(d.name.removeprefix("cycle_")): len(
            list((d / "member_00").glob("wrfrst_*"))
        )
        for d in sorted(exp.paths.scratch_restart.glob("cycle_*"))
    }


def test_cleanup_keeps_what_the_current_cycle_starts_from(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", "")
    write_restart_files(exp, range(5))

    exp.current_cycle_i = 5
    exp.clean_restart_files()

    # Cycle 5 starts from cycle 4's end-of-cycle file, older cycles are gone
    assert remaining(exp) == {4: 1}


def test_cleanup_keeps_listed_cycles(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", "keep_restart_files_for_cycles = [1, 2]")
    write_restart_files(exp, range(5))

    exp.current_cycle_i = 5
    exp.clean_restart_files()

    assert remaining(exp) == {1: 2, 2: 2, 4: 1}


def test_cleanup_can_be_turned_off(tmp_path: Path):
    exp = make_experiment(tmp_path / "exp", "keep_restart_files = true")
    write_restart_files(exp, range(5))

    exp.current_cycle_i = 5
    exp.clean_restart_files()

    assert remaining(exp) == {i: 2 for i in range(5)}


def test_namelist_differences():
    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    ours = config.read_config(template, inject_environment=False)
    theirs = config.read_config(template, inject_environment=False)
    theirs.wrf_namelist["physics"]["mp_physics"] = 99
    theirs.wrf_namelist["chem"] = {"dust_alpha": 0.5}
    theirs.wrf_namelist_per_member = {"1": {"chem": {"dust_alpha": 0.7}}}

    differences = config.wrf_namelist_differences(ours, theirs)

    assert differences == [
        "wrf_namelist.chem.dust_alpha: (unset) / 0.5",
        f"wrf_namelist.physics.mp_physics: {ours.wrf_namelist['physics']['mp_physics']} / 99",
        "wrf_namelist_per_member.1.chem.dust_alpha: (unset) / 0.7",
    ]
    assert config.wrf_namelist_differences(ours, ours) == []
