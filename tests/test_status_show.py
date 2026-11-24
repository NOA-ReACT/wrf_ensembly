"""
`status show` is run by hand, over and over, on experiments with hundreds of cycles
that live on Lustre, where every status file opened is a round trip to the metadata
server. It must only read the current cycle.
"""

from importlib import resources
from pathlib import Path

import pytest
from click.testing import CliRunner

from wrf_ensembly import experiment
from wrf_ensembly.cli import cli
from wrf_ensembly.experiment import ExperimentState, StateTransition

N_MEMBERS = 4
CURRENT_CYCLE = 6
ADVANCED = (0, 1)
"""Members that have advanced in the current cycle"""


@pytest.fixture
def exp(tmp_path: Path) -> experiment.Experiment:
    """An experiment on cycle 6 with two of its four members advanced"""

    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    text = template.read_text().replace("n_members = 20", f"n_members = {N_MEMBERS}")
    path = tmp_path / "exp"
    path.mkdir()
    (path / "config.toml").write_text(text)
    exp = experiment.Experiment(path)
    exp.paths.create_directories()
    exp.state.initialize()

    for cycle in range(CURRENT_CYCLE):
        for member in range(N_MEMBERS):
            exp.state.set_member_advanced(cycle, member)
        for transition in StateTransition:
            exp.state.set_marker(cycle, transition.value)
    for member in ADVANCED:
        exp.state.set_member_advanced(CURRENT_CYCLE, member)
    exp.state.set_current_cycle(CURRENT_CYCLE)
    return exp


@pytest.fixture
def status_reads(monkeypatch) -> list[Path]:
    """Every status file read from here on"""

    reads: list[Path] = []
    read_json = ExperimentState._read_json

    def recording_read_json(self, path):
        reads.append(path)
        return read_json(self, path)

    monkeypatch.setattr(ExperimentState, "_read_json", recording_read_json)
    return reads


def show(exp: experiment.Experiment, *args: str) -> str:
    result = CliRunner().invoke(
        cli, [str(exp.paths.experiment_path), "status", "show", *args]
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_show_reads_only_the_current_cycle(exp, status_reads):
    output = show(exp)

    assert "advancing_members" in output
    assert "Runtime Stats" not in output

    current = exp.paths.cycle_status_path(CURRENT_CYCLE)
    past = [
        p
        for p in status_reads
        if exp.paths.status_cycles in p.parents and current not in p.parents
    ]
    assert past == []

    # Each advanced member's file once to load the experiment and once for the cycle
    # state, not once more for every row of the table
    member_reads = [p for p in status_reads if current in p.parents]
    assert len(member_reads) == 2 * len(ADVANCED)


def test_show_of_a_complete_ensemble(exp):
    for member in range(N_MEMBERS):
        exp.state.set_member_advanced(CURRENT_CYCLE, member)
    output = show(exp)
    assert "members_advanced" in output
    assert "✗" not in output.split("Member Status")[1]

    exp.state.set_marker(CURRENT_CYCLE, StateTransition.FILTER_COMPLETE.value)
    assert "filter_complete" in show(exp)
