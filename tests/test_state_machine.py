from importlib import resources
from pathlib import Path

import pytest

from wrf_ensembly import config
from wrf_ensembly.experiment.paths import ExperimentPaths
from wrf_ensembly.experiment.state_machine import (
    CycleState,
    ExperimentStateError,
    ExperimentStateMachine,
    StateTransition,
)
from wrf_ensembly.experiment.state_store import ExperimentState

N_MEMBERS = 4
N_CYCLES = 6


def make_machine(tmp_path: Path) -> ExperimentStateMachine:
    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template)
    cfg.assimilation.n_members = N_MEMBERS

    state = ExperimentState(ExperimentPaths(tmp_path, cfg), N_MEMBERS)
    state.initialize()
    return ExperimentStateMachine(
        state=state, n_cycles=N_CYCLES, n_members=N_MEMBERS, current_cycle_idx=0
    )


def advance_all(machine: ExperimentStateMachine, cycle: int):
    for i in range(N_MEMBERS):
        machine.state.set_member_advanced(cycle, i)


def test_state_is_derived_from_member_files(tmp_path):
    machine = make_machine(tmp_path)
    cycle = machine.get_cycle(0)

    assert cycle.current_state == CycleState.INITIALIZED

    machine.state.set_member_advanced(0, 0)
    assert cycle.current_state == CycleState.ADVANCING_MEMBERS

    advance_all(machine, 0)
    assert cycle.current_state == CycleState.MEMBERS_ADVANCED


def test_markers_take_precedence_over_member_count(tmp_path):
    machine = make_machine(tmp_path)
    advance_all(machine, 0)
    cycle = machine.get_cycle(0)

    cycle.transition(StateTransition.FILTER_COMPLETE)
    assert cycle.current_state == CycleState.FILTER_COMPLETE

    cycle.transition(StateTransition.ANALYSIS_COMPLETE)
    assert cycle.current_state == CycleState.ANALYSIS_COMPLETE

    cycle.transition(StateTransition.CYCLE_COMPLETE)
    assert cycle.current_state == CycleState.CYCLE_COMPLETE


def test_state_is_recomputed_not_cached(tmp_path):
    """
    The old machine cached the state on the instance, so a member advancing in another
    process went unnoticed until the next command started.
    """

    machine = make_machine(tmp_path)
    cycle = machine.get_cycle(0)
    assert cycle.current_state == CycleState.INITIALIZED

    advance_all(machine, 0)

    # Same object, no reload
    assert cycle.current_state == CycleState.MEMBERS_ADVANCED


def test_filter_rejected_until_every_member_advanced(tmp_path):
    machine = make_machine(tmp_path)

    can_run, error = machine.can_run_filter(0)
    assert not can_run

    for i in range(N_MEMBERS - 1):
        machine.state.set_member_advanced(0, i)
    can_run, error = machine.can_run_filter(0)
    assert not can_run
    assert "advancing_members" in error

    machine.state.set_member_advanced(0, N_MEMBERS - 1)
    can_run, _ = machine.can_run_filter(0)
    assert can_run


def test_invalid_transition_raises(tmp_path):
    machine = make_machine(tmp_path)
    cycle = machine.get_cycle(0)

    with pytest.raises(ExperimentStateError):
        cycle.transition(StateTransition.FILTER_COMPLETE)

    advance_all(machine, 0)
    with pytest.raises(ExperimentStateError):
        cycle.transition(StateTransition.ANALYSIS_COMPLETE)


def test_analysis_requires_filter(tmp_path):
    machine = make_machine(tmp_path)
    advance_all(machine, 0)

    can_run, _ = machine.can_run_analysis(0)
    assert not can_run

    machine.get_cycle(0).transition(StateTransition.FILTER_COMPLETE)
    can_run, _ = machine.can_run_analysis(0)
    assert can_run


def test_cycling_uses_forecast_when_filter_skipped(tmp_path):
    machine = make_machine(tmp_path)
    advance_all(machine, 0)

    can_cycle, use_forecast, _ = machine.can_cycle_to_next(0)
    assert can_cycle
    assert use_forecast

    machine.get_cycle(0).transition(StateTransition.FILTER_COMPLETE)
    machine.get_cycle(0).transition(StateTransition.ANALYSIS_COMPLETE)
    can_cycle, use_forecast, _ = machine.can_cycle_to_next(0)
    assert can_cycle
    assert not use_forecast


def test_can_advance_member(tmp_path):
    machine = make_machine(tmp_path)

    assert machine.can_advance_member(0, 0)[0]

    machine.state.set_member_advanced(0, 0)
    assert machine.can_advance_member(0, 1)[0]

    advance_all(machine, 0)
    assert not machine.can_advance_member(0, 0)[0]

    # Only the current cycle can be advanced
    assert not machine.can_advance_member(3, 0)[0]


def test_non_current_cycles_report_their_own_state(tmp_path):
    """
    Regression: state used to be one scalar for the whole experiment, so every cycle
    other than the current one always looked INITIALIZED. `reset-cycle --cycle N` relied
    on this and silently did nothing.
    """

    machine = make_machine(tmp_path)

    advance_all(machine, 3)
    machine.get_cycle(3).transition(StateTransition.FILTER_COMPLETE)

    machine.current_cycle_idx = 5
    assert machine.get_cycle(3).current_state == CycleState.FILTER_COMPLETE
    assert machine.get_cycle(4).current_state == CycleState.INITIALIZED
    assert machine.current_cycle.current_state == CycleState.INITIALIZED


def test_reset_only_touches_its_own_cycle(tmp_path):
    machine = make_machine(tmp_path)

    for cycle in (3, 5):
        advance_all(machine, cycle)
        machine.get_cycle(cycle).transition(StateTransition.FILTER_COMPLETE)

    machine.current_cycle_idx = 5
    machine.get_cycle(3).reset()

    assert machine.get_cycle(3).current_state == CycleState.INITIALIZED
    assert machine.get_cycle(5).current_state == CycleState.FILTER_COMPLETE


@pytest.mark.parametrize(
    "target",
    [
        CycleState.INITIALIZED,
        CycleState.MEMBERS_ADVANCED,
        CycleState.FILTER_COMPLETE,
        CycleState.ANALYSIS_COMPLETE,
        CycleState.CYCLE_COMPLETE,
    ],
)
def test_force_state(tmp_path, target):
    machine = make_machine(tmp_path)
    cycle = machine.get_cycle(0)

    # Start somewhere else entirely, to check the previous state is cleaned up
    advance_all(machine, 0)
    cycle.transition(StateTransition.FILTER_COMPLETE)
    cycle.transition(StateTransition.ANALYSIS_COMPLETE)

    cycle.force_state(target)
    assert cycle.current_state == target


def test_force_state_rejects_advancing_members(tmp_path):
    machine = make_machine(tmp_path)

    with pytest.raises(ExperimentStateError):
        machine.get_cycle(0).force_state(CycleState.ADVANCING_MEMBERS)


def test_required_actions_reports_progress(tmp_path):
    machine = make_machine(tmp_path)
    machine.state.set_member_advanced(0, 0)

    assert "1/4" in machine.get_cycle(0).get_required_actions()
