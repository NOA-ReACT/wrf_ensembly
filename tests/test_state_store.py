import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from pathlib import Path

import pytest

from wrf_ensembly import config
from wrf_ensembly.experiment.paths import ExperimentPaths
from wrf_ensembly.experiment.state_store import ExperimentState

N_MEMBERS = 40


def make_state(tmp_path: Path, n_members: int = N_MEMBERS) -> ExperimentState:
    """An ExperimentState rooted at a temporary experiment directory"""

    template = resources.files("wrf_ensembly.config_templates").joinpath("default.toml")
    cfg = config.read_config(template)
    cfg.assimilation.n_members = n_members

    paths = ExperimentPaths(tmp_path, cfg)
    state = ExperimentState(paths, n_members)
    state.initialize()
    return state


def test_initialize_is_idempotent_and_starts_at_cycle_zero(tmp_path):
    state = make_state(tmp_path)
    assert state.get_current_cycle() == 0

    state.set_current_cycle(4)
    state.initialize()
    assert state.get_current_cycle() == 4


def test_current_cycle_round_trips(tmp_path):
    state = make_state(tmp_path)

    state.set_current_cycle(7)
    assert state.get_current_cycle() == 7

    # A fresh store reading the same directory sees the same value
    assert make_state(tmp_path).get_current_cycle() == 7


def test_member_advancement_is_per_cycle(tmp_path):
    state = make_state(tmp_path)

    state.set_member_advanced(0, 3)
    assert state.get_advanced_members(0) == {3}

    # A late write for an old cycle must not leak into the next one
    assert state.get_advanced_members(1) == set()


def test_runtime_statistics_round_trip(tmp_path):
    state = make_state(tmp_path)
    start = dt.datetime(2026, 8, 10, 12, 0, 0)
    end = dt.datetime(2026, 8, 10, 12, 40, 0)

    state.set_member_advanced(2, 5, start=start, end=end, duration_s=2400)

    record = state.get_member(2, 5)
    assert record is not None
    assert record.advanced
    assert record.runtime is not None
    assert record.runtime.cycle == 2
    assert record.runtime.start == start
    assert record.runtime.duration_s == 2400


def test_advancement_without_runtime_statistics(tmp_path):
    """This is what `status reconcile` writes, since timings cannot be recovered"""

    state = make_state(tmp_path)
    state.set_member_advanced(0, 1)

    record = state.get_member(0, 1)
    assert record is not None
    assert record.advanced
    assert record.runtime is None
    assert state.get_all_runtime_statistics() == []


def test_get_all_runtime_statistics_is_sorted(tmp_path):
    state = make_state(tmp_path)
    start = dt.datetime(2026, 8, 10, 12, 0, 0)

    for cycle in (1, 0):
        for member in (2, 0):
            state.set_member_advanced(
                cycle, member, start=start, end=start, duration_s=10
            )

    keys = [(stat.cycle, member_i) for stat, member_i in state.get_all_runtime_statistics()]
    assert keys == [(0, 0), (0, 2), (1, 0), (1, 2)]


def test_clear_runtime_statistics_keeps_advancement(tmp_path):
    state = make_state(tmp_path)
    start = dt.datetime(2026, 8, 10, 12, 0, 0)
    state.set_member_advanced(0, 1, start=start, end=start, duration_s=10)

    state.clear_runtime_statistics()

    assert state.get_all_runtime_statistics() == []
    assert state.get_advanced_members(0) == {1}


def test_clear_member_and_cycle(tmp_path):
    state = make_state(tmp_path)
    for i in range(3):
        state.set_member_advanced(0, i)

    state.clear_member(0, 1)
    assert state.get_advanced_members(0) == {0, 2}

    state.clear_cycle_members(0)
    assert state.get_advanced_members(0) == set()


def test_markers(tmp_path):
    state = make_state(tmp_path)

    assert not state.is_marker_set(0, "filter_complete")
    state.set_marker(0, "filter_complete")
    assert state.is_marker_set(0, "filter_complete")

    # Markers are per cycle
    assert not state.is_marker_set(1, "filter_complete")

    state.clear_marker(0, "filter_complete")
    assert not state.is_marker_set(0, "filter_complete")


def test_optional_operations(tmp_path):
    state = make_state(tmp_path)

    assert not state.is_optional_operation_complete(0, "apply_perturbations")
    state.mark_optional_operation_complete(0, "apply_perturbations")
    assert state.is_optional_operation_complete(0, "apply_perturbations")
    assert not state.is_optional_operation_complete(1, "apply_perturbations")


def test_reset_clears_everything(tmp_path):
    state = make_state(tmp_path)
    state.set_member_advanced(0, 0)
    state.set_marker(0, "filter_complete")
    state.mark_optional_operation_complete(0, "apply_perturbations")
    state.set_current_cycle(3)

    state.reset()

    assert state.get_current_cycle() == 0
    assert state.get_advanced_members(0) == set()
    assert not state.is_marker_set(0, "filter_complete")
    assert not state.is_optional_operation_complete(0, "apply_perturbations")


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        "{not json",
        '{"advanced": ',
        # Valid JSON, but not an object
        "[1, 2, 3]",
        "42",
        '"advanced"',
        "null",
    ],
)
def test_unreadable_member_file_is_treated_as_not_advanced(tmp_path, content):
    """One bad file must never be able to wedge an experiment"""

    state = make_state(tmp_path)
    state.set_member_advanced(0, 0)
    state.set_member_advanced(0, 1)

    path = state.paths.member_status_path(0, 1)
    path.write_text(content)

    assert state.get_member(0, 1) is None
    assert state.get_advanced_members(0) == {0}


def test_member_files_beyond_the_ensemble_size_are_ignored(tmp_path):
    """Left over from a run with a larger ensemble"""

    state = make_state(tmp_path, n_members=4)
    for i in range(6):
        state.set_member_advanced(0, i)

    assert state.get_advanced_members(0) == {0, 1, 2, 3}
    assert state.count_advanced(0) == 4


def test_missing_experiment_file_reads_as_cycle_zero(tmp_path):
    state = make_state(tmp_path)
    state.set_current_cycle(5)
    state.paths.status_experiment_file.unlink()

    assert state.get_current_cycle() == 0


def test_invalid_current_cycle_falls_back_to_zero(tmp_path):
    state = make_state(tmp_path)
    state.paths.status_experiment_file.write_text(json.dumps({"current_cycle": "five"}))

    assert state.get_current_cycle() == 0


@pytest.mark.parametrize("content", ["[1, 2, 3]", "42", "null", "{oops"])
def test_unreadable_experiment_file_falls_back_to_zero(tmp_path, content):
    state = make_state(tmp_path)
    state.set_current_cycle(3)
    state.paths.status_experiment_file.write_text(content)

    assert state.get_current_cycle() == 0

    # And it can be written over without raising
    state.set_current_cycle(2)
    assert state.get_current_cycle() == 2


def test_count_advanced_returns_immediately_when_complete(tmp_path):
    state = make_state(tmp_path, n_members=2)
    state.set_member_advanced(0, 0)
    state.set_member_advanced(0, 1)

    # Would block for `timeout` if it did not notice the count was already reached
    assert state.count_advanced(0, expect=2, timeout=30.0) == 2


def _advance(args) -> int:
    """Runs in a separate process: one member recording its own advancement"""

    experiment_path, member_i = args
    state = make_state(Path(experiment_path))
    state.set_member_advanced(
        0,
        member_i,
        start=dt.datetime(2026, 8, 10, 12, 0, 0),
        end=dt.datetime(2026, 8, 10, 12, 40, 0),
        duration_s=2400,
    )
    return member_i


def test_concurrent_member_writes(tmp_path):
    """
    The case that breaks sqlite on a network filesystem: every member of the ensemble
    recording its result at the same time, from separate processes.
    """

    make_state(tmp_path)

    with ProcessPoolExecutor(max_workers=8) as executor:
        written = set(
            executor.map(_advance, [(str(tmp_path), i) for i in range(N_MEMBERS)])
        )

    assert written == set(range(N_MEMBERS))

    state = make_state(tmp_path)
    assert state.get_advanced_members(0) == set(range(N_MEMBERS))
    assert len(state.get_all_runtime_statistics()) == N_MEMBERS

    # No temporary files left behind
    leftovers = list(state.paths.cycle_members_path(0).glob(".*"))
    assert leftovers == []
