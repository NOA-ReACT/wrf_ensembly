"""
Formal state machine for Experiments, checking which actions are allowed at which
cycle states.

The purpose of this formality is to reduce errors caused by forgetting or double-running
commands.

The state of a cycle is not stored anywhere: it is derived, on demand, from the status
files on disk (see `state_store.py`). Member advancement in particular is computed by
counting the per-member files, which is what makes it safe for members to advance in
parallel on different nodes without any coordination. Only the three completion steps
(filter, analysis, cycle) are recorded, as marker files, so the transitions below map
one-to-one onto things that are actually written down.
"""

from dataclasses import dataclass
from enum import Enum

from .state_store import ExperimentState


class ExperimentStateError(Exception):
    """Raised when an operation is attempted in an invalid experiment state."""

    pass


class CycleState(Enum):
    """
    States for a single cycle in the assimilation workflow.
    """

    # Initial state - no work done
    INITIALIZED = "initialized"

    # Members are being advanced (some may be complete, some may be running)
    ADVANCING_MEMBERS = "advancing_members"

    # All members advanced, ready for filter
    MEMBERS_ADVANCED = "members_advanced"

    # Filter complete, ready for analysis
    FILTER_COMPLETE = "filter_complete"

    # Analysis complete, ready to cycle
    ANALYSIS_COMPLETE = "analysis_complete"

    # Cycle complete, can advance to next cycle
    CYCLE_COMPLETE = "cycle_complete"


class StateTransition(Enum):
    """
    Valid operations that trigger state transitions.

    There is one per marker file; the states before FILTER_COMPLETE are reached by
    advancing members, not by an explicit transition.
    """

    FILTER_COMPLETE = "filter_complete"
    ANALYSIS_COMPLETE = "analysis_complete"
    CYCLE_COMPLETE = "cycle_complete"


MARKER_NAMES = [t.value for t in StateTransition]
"""Names of every marker file a cycle can have"""

STEP_NAMES = {
    StateTransition.FILTER_COMPLETE: "filter",
    StateTransition.ANALYSIS_COMPLETE: "analysis",
    StateTransition.CYCLE_COMPLETE: "cycle",
}
"""The command each transition corresponds to, for error messages"""

STATES_IN_ORDER = [
    (StateTransition.FILTER_COMPLETE, CycleState.FILTER_COMPLETE),
    (StateTransition.ANALYSIS_COMPLETE, CycleState.ANALYSIS_COMPLETE),
    (StateTransition.CYCLE_COMPLETE, CycleState.CYCLE_COMPLETE),
]
"""The recorded states of a cycle, in the order they are reached"""


@dataclass
class StateTransitionRule:
    """Defines a valid transition between states."""

    from_state: CycleState
    transition: StateTransition
    to_state: CycleState


class CycleStateMachine:
    """
    Manages state transitions for a single cycle.

    Enforces valid operation ordering and provides clear error messages
    when invalid operations are attempted.
    """

    # Define all valid transitions
    TRANSITIONS = [
        StateTransitionRule(
            CycleState.MEMBERS_ADVANCED,
            StateTransition.FILTER_COMPLETE,
            CycleState.FILTER_COMPLETE,
        ),
        StateTransitionRule(
            CycleState.FILTER_COMPLETE,
            StateTransition.ANALYSIS_COMPLETE,
            CycleState.ANALYSIS_COMPLETE,
        ),
        # Cycling to next cycle from analysis
        StateTransitionRule(
            CycleState.ANALYSIS_COMPLETE,
            StateTransition.CYCLE_COMPLETE,
            CycleState.CYCLE_COMPLETE,
        ),
        # Allow cycling from MEMBERS_ADVANCED if using forecast (skip filter/analysis)
        StateTransitionRule(
            CycleState.MEMBERS_ADVANCED,
            StateTransition.CYCLE_COMPLETE,
            CycleState.CYCLE_COMPLETE,
        ),
    ]

    _transition_map = {(rule.from_state, rule.transition): rule for rule in TRANSITIONS}

    def __init__(self, state: ExperimentState, cycle_idx: int, n_members: int):
        self.state = state
        self.cycle_idx = cycle_idx
        self.n_members = n_members

    @property
    def current_state(self) -> CycleState:
        """
        Work out the state of this cycle from the files on disk.

        Because this is always recomputed, it cannot go stale while the model runs, and
        two members finishing at the same time cannot disagree about the result.
        """

        if self.state.is_marker_set(self.cycle_idx, StateTransition.CYCLE_COMPLETE.value):
            return CycleState.CYCLE_COMPLETE
        if self.state.is_marker_set(
            self.cycle_idx, StateTransition.ANALYSIS_COMPLETE.value
        ):
            return CycleState.ANALYSIS_COMPLETE
        if self.state.is_marker_set(
            self.cycle_idx, StateTransition.FILTER_COMPLETE.value
        ):
            return CycleState.FILTER_COMPLETE

        n_advanced = self.state.count_advanced(self.cycle_idx)
        if n_advanced >= self.n_members:
            return CycleState.MEMBERS_ADVANCED
        if n_advanced > 0:
            return CycleState.ADVANCING_MEMBERS
        return CycleState.INITIALIZED

    def can_transition(self, transition: StateTransition) -> tuple[bool, str]:
        """
        Check if a transition is valid from the current state.

        Args:
            transition: The transition to check

        Returns:
            (is_valid, error_message)
        """

        current_state = self.current_state
        rule = self._transition_map.get((current_state, transition))

        if rule is None:
            step = STEP_NAMES.get(transition, transition.value)
            return False, (
                f"Cannot run {step} for cycle {self.cycle_idx}, which is in state "
                f"{current_state.value}. Next required action: "
                f"{self._get_required_actions(current_state)}"
            )

        return True, ""

    def transition(self, transition: StateTransition) -> None:
        """
        Execute a state transition, recording it on disk.

        Args:
            transition: The transition to execute

        Raises:
            ExperimentStateError: If transition is invalid
        """

        valid, error = self.can_transition(transition)
        if not valid:
            raise ExperimentStateError(error)

        self.state.set_marker(self.cycle_idx, transition.value)

    def reset(self) -> None:
        """
        Return this cycle to INITIALIZED by removing its markers and forgetting which
        members have advanced. Does not touch any model files.
        """

        self.state.clear_cycle_markers(self.cycle_idx, MARKER_NAMES)
        self.state.clear_cycle_members(self.cycle_idx)

    def force_state(self, target: CycleState) -> None:
        """
        Force this cycle into a given state by writing whatever files that state implies,
        skipping the usual ordering checks.

        This is the escape hatch behind `status set-experiment`; the normal workflow goes
        through `transition()`. Members that already advanced keep their recorded
        runtimes.

        Raises:
            ExperimentStateError: if the target state cannot be expressed
        """

        if target == CycleState.ADVANCING_MEMBERS:
            raise ExperimentStateError(
                "advancing_members is derived from how many members have advanced, so it "
                "cannot be set directly. Use `status set-member` instead."
            )

        self.state.clear_cycle_markers(self.cycle_idx, MARKER_NAMES)

        if target == CycleState.INITIALIZED:
            self.state.clear_cycle_members(self.cycle_idx)
            return

        # Every remaining state implies a complete ensemble
        advanced = self.state.get_advanced_members(self.cycle_idx)
        for i in range(self.n_members):
            if i not in advanced:
                self.state.set_member_advanced(self.cycle_idx, i)

        for transition, state in STATES_IN_ORDER:
            if target == CycleState.MEMBERS_ADVANCED:
                break
            self.state.set_marker(self.cycle_idx, transition.value)
            if state == target:
                break

    def _get_required_actions(self, current_state: CycleState) -> str:
        """Get human-readable next actions for current state."""

        if current_state == CycleState.ADVANCING_MEMBERS:
            n_advanced = self.state.count_advanced(self.cycle_idx)
            return (
                f"wait for all members to advance ({n_advanced}/{self.n_members} done)"
            )

        actions = {
            CycleState.INITIALIZED: "start advancing members",
            CycleState.MEMBERS_ADVANCED: "run filter or cycle with forecast",
            CycleState.FILTER_COMPLETE: "run analysis",
            CycleState.ANALYSIS_COMPLETE: "cycle to next period",
            CycleState.CYCLE_COMPLETE: "nothing (cycle is complete)",
        }
        return actions.get(current_state, "unknown")

    def get_required_actions(self) -> str:
        """Public interface to get required actions."""

        return self._get_required_actions(self.current_state)


class ExperimentStateMachine:
    """
    Manages the overall experiment state across multiple cycles.
    """

    def __init__(
        self,
        state: ExperimentState,
        n_cycles: int,
        n_members: int,
        current_cycle_idx: int = 0,
    ):
        self.state = state
        self.n_cycles = n_cycles
        self.n_members = n_members
        self.current_cycle_idx = current_cycle_idx

    @property
    def current_cycle(self) -> CycleStateMachine:
        """Get the state machine for the current cycle."""

        return self.get_cycle(self.current_cycle_idx)

    def get_cycle(self, cycle_idx: int) -> CycleStateMachine:
        """
        Get the state machine for a specific cycle.

        Every cycle's state is read from its own files, so this is accurate for past and
        future cycles, not just the current one.
        """

        return CycleStateMachine(self.state, cycle_idx, self.n_members)

    def can_advance_member(self, cycle_idx: int, member_idx: int) -> tuple[bool, str]:
        """Check if a specific member can be advanced."""

        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                f"Cannot advance cycle {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        cycle = self.get_cycle(cycle_idx)
        state = cycle.current_state

        # Can advance as long as the filter has not run yet
        if state in (CycleState.INITIALIZED, CycleState.ADVANCING_MEMBERS):
            return True, ""

        if state == CycleState.MEMBERS_ADVANCED:
            return False, "All members have already advanced"

        return False, f"Cannot advance members in state {state.value}"

    def can_run_filter(self, cycle_idx: int) -> tuple[bool, str]:
        """Check if filter can be run for a cycle."""

        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                f"Cannot run filter for cycle {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        return self.get_cycle(cycle_idx).can_transition(StateTransition.FILTER_COMPLETE)

    def can_run_analysis(self, cycle_idx: int) -> tuple[bool, str]:
        """Check if analysis can be run for a cycle."""

        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                f"Cannot run analysis for cycle {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        return self.get_cycle(cycle_idx).can_transition(
            StateTransition.ANALYSIS_COMPLETE
        )

    def can_cycle_to_next(self, cycle_idx: int) -> tuple[bool, bool, str]:
        """
        Check if we can advance to the next cycle.

        Returns:
            - able_to_cycle: If we are able to cycle (bool)
            - use_forecast: Whether we should use analysis of forecast files (true for forecasts, false for analysis)
            - error_message: An error message
        """

        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                False,
                f"Cannot cycle from {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        cycle = self.get_cycle(cycle_idx)
        state = cycle.current_state

        if state == CycleState.ANALYSIS_COMPLETE:
            return True, False, ""
        if state == CycleState.MEMBERS_ADVANCED:
            return True, True, ""

        return (
            False,
            False,
            f"Cannot cycle from state {state.value}. Next required action: "
            f"{cycle.get_required_actions()}",
        )

    def advance_to_next_cycle(self):
        """Move to the next cycle."""

        self.current_cycle_idx += 1
