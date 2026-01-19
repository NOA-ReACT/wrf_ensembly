"""
Formal state machine for Experiments, checking which actions are allowed at which
cycle states.

The purpose of this formality is to reduce errors caused by forgetting or double-running
commands.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
    """Valid operations that trigger state transitions."""

    START_ADVANCING = "start_advancing"
    MEMBER_ADVANCED = "member_advanced"
    ALL_MEMBERS_ADVANCED = "all_members_advanced"
    FILTER_COMPLETE = "filter_complete"
    ANALYSIS_COMPLETE = "analysis_complete"
    CYCLE_COMPLETE = "cycle_complete"


@dataclass
class StateTransitionRule:
    """Defines a valid transition between states."""

    from_state: CycleState
    transition: StateTransition
    to_state: CycleState
    requires_all_members: bool = False


class CycleStateMachine:
    """
    Manages state transitions for a single cycle.

    Enforces valid operation ordering and provides clear error messages
    when invalid operations are attempted.
    """

    # Define all valid transitions
    TRANSITIONS = [
        # Initial advancement
        StateTransitionRule(
            CycleState.INITIALIZED,
            StateTransition.START_ADVANCING,
            CycleState.ADVANCING_MEMBERS,
        ),
        # Member advancement tracking
        StateTransitionRule(
            CycleState.ADVANCING_MEMBERS,
            StateTransition.MEMBER_ADVANCED,
            CycleState.ADVANCING_MEMBERS,
        ),
        StateTransitionRule(
            CycleState.ADVANCING_MEMBERS,
            StateTransition.ALL_MEMBERS_ADVANCED,
            CycleState.MEMBERS_ADVANCED,
            requires_all_members=True,
        ),
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

    def __init__(self, initial_state: CycleState = CycleState.INITIALIZED):
        self.current_state = initial_state

        self._transition_map = {}
        for rule in self.TRANSITIONS:
            key = (rule.from_state, rule.transition)
            self._transition_map[key] = rule

    def can_transition(
        self,
        transition: StateTransition,
        members_advanced: int | None = None,
        total_members: int | None = None,
    ) -> tuple[bool, str]:
        """
        Check if a transition is valid from the current state.

        Args:
            transition: The transition to check
            members_advanced: Number of members that have advanced (for ALL_MEMBERS_ADVANCED)
            total_members: Total number of members (for ALL_MEMBERS_ADVANCED)

        Returns:
            (is_valid, error_message)
        """
        key = (self.current_state, transition)
        rule = self._transition_map.get(key)

        if rule is None:
            return False, (
                f"Cannot {transition.value} from state {self.current_state.value}. "
                f"Current state requires: {self._get_required_actions()}"
            )

        # Check member advancement requirement
        if rule.requires_all_members:
            if members_advanced is None or total_members is None:
                return False, "Member count information required"
            if members_advanced < total_members:
                return (
                    False,
                    f"Only {members_advanced}/{total_members} members have advanced. "
                    f"All members must complete before proceeding.",
                )

        return True, ""

    def transition(
        self,
        transition: StateTransition,
        members_advanced: Optional[int] = None,
        total_members: Optional[int] = None,
    ) -> None:
        """
        Execute a state transition.

        Args:
            transition: The transition to execute
            members_advanced: Number of members that have advanced
            total_members: Total number of members

        Raises:
            ValueError: If transition is invalid
        """
        valid, error = self.can_transition(transition, members_advanced, total_members)
        if not valid:
            raise ValueError(error)

        rule = self._transition_map[(self.current_state, transition)]
        self.current_state = rule.to_state

    def _get_required_actions(self) -> str:
        """Get human-readable next actions for current state."""
        actions = {
            CycleState.INITIALIZED: "start advancing members",
            CycleState.ADVANCING_MEMBERS: "wait for all members to advance",
            CycleState.MEMBERS_ADVANCED: "run filter or cycle with forecast",
            CycleState.FILTER_COMPLETE: "run analysis",
            CycleState.ANALYSIS_COMPLETE: "cycle to next period",
            CycleState.CYCLE_COMPLETE: "nothing (cycle is complete)",
        }
        return actions.get(self.current_state, "unknown")

    def get_required_actions(self) -> str:
        """Public interface to get required actions."""
        return self._get_required_actions()


class ExperimentStateMachine:
    """
    Manages the overall experiment state across multiple cycles.

    Tracks preprocessing state and coordinates cycle state machines.
    """

    def __init__(self, n_cycles: int, current_cycle_idx: int = 0):
        self.cycles = {i: CycleStateMachine() for i in range(n_cycles)}
        self.current_cycle_idx = current_cycle_idx
        self.preprocessing_complete = False

    @property
    def current_cycle(self) -> CycleStateMachine:
        """Get the state machine for the current cycle."""
        return self.cycles[self.current_cycle_idx]

    def get_cycle(self, cycle_idx: int) -> CycleStateMachine:
        """Get the state machine for a specific cycle."""
        return self.cycles[cycle_idx]

    def can_advance_member(self, cycle_idx: int, member_idx: int) -> tuple[bool, str]:
        """Check if a specific member can be advanced."""
        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                f"Cannot advance cycle {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        cycle = self.cycles[cycle_idx]

        # Can advance if in INITIALIZED or ADVANCING_MEMBERS state
        if cycle.current_state == CycleState.INITIALIZED:
            return cycle.can_transition(StateTransition.START_ADVANCING)
        elif cycle.current_state == CycleState.ADVANCING_MEMBERS:
            return True, ""
        else:
            return (
                False,
                f"Cannot advance members in state {cycle.current_state.value}",
            )

    def can_run_filter(
        self, cycle_idx: int, members_advanced: int, total_members: int
    ) -> tuple[bool, str]:
        """Check if filter can be run for a cycle."""
        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                f"Cannot run filter for cycle {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        cycle = self.cycles[cycle_idx]

        # Must be in MEMBERS_ADVANCED state
        if cycle.current_state != CycleState.MEMBERS_ADVANCED:
            return (
                False,
                f"Cannot run filter from state {cycle.current_state.value}. "
                f"Must complete member advancement first.",
            )

        return True, ""

    def can_run_analysis(self, cycle_idx: int) -> tuple[bool, str]:
        """Check if analysis can be run for a cycle."""
        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                f"Cannot run analysis for cycle {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        cycle = self.cycles[cycle_idx]

        # Must be in FILTER_COMPLETE state
        if cycle.current_state != CycleState.FILTER_COMPLETE:
            return (
                False,
                f"Cannot run analysis from state {cycle.current_state.value}. "
                f"Must complete filter first.",
            )

        return True, ""

    def can_cycle_to_next(
        self, cycle_idx: int, use_forecast: bool = False
    ) -> tuple[bool, str]:
        """Check if we can advance to the next cycle."""
        if cycle_idx != self.current_cycle_idx:
            return (
                False,
                f"Cannot cycle from {cycle_idx}, currently on cycle {self.current_cycle_idx}",
            )

        cycle = self.cycles[cycle_idx]

        # If using forecast, we can cycle from MEMBERS_ADVANCED
        # Otherwise we need ANALYSIS_COMPLETE
        valid_states = (
            [CycleState.MEMBERS_ADVANCED, CycleState.ANALYSIS_COMPLETE]
            if use_forecast
            else [CycleState.ANALYSIS_COMPLETE]
        )

        if cycle.current_state not in valid_states:
            return (
                False,
                f"Cannot cycle from state {cycle.current_state.value}. "
                f"Need {'MEMBERS_ADVANCED (with --use-forecast) or ANALYSIS_COMPLETE' if use_forecast else 'ANALYSIS_COMPLETE'}",
            )

        return True, ""

    def advance_to_next_cycle(self):
        """Move to the next cycle."""
        self.current_cycle_idx += 1

    def load_cycle_state(self, cycle_idx: int, state: CycleState):
        """Load a cycle's state from database."""
        if cycle_idx in self.cycles:
            self.cycles[cycle_idx].current_state = state
