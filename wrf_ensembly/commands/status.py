from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from wrf_ensembly import experiment
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="status", cls=GroupWithStartEndPrint)
def status_cli():
    """Commands for viewing and managing experiment status"""
    pass


@status_cli.command()
@pass_experiment_path
def show(experiment_path: Path):
    """Prints the current experiment status with a rich table showing member status and experiment state"""

    logger.setup("status-show", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Experiment State Table
    exp_table = Table(title="Experiment State")
    exp_table.add_column("Property", style="bold cyan")
    exp_table.add_column("Value", style="bold white")

    current_cycle = (
        exp.cycles[exp.current_cycle_i]
        if exp.current_cycle_i < len(exp.cycles)
        else None
    )

    exp_table.add_row("Current Cycle", str(exp.current_cycle_i))
    exp_table.add_row("Total Cycles", str(len(exp.cycles)))
    exp_table.add_row(
        "Cycle State", exp.state_machine.current_cycle.current_state.value
    )
    exp_table.add_row("Filter Run", "✓" if exp.filter_run else "✗")
    exp_table.add_row("Analysis Run", "✓" if exp.analysis_run else "✗")
    exp_table.add_row("All Members Advanced", "✓" if exp.all_members_advanced else "✗")

    if current_cycle:
        exp_table.add_row(
            "Cycle Start", current_cycle.start.strftime("%Y-%m-%d %H:%M:%S")
        )
        exp_table.add_row("Cycle End", current_cycle.end.strftime("%Y-%m-%d %H:%M:%S"))

    # Member Status Table
    member_table = Table(title="Member Status")
    member_table.add_column("Member", justify="center", style="bold cyan")
    member_table.add_column("Advanced", justify="center")
    member_table.add_column("Runtime Stats", justify="center")

    for member in exp.members:
        # Count runtime statistics for this member
        stats_count = len(member.runtime_statistics)

        member_table.add_row(
            str(member.i), "✓" if member.advanced else "✗", str(stats_count) + " cycles"
        )

    console = Console()
    console.print(exp_table)
    console.print()
    console.print(member_table)


@status_cli.command()
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Also print the full per-row table with every (cycle, member) entry",
)
@pass_experiment_path
def runtime_stats(experiment_path: Path, show_all: bool):
    """Prints aggregated runtime statistics per cycle, per member, and overall"""

    logger.setup("status-runtime-stats", experiment_path)
    exp = experiment.Experiment(experiment_path)

    console = Console()

    # Collect all runtime statistics
    all_stats: list[tuple] = []
    by_cycle: dict[int, list] = defaultdict(list)
    by_member: dict[int, list[int]] = defaultdict(list)
    all_durations: list[int] = []
    for member in exp.members:
        for stat in member.runtime_statistics:
            all_stats.append((stat, member.i))
            by_cycle[stat.cycle].append((stat, member.i))
            by_member[member.i].append(stat.duration_s)
            all_durations.append(stat.duration_s)

    if not all_stats:
        console.print("[yellow]No runtime statistics available[/yellow]")
        return

    all_stats.sort(key=lambda x: (x[0].cycle, x[1]))

    def _s_to_min(seconds: float) -> float:
        return seconds / 60.0

    def _agg(values: list[int]) -> tuple[float, float, float, float]:
        arr = np.asarray(values, dtype=float) / 60.0
        std = float(arr.std(ddof=0)) if arr.size > 1 else 0.0
        return float(arr.mean()), float(arr.min()), float(arr.max()), std

    # 1. Per-cycle table
    cycle_table = Table(title="Per-Cycle Runtime")
    cycle_table.add_column("Cycle", justify="center", style="bold cyan")
    cycle_table.add_column("N", justify="right")
    cycle_table.add_column("Mean (min)", justify="right", style="bold green")
    cycle_table.add_column("Min (min)", justify="right")
    cycle_table.add_column("Max (min)", justify="right")
    cycle_table.add_column("Std (min)", justify="right")
    cycle_table.add_column("Wall (min)", justify="right", style="bold magenta")

    for cycle in sorted(by_cycle.keys()):
        entries = by_cycle[cycle]
        durations = [s.duration_s for s, _ in entries]
        mean, mn, mx, std = _agg(durations)
        wall = _s_to_min(
            (
                max(s.end for s, _ in entries) - min(s.start for s, _ in entries)
            ).total_seconds()
        )
        cycle_table.add_row(
            str(cycle),
            str(len(durations)),
            f"{mean:.1f}",
            f"{mn:.1f}",
            f"{mx:.1f}",
            f"{std:.1f}",
            f"{wall:.1f}",
        )

    # 2. Per-member table
    member_table = Table(title="Per-Member Runtime")
    member_table.add_column("Member", justify="center", style="bold cyan")
    member_table.add_column("N", justify="right")
    member_table.add_column("Mean (min)", justify="right", style="bold green")
    member_table.add_column("Min (min)", justify="right")
    member_table.add_column("Max (min)", justify="right")
    member_table.add_column("Std (min)", justify="right")

    for member_i in sorted(by_member.keys()):
        durations = by_member[member_i]
        mean, mn, mx, std = _agg(durations)
        member_table.add_row(
            str(member_i),
            str(len(durations)),
            f"{mean:.1f}",
            f"{mn:.1f}",
            f"{mx:.1f}",
            f"{std:.1f}",
        )

    # 3. Recent cycles per member table
    available_cycles = sorted(by_cycle.keys())
    current_cycle = exp.current_cycle_i
    if current_cycle in by_cycle:
        last_cycle = current_cycle - 1 if (current_cycle - 1) in by_cycle else None
    else:
        # Fall back to the two most recent cycles that have data
        current_cycle = available_cycles[-1]
        last_cycle = available_cycles[-2] if len(available_cycles) >= 2 else None

    def _member_duration(cycle: int, member_i: int) -> float | None:
        for s, mi in by_cycle[cycle]:
            if mi == member_i:
                return _s_to_min(s.duration_s)
        return None

    members_in_view: set[int] = set()
    for s, mi in by_cycle[current_cycle]:
        members_in_view.add(mi)
    if last_cycle is not None:
        for s, mi in by_cycle[last_cycle]:
            members_in_view.add(mi)

    recent_table = Table(title="Recent Cycles per Member")
    recent_table.add_column("Member", justify="center", style="bold cyan")
    if last_cycle is not None:
        recent_table.add_column(f"Cycle {last_cycle} (min)", justify="right")
    recent_table.add_column(f"Cycle {current_cycle} (min)", justify="right")
    if last_cycle is not None:
        recent_table.add_column("Δ (min)", justify="right")

    for member_i in sorted(members_in_view):
        cur = _member_duration(current_cycle, member_i)
        cur_str = f"{cur:.1f}" if cur is not None else "-"
        if last_cycle is not None:
            last = _member_duration(last_cycle, member_i)
            last_str = f"{last:.1f}" if last is not None else "-"
            if cur is not None and last is not None:
                delta = cur - last
                if delta < 0:
                    delta_str = f"[green]{delta:+.1f}[/green]"
                elif delta > 0:
                    delta_str = f"[red]{delta:+.1f}[/red]"
                else:
                    delta_str = "0.0"
            else:
                delta_str = ""
            recent_table.add_row(str(member_i), last_str, cur_str, delta_str)
        else:
            recent_table.add_row(str(member_i), cur_str)

    # 4. Overall row
    o_mean, o_min, o_max, o_std = _agg(all_durations)
    overall_table = Table(title="Overall Runtime")
    overall_table.add_column("N (rows)", justify="right")
    overall_table.add_column("Members", justify="right")
    overall_table.add_column("Cycles", justify="right")
    overall_table.add_column("Mean (min)", justify="right", style="bold green")
    overall_table.add_column("Min (min)", justify="right")
    overall_table.add_column("Max (min)", justify="right")
    overall_table.add_column("Std (min)", justify="right")
    overall_table.add_column("Total (min)", justify="right", style="bold magenta")
    overall_table.add_row(
        str(len(all_durations)),
        str(len(by_member)),
        str(len(by_cycle)),
        f"{o_mean:.1f}",
        f"{o_min:.1f}",
        f"{o_max:.1f}",
        f"{o_std:.1f}",
        f"{_s_to_min(sum(all_durations)):.1f}",
    )

    console.print(cycle_table)
    console.print()
    console.print(member_table)
    console.print()
    console.print(recent_table)
    console.print()
    console.print(overall_table)

    if show_all:
        detail_table = Table(title="Runtime Statistics (all entries)")
        detail_table.add_column("Cycle", justify="center", style="bold cyan")
        detail_table.add_column("Member", justify="center", style="bold cyan")
        detail_table.add_column("Start Time", justify="center")
        detail_table.add_column("End Time", justify="center")
        detail_table.add_column("Duration (s)", justify="right", style="bold green")
        for stat, member_i in all_stats:
            detail_table.add_row(
                str(stat.cycle),
                str(member_i),
                stat.start.strftime("%Y-%m-%d %H:%M:%S"),
                stat.end.strftime("%Y-%m-%d %H:%M:%S"),
                str(stat.duration_s),
            )
        console.print()
        console.print(detail_table)


@status_cli.command()
@pass_experiment_path
def clear_runtime_stats(experiment_path: Path):
    """Clears all runtime statistics from the database"""

    logger.setup("status-clear-runtime-stats", experiment_path)
    exp = experiment.Experiment(experiment_path)

    with exp.db as db_conn:
        db_conn.clear_runtime_statistics()

    logger.info("Cleared all runtime statistics")


@status_cli.command()
@click.option("--confirm", is_flag=True, help="Confirm the reset action")
@pass_experiment_path
def reset(experiment_path: Path, confirm: bool):
    """Resets the experiment state to the beginning (cycle 0, no advanced members)"""

    logger.setup("status-reset", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if not confirm:
        logger.warning(
            "This will reset the experiment to cycle 0 and clear all member advancement status."
        )
        logger.warning("Add --confirm flag to proceed.")
        return

    # Reset experiment state
    exp.current_cycle_i = 0

    # Reset state machine to initial state
    from wrf_ensembly.experiment import CycleState

    exp.state_machine.current_cycle_idx = 0
    exp.state_machine.current_cycle.current_state = CycleState.INITIALIZED

    # Reset all members
    for member in exp.members:
        member.advanced = False
        member.runtime_statistics.clear()

    # Clear database
    with exp.db as db_conn:
        db_conn.reset_experiment()

    logger.info("Reset experiment state to cycle 0")


@status_cli.command()
@click.argument("member_index", type=int)
@click.argument("advanced", type=bool)
@pass_experiment_path
def set_member(experiment_path: Path, member_index: int, advanced: bool):
    """Set the advanced status for a specific member by index"""

    logger.setup("status-set-member", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if member_index < 0 or member_index >= len(exp.members):
        logger.error(
            f"Member index {member_index} out of range (0-{len(exp.members) - 1})"
        )
        return

    # Update in database
    with exp.db as db_conn:
        db_conn.set_member_advanced(member_index, advanced)

    # Update local object
    exp.members[member_index].advanced = advanced

    status_str = "advanced" if advanced else "not advanced"
    logger.info(f"Set member {member_index} status to: {status_str}")


@status_cli.command()
@click.argument("advanced", type=bool)
@pass_experiment_path
def set_all_members(experiment_path: Path, advanced: bool):
    """Set the advanced status for all members"""

    logger.setup("status-set-all-members", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Update all members in database
    with exp.db as db_conn:
        for member in exp.members:
            db_conn.set_member_advanced(member.i, advanced)
            member.advanced = advanced

    status_str = "advanced" if advanced else "not advanced"
    logger.info(f"Set all {len(exp.members)} members to: {status_str}")


@status_cli.command()
@click.option("--cycle", type=int, help="Set current cycle number")
@click.option(
    "--state",
    type=click.Choice(
        [
            "initialized",
            "advancing_members",
            "members_advanced",
            "filter_complete",
            "analysis_complete",
            "cycle_complete",
        ]
    ),
    help="Set the cycle state",
)
@pass_experiment_path
def set_experiment(experiment_path: Path, cycle: int, state: str):
    """Set the experiment state (cycle number and/or cycle state)"""

    logger.setup("status-set-experiment", experiment_path)
    exp = experiment.Experiment(experiment_path)

    from wrf_ensembly.experiment import CycleState

    # Validate cycle number
    if cycle is not None and (cycle < 0 or cycle >= len(exp.cycles)):
        logger.error(f"Cycle {cycle} out of range (0-{len(exp.cycles) - 1})")
        return

    # Use current values if not specified
    current_cycle = cycle if cycle is not None else exp.current_cycle_i

    # Update cycle if changed
    if cycle is not None:
        exp.current_cycle_i = current_cycle
        exp.state_machine.current_cycle_idx = current_cycle

    # Update state if specified
    if state is not None:
        cycle_state = CycleState(state)
        exp.state_machine.current_cycle.current_state = cycle_state
        logger.info(f"Set cycle {current_cycle} state to: {state}")

    # Save to database
    exp.save_status_to_db()

    logger.info(
        f"Set experiment state - Cycle: {current_cycle}, State: {exp.state_machine.current_cycle.current_state.value}"
    )
