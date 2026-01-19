from pathlib import Path

import click
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
@pass_experiment_path
def runtime_stats(experiment_path: Path):
    """Prints the runtime statistics of all members across all cycles in a rich table"""

    logger.setup("status-runtime-stats", experiment_path)
    exp = experiment.Experiment(experiment_path)

    table = Table(title="Runtime Statistics")
    table.add_column("Cycle", justify="center", style="bold cyan")
    table.add_column("Member", justify="center", style="bold cyan")
    table.add_column("Start Time", justify="center")
    table.add_column("End Time", justify="center")
    table.add_column("Duration (s)", justify="right", style="bold green")

    # Collect all runtime statistics from all members
    all_stats = []
    for member in exp.members:
        for stat in member.runtime_statistics:
            all_stats.append((stat, member.i))

    # Sort by cycle, then by member
    all_stats.sort(key=lambda x: (x[0].cycle, x[1]))

    for stat, member_i in all_stats:
        table.add_row(
            str(stat.cycle),
            str(member_i),
            stat.start.strftime("%Y-%m-%d %H:%M:%S"),
            stat.end.strftime("%Y-%m-%d %H:%M:%S"),
            str(stat.duration_s),
        )

    console = Console()
    if len(all_stats) > 0:
        console.print(table)
    else:
        console.print("[yellow]No runtime statistics available[/yellow]")


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
