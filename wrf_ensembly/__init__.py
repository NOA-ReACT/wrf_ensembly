import typer

from wrf_ensembly import (
    commands,
    config,
    console,
    cycling,
    member_info,
    namelist,
    templates,
    utils,
    wrf,
)


def main():
    """CLI entry point for the application"""

    app = typer.Typer()
    app.add_typer(commands.experiment.app, name="experiment")
    app.add_typer(commands.preprocess.app, name="preprocess")
    app.add_typer(commands.ensemble.app, name="ensemble")
    app.add_typer(commands.slurm.app, name="slurm")
    app.add_typer(commands.observations.app, name="observations")

    app()
