import pathlib

import click

from wrf_ensembly import commands


@click.group()
@click.argument(
    "experiment_path",
    required=True,
    type=click.Path(exists=False, resolve_path=True, path_type=pathlib.Path),
)
@click.pass_context
def cli(ctx, experiment_path: str):
    ctx.ensure_object(dict)
    ctx.obj["experiment_path"] = experiment_path


cli.add_command(commands.experiment.experiment_cli)
cli.add_command(commands.preprocess.preprocess_cli)
cli.add_command(commands.ensemble.ensemble_cli)
cli.add_command(commands.observations.observations_cli)
cli.add_command(commands.slurm.slurm_cli)
