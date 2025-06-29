import pathlib

import click

from wrf_ensembly.commands.ensemble import ensemble_cli
from wrf_ensembly.commands.experiment import experiment_cli
from wrf_ensembly.commands.observations import observations_cli
from wrf_ensembly.commands.postprocess import postprocess_cli
from wrf_ensembly.commands.preprocess import preprocess_cli
from wrf_ensembly.commands.slurm import slurm_cli
from wrf_ensembly.commands.status import status_cli


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


cli.add_command(experiment_cli)
cli.add_command(preprocess_cli)
cli.add_command(ensemble_cli)
cli.add_command(observations_cli)
cli.add_command(slurm_cli)
cli.add_command(postprocess_cli)
cli.add_command(status_cli)

if __name__ == "__main__":
    cli()
