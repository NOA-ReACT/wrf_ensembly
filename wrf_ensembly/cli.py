import pathlib

import click


class LazyGroup(click.Group):
    """
    A Click group that lazy-loads subcommands
    Brings a small improvement in start-up time
    More info:
    https://click.palletsprojects.com/en/stable/complex/#lazily-loading-subcommands
    """

    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx):
        return sorted(self.lazy_subcommands.keys())

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.lazy_subcommands:
            module_path, attr = self.lazy_subcommands[cmd_name]
            mod = __import__(module_path, fromlist=[attr])
            return getattr(mod, attr)
        return None


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "experiment": ("wrf_ensembly.commands.experiment", "experiment_cli"),
        "preprocess": ("wrf_ensembly.commands.preprocess", "preprocess_cli"),
        "ensemble": ("wrf_ensembly.commands.ensemble", "ensemble_cli"),
        "obs-sequence": (
            "wrf_ensembly.commands.obs_sequence",
            "observation_sequence_cli",
        ),
        "slurm": ("wrf_ensembly.commands.slurm", "slurm_cli"),
        "postprocess": ("wrf_ensembly.commands.postprocess", "postprocess_cli"),
        "status": ("wrf_ensembly.commands.status", "status_cli"),
        "observations": ("wrf_ensembly.commands.observations", "observations_cli"),
        "validation": ("wrf_ensembly.commands.validation", "validation_cli"),
        "plots": ("wrf_ensembly.commands.plots", "plots_cli"),
    },
)
@click.argument(
    "experiment_path",
    required=True,
    type=click.Path(exists=False, resolve_path=True, path_type=pathlib.Path),
)
@click.pass_context
def cli(ctx, experiment_path: str):
    ctx.ensure_object(dict)
    ctx.obj["experiment_path"] = experiment_path


if __name__ == "__main__":
    cli()
