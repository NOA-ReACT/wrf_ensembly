from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated
from wrf_ensembly import config, cycling, experiment, jobfiles, member_info, utils
from wrf_ensembly.console import logger

app = typer.Typer()


@app.command()
def preprocessing(
    experiment_path: Annotated[
        Path, typer.Argument(..., help="Path to the experiment directory")
    ]
):
    """Creates a jobfile for running all preprocessing steps. Useful if you want to run WPS and real on your processing nodes."""

    logger.setup("slurm-preprocessing", experiment_path)
    exp = experiment.Experiment(experiment_path)
    jobfiles.generate_preprocess_jobfile(exp)


@app.command()
def advance_members(
    experiment_path: Annotated[
        Path, typer.Argument(..., help="Path to the experiment directory")
    ],
    cycle: Annotated[
        Optional[int],
        typer.Argument(
            ...,
            help="Cycle to advance members to",
        ),
    ] = None,
):
    """Create a SLURM jobfile to advance each member of the ensemble"""

    logger.setup(f"slurm-advance-members", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # If a cycle is passed by the user, generate jobfiles for that cycle,
    # otherwise grab the current cycle
    if cycle is None:
        exp.ensure_same_cycle()
        cycle = exp.members[0].current_cycle_i

    logger.info(f"Writing jobfiles for cycle {cycle}")
    jobfiles.generate_advance_jobfiles(exp, cycle)


@app.command()
def make_analysis(
    experiment_path: Annotated[
        Path, typer.Argument(..., help="Path to the experiment directory")
    ],
    cycle: Annotated[int, typer.Argument(..., help="Current cycle")],
):
    """
    Creates a SLURM jobfile for the `filter`, `analysis` and `cycle` steps. At runtime,
    the job script will check whether there are observations available for the current
    cycle and will only run `filter` and `analysis` if they are found. Otherwise, only
    `cycle` will be run with the `--use-forecast` flag.
    """

    logger.setup(f"slurm-make-analysis", experiment_path)
    exp = experiment.Experiment(experiment_path)
    jobfiles.generate_make_analysis_jobfile(exp, cycle)


@app.command()
def run_experiment(
    experiment_path: Annotated[
        Path, typer.Argument(..., help="Path to the experiment directory")
    ],
    resume: Annotated[
        Optional[bool],
        typer.Option(..., help="Resume the experiment from the current cycle"),
    ] = False,
    only_next_cycle: Annotated[
        Optional[bool], typer.Option(..., help="Only run the next cycle")
    ] = False,
    in_waves: Annotated[
        Optional[bool],
        typer.Option(..., help="Queue next cycle after the current cycle is done"),
    ] = False,
):
    """
    Creates jobfiles for all experiment steps and queues them in the correct order. This
    does not deal with the initial steps (setup, initial/boundary conditions, ...), only
    the member advancing, analysis and cycling.

    If for some cycle there are not prepared observations (in the `obs` directory), the
    generated job will skip the analysis step and go straight to cycling.

    If there is a job limit on your local HPC and you cannot queue the whole experiment,
    use the `--in-waves` option that only queues the current cycle. At the last job, the
    next cycle will be queued.
    """

    logger.setup(f"slurm-run-experiment", experiment_path)
    exp = experiment.Experiment(experiment_path)
    slurm_command = exp.cfg.slurm.sbatch_command

    # If we need to resume, grab current cycle and filter the cycles list
    cycles = exp.cycles
    if resume:
        exp.ensure_same_cycle()
        current_cycle = exp.members[0].current_cycle_i
        cycles = list(filter(lambda c: c.index >= current_cycle, cycles))

    # If we only want to run the next cycle, keep only the first element of the list
    if only_next_cycle or in_waves:
        cycles = [cycles[0]]

    last_cycle_dependency = None
    for cycle in cycles:
        # Generate all member jobfiles, queue them and keep jobids
        jf = jobfiles.generate_advance_jobfiles(exp, cycle.index)

        if last_cycle_dependency is not None:
            dependency = f"--dependency=afterok:{last_cycle_dependency}"
        else:
            dependency = None

        ids = []
        for f in jf:
            cmd = slurm_command.split(" ")
            if dependency is not None:
                cmd.append(dependency)
            cmd.append(str(f.resolve()))
            print(cmd)
            res = utils.call_external_process(cmd)
            if not res.success:
                logger.error("Could not queue jobfile, output:")
                logger.error(res.stdout)
                exit(1)

            id = int(res.stdout.strip())
            ids.append(id)

            logger.info(f"Queued {f} with ID {id}")

        # Generate the analysis jobfile, queue it and keep jobid
        jf = jobfiles.generate_make_analysis_jobfile(exp, cycle.index, in_waves)  # type: ignore
        dependency = "--dependency=afterok:" + ":".join(map(str, ids))
        res = utils.call_external_process(
            [*slurm_command.split(" "), dependency, str(jf.resolve())]
        )
        last_cycle_dependency = int(res.stdout.strip())

        logger.info(f"Queued {jf} with ID {last_cycle_dependency}")
