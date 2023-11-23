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
    all_cycles: Annotated[
        bool,
        typer.Option(
            ...,
            help="After the current cycle, automatically queue next cycle, until experiment is over",
        ),
    ] = True,
    compute_statistics: Annotated[
        bool,
        typer.Option(
            ...,
            help="Compute statistics for the current cycle after the analysis step",
        ),
    ] = False,
    delete_members: Annotated[
        bool,
        typer.Option(
            ...,
            help="Requires --compute-statistics. If set, the individual member's forecasts are deleted.",
        ),
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
    exp.ensure_same_cycle()
    current_cycle = exp.cycles[exp.members[0].current_cycle_i]

    # Check if the current cycle is the last one
    if current_cycle.index == len(exp.cycles) - 1:
        try:
            exp.ensure_current_cycle_state({"advanced": True})
            logger.error("Last cycle already advanced, experiment finished")
            raise typer.Exit(1)
        except ValueError:
            pass

    # Generate all member jobfiles, queue them and keep jobids
    jfs = jobfiles.generate_advance_jobfiles(exp, current_cycle.index)

    ids = []
    for jf in jfs:
        cmd = slurm_command.split(" ")
        cmd.append(str(jf.resolve()))

        res = utils.call_external_process(cmd)
        if not res.success:
            logger.error("Could not queue jobfile, output:")
            logger.error(res.stdout)
            exit(1)

        id = int(res.stdout.strip())
        ids.append(id)

        logger.info(f"Queued {jf} with ID {id}")

    # Generate the analysis jobfile, queue it and keep jobid
    jf = jobfiles.generate_make_analysis_jobfile(
        exp, current_cycle.index, all_cycles, compute_statistics, delete_members
    )
    dependency = "--dependency=afterok:" + ":".join(map(str, ids))
    res = utils.call_external_process(
        [*slurm_command.split(" "), dependency, str(jf.resolve())]
    )
    analysis_jobid = int(res.stdout.strip())
    ids.append(analysis_jobid)
    logger.info(f"Queued {jf} with ID {analysis_jobid}")

    if compute_statistics:
        jf = jobfiles.generate_statistics_jobfile(
            exp, current_cycle.index, delete_members
        )
        res = utils.call_external_process(
            [
                *slurm_command.split(" "),
                f"--dependency=afterok:{analysis_jobid}",
                str(jf.resolve()),
            ]
        )
        logger.info(f"Queued {jf} with ID {res.stdout.strip()}")
        ids.append(int(res.stdout.strip()))

    logger.info(f"First JobID: {min(ids)}, last JobID: {max(ids)}")
