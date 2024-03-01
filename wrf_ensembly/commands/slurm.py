import sys
from pathlib import Path
from typing import Optional

import click

from wrf_ensembly import experiment, external, jobfiles
from wrf_ensembly.click_utils import pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="slurm")
def slurm_cli():
    pass


@slurm_cli.command()
@pass_experiment_path
def preprocessing(experiment_path: Path):
    """Creates a jobfile for running all preprocessing steps. Useful if you want to run WPS and real on your processing nodes."""

    logger.setup("slurm-preprocessing", experiment_path)
    exp = experiment.Experiment(experiment_path)
    jobfiles.generate_preprocess_jobfile(exp)


@slurm_cli.command()
@click.option("--cycle", type=int, help="Cycle to advance members to")
@pass_experiment_path
def advance_members(experiment_path: Path, cycle: Optional[int]):
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


@slurm_cli.command()
@click.argument("cycle", type=int)
@pass_experiment_path
def make_analysis(experiment_path: Path, cycle: int):
    """
    Creates a SLURM jobfile for the `filter`, `analysis` and `cycle` steps. At runtime,
    the job script will check whether there are observations available for the current
    cycle and will only run `filter` and `analysis` if they are found. Otherwise, only
    `cycle` will be run with the `--use-forecast` flag.
    """

    logger.setup(f"slurm-make-analysis", experiment_path)
    exp = experiment.Experiment(experiment_path)
    jobfiles.generate_make_analysis_jobfile(exp, cycle)


@slurm_cli.command()
@click.option(
    "--all-cycles/--next-cycle-only",
    help="After the current cycle, automatically queue next cycle, until experiment is over",
    default=True,
)
@click.option(
    "--compute-statistics",
    is_flag=True,
    help="Compute statistics for the current cycle after the analysis step",
)
@click.option(
    "--delete-members",
    is_flag=True,
    help="Requires --compute-statistics. If set, the individual member's forecasts are deleted.",
)
@pass_experiment_path
def run_experiment(
    experiment_path: Path,
    all_cycles: bool,
    compute_statistics: bool,
    delete_members: bool,
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
            sys.exit(1)
        except ValueError:
            pass

    # Generate all member jobfiles, queue them and keep jobids
    jfs = jobfiles.generate_advance_jobfiles(exp, current_cycle.index)

    ids = []
    for jf in jfs:
        cmd = slurm_command.split(" ")
        cmd.append(str(jf.resolve()))

        res = external.runc(cmd)
        if res.returncode != 0:
            logger.error("Could not queue jobfile, output:")
            logger.error(res.output)
            exit(1)

        id = int(res.output.strip())
        ids.append(id)

        logger.info(f"Queued {jf} with ID {id}")

    # Generate the analysis jobfile, queue it and keep jobid
    jf = jobfiles.generate_make_analysis_jobfile(
        exp, current_cycle.index, all_cycles, compute_statistics, delete_members
    )
    dependency = "--dependency=afterok:" + ":".join(map(str, ids))
    res = external.runc([*slurm_command.split(" "), dependency, str(jf.resolve())])
    analysis_jobid = int(res.output.strip())
    ids.append(analysis_jobid)
    logger.info(f"Queued {jf} with ID {analysis_jobid}")

    if compute_statistics:
        jf = jobfiles.generate_statistics_jobfile(
            exp, current_cycle.index, delete_members
        )
        res = external.runc(
            [
                *slurm_command.split(" "),
                f"--dependency=afterok:{analysis_jobid}",
                str(jf.resolve()),
            ]
        )
        logger.info(f"Queued {jf} with ID {res.output.strip()}")
        ids.append(int(res.output.strip()))

    logger.info(f"First JobID: {min(ids)}, last JobID: {max(ids)}")
