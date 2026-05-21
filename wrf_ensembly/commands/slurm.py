import sys
from pathlib import Path

import click

from wrf_ensembly import experiment, external, jobfiles
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="slurm", cls=GroupWithStartEndPrint)
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
@pass_experiment_path
def advance_members(experiment_path: Path):
    """Create a SLURM job array file to advance each non-advanced member of the ensemble"""

    logger.setup("slurm-advance-members", experiment_path)
    exp = experiment.Experiment(experiment_path)

    logger.info("Writing array jobfile for advancing members...")
    result = jobfiles.generate_advance_array_jobfile(exp)
    if result is None:
        logger.info("All members already advanced, no jobfile generated")


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

    logger.setup("slurm-make-analysis", experiment_path)
    exp = experiment.Experiment(experiment_path)
    jobfiles.generate_make_analysis_jobfile(exp, cycle)


@slurm_cli.command()
@click.argument("cycle", type=int)
@click.option(
    "--clean-scratch",
    is_flag=True,
    help="Requires --run-postprocess. If set, the individual member's forecasts are deleted from the scratch directories",
)
@pass_experiment_path
def postprocess(experiment_path: Path, cycle: int, clean_scratch: bool):
    """Create a SLURM jobfile to postprocess the WRF output"""

    logger.setup("slurm-postprocessing", experiment_path)
    exp = experiment.Experiment(experiment_path)

    logger.info("Writing jobfile for postprocessing...")
    jobfiles.generate_postprocess_jobfile(exp, cycle, clean_scratch)


@slurm_cli.command()
@click.option(
    "--clean-scratch",
    is_flag=True,
    help="Requires --run-postprocess. If set, the individual member's forecasts are deleted from the scratch directories",
)
@click.option(
    "--first-cycle",
    type=int,
    help="Queue postprocessing for all cycles starting from this one",
)
@click.option(
    "--last-cycle",
    type=int,
    help="Queue postprocessing for all cycles up to this one (inclusive)",
)
@click.option(
    "--max-parallel",
    type=int,
    default=30,
    help="Maximum number of array tasks running in parallel",
)
@pass_experiment_path
def queue_all_postprocessing(
    experiment_path: Path,
    clean_scratch: bool,
    first_cycle: int | None,
    last_cycle: int | None,
    max_parallel: int,
):
    """Queue postprocessing for all cycles of the experiment as a SLURM job array"""

    logger.setup("slurm-queue-all-postprocessing", experiment_path)
    exp = experiment.Experiment(experiment_path)

    min_cycle = first_cycle if first_cycle is not None and first_cycle > 0 else 0
    max_cycle = (
        last_cycle
        if last_cycle is not None and last_cycle < len(exp.cycles)
        else len(exp.cycles) - 1
    )

    logger.info(
        f"Queueing postprocessing array for cycles {min_cycle}-{max_cycle} (max {max_parallel} parallel)..."
    )
    jf = jobfiles.generate_postprocess_array_jobfile(
        exp, min_cycle, max_cycle, max_parallel, clean_scratch
    )

    res = external.runc([*exp.cfg.slurm.sbatch_command.split(" "), str(jf.resolve())])
    logger.info(f"Queued {jf} with array ID {res.output.strip()}")


@slurm_cli.command()
@click.option(
    "--all-cycles/--next-cycle-only",
    help="After the current cycle, automatically queue next cycle, until experiment is over",
    default=True,
)
@click.option(
    "--run-postprocess",
    is_flag=True,
    help="Compute statistics for the current cycle after the analysis step",
)
@click.option(
    "--clean-scratch",
    is_flag=True,
    help="Requires --run-postprocess. If set, the individual member's forecasts are deleted from the scratch directories",
)
@click.option(
    "--only-advance",
    is_flag=True,
    help="Only queue the advance steps",
)
@click.option(
    "--run-until", type=int, required=False, help="Run until this cycle (end-inclusive)"
)
@click.option(
    "--max-parallel",
    type=int,
    required=False,
    help="Maximum number of parallel jobs to run",
)
@pass_experiment_path
def run_experiment(
    experiment_path: Path,
    all_cycles: bool,
    run_postprocess: bool,
    clean_scratch: bool,
    only_advance: bool,
    run_until: int | None,
    max_parallel: int | None,
):
    """
    Creates jobfiles for all experiment steps and queues them in the correct order. This
    does not deal with the initial steps (setup, initial/boundary conditions, ...), only
    the member advancing, analysis and cycling. Postprocessing will be queued if you use
    `--run_postprocess`.

    If for some cycle there are not prepared observations (in the `obs` directory), the
    generated job will skip the analysis step and go straight to cycling.
    """

    logger.setup("slurm-run-experiment", experiment_path)
    exp = experiment.Experiment(experiment_path)
    slurm_command = exp.cfg.slurm.sbatch_command

    current_cycle = exp.current_cycle

    # Check if the current cycle is the last one and if it's done
    if current_cycle.index == len(exp.cycles) - 1 and exp.all_members_advanced:
        logger.error("Last cycle already advanced, experiment finished")
        sys.exit(1)

    # Generate and queue the advance members job array
    advance_job_id: int | None = None
    result = jobfiles.generate_advance_array_jobfile(exp, max_parallel)
    if result is not None:
        jf, pending_members = result
        res = external.runc([*slurm_command.split(" "), str(jf.resolve())])
        if res.returncode != 0:
            logger.error("Could not queue array jobfile, output:")
            logger.error(res.output)
            exit(1)

        advance_job_id = int(res.output.strip())
        logger.info(
            f"Queued {jf} with array job ID {advance_job_id} for {len(pending_members)} members"
        )

    if only_advance:
        if advance_job_id is not None:
            logger.info(f"Advance array JobID: {advance_job_id}")
        else:
            logger.info("No members needed advancing")
        return

    # Generate the analysis jobfile, queue it with dependency on the advance array
    queue_next_cycle = all_cycles
    if run_until is not None and current_cycle.index == run_until:
        logger.warning("Reached --run-until limit, will not queue next cycle")
        queue_next_cycle = False

    jf = jobfiles.generate_make_analysis_jobfile(
        exp,
        current_cycle.index,
        queue_next_cycle,
        run_postprocess,
        clean_scratch,
        run_until,
    )
    if advance_job_id is not None:
        res = external.runc(
            [
                *slurm_command.split(" "),
                f"--dependency=afterok:{advance_job_id}",
                str(jf.resolve()),
            ]
        )
    else:
        res = external.runc([*slurm_command.split(" "), str(jf.resolve())])

    analysis_jobid = int(res.output.strip())
    logger.info(f"Queued {jf} with ID {analysis_jobid}")

    if run_postprocess:
        jf = jobfiles.generate_postprocess_jobfile(
            exp, current_cycle.index, clean_scratch
        )
        res = external.runc(
            [
                *slurm_command.split(" "),
                f"--dependency=afterok:{analysis_jobid}",
                str(jf.resolve()),
            ]
        )
        logger.info(f"Queued {jf} with ID {res.output.strip()}")
