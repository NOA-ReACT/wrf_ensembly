import sys
from pathlib import Path
from typing import Optional

import click

from wrf_ensembly import external, jobfiles
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly import experiment


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
    """Create a SLURM jobfile to advance each member of the ensemble"""

    logger.setup(f"slurm-advance-members", experiment_path)
    exp = experiment.Experiment(experiment_path)

    logger.info(f"Writing jobfiles advancing members...")
    jobfiles.generate_advance_jobfiles(exp)


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
@click.argument("cycle", type=int)
@click.option(
    "--clean-scratch",
    is_flag=True,
    help="Requires --run-postprocess. If set, the individual member's forecasts are deleted from the scratch directories",
)
@pass_experiment_path
def postprocess(experiment_path: Path, cycle: int, clean_scratch: bool):
    """Create a SLURM jobfile to postprocess the WRF output"""

    logger.setup(f"slurm-postprocessing", experiment_path)
    exp = experiment.Experiment(experiment_path)

    logger.info(f"Writing jobfile for postprocessing...")
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
    help="Queue postprocessing for all cycles up to this one",
)
@pass_experiment_path
def queue_all_postprocessing(
    experiment_path: Path,
    clean_scratch: bool,
    first_cycle: Optional[int],
    last_cycle: Optional[int],
):
    """Queue postprocessing for all cycles of the experiment"""

    logger.setup("slurm-queue-all-postprocessing", experiment_path)
    exp = experiment.Experiment(experiment_path)

    min_cycle = 0
    max_cycle = len(exp.cycles)
    if last_cycle is not None and last_cycle <= max_cycle:
        max_cycle = last_cycle
    if first_cycle is not None and first_cycle > 0:
        min_cycle = first_cycle

    for i in range(min_cycle, max_cycle):
        logger.info(f"Queueing postprocessing for cycle {i}...")
        jf = jobfiles.generate_postprocess_jobfile(exp, i, clean_scratch)

        res = external.runc(
            [
                *exp.cfg.slurm.sbatch_command.split(" "),
                str(jf.resolve()),
            ]
        )
        logger.info(f"Queued {jf} with ID {res.output.strip()}")


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
@pass_experiment_path
def run_experiment(
    experiment_path: Path,
    all_cycles: bool,
    run_postprocess: bool,
    clean_scratch: bool,
    only_advance: bool,
    run_until: int | None,
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

    # Generate all member jobfiles, queue them and keep jobids
    jfs = jobfiles.generate_advance_jobfiles(exp)

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

    if only_advance:
        logger.info(f"First JobID: {min(ids)}, last JobID: {max(ids)}")
        return

    # Generate the analysis jobfile, queue it and keep jobid
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
    if len(ids) > 0:
        dependency = "--dependency=afterok:" + ":".join(map(str, ids))
        res = external.runc([*slurm_command.split(" "), dependency, str(jf.resolve())])
    else:
        res = external.runc([*slurm_command.split(" "), str(jf.resolve())])

    analysis_jobid = int(res.output.strip())
    ids.append(analysis_jobid)
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
        ids.append(int(res.output.strip()))

    logger.info(f"First JobID: {min(ids)}, last JobID: {max(ids)}")
