from pathlib import Path
from typing import Optional

from wrf_ensembly import config, experiment, templates
from wrf_ensembly.console import logger


def generate_preprocess_jobfile(exp: experiment.Experiment) -> Path:
    """
    Generate a SLURM jobfile to run the preprocessing steps (WPS and real).

    Returns:
        A Path object to the jobfile
    """

    exp.paths.jobfiles.mkdir(parents=True, exist_ok=True)

    base_cmd = f"{exp.cfg.slurm.python_command} -m wrf_ensembly preprocess %SUBCOMMAND% {exp.paths.experiment_path.resolve()}"
    commands = [
        base_cmd.replace("%SUBCOMMAND%", "setup"),
        base_cmd.replace("%SUBCOMMAND%", "geogrid"),
        base_cmd.replace("%SUBCOMMAND%", "ungrib"),
        base_cmd.replace("%SUBCOMMAND%", "metgrid"),
    ] + [
        base_cmd.replace("%SUBCOMMAND%", "real") + f" {cycle}"
        for cycle in range(len(exp.cycles))
    ]

    jobfile = exp.paths.jobfiles / "preprocess.sh"
    jobfile.parent.mkdir(parents=True, exist_ok=True)

    jobname = f"{exp.cfg.metadata.name}_preprocess"

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_directives=exp.cfg.slurm.directives_large | {"job-name": jobname},
            env_modules=exp.cfg.slurm.env_modules,
            commands=commands,
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")

    return jobfile


def generate_advance_jobfiles(exp: experiment.Experiment, cycle: int) -> list[Path]:
    """
    Generates a SLURM jobfile to advance a given member in a given cycle.

    Returns:
        A list of Path objects to the jobfiles
    """

    exp.paths.jobfiles.mkdir(parents=True, exist_ok=True)

    # Write one jobfile for each member
    base_cmd = f"{exp.cfg.slurm.python_command} -m wrf_ensembly ensemble advance-member {exp.paths.experiment_path.resolve()}"

    files = []
    for i in range(exp.cfg.assimilation.n_members):
        job_name = f"{exp.cfg.metadata.name}_cycle_{cycle}_member_{i}"
        jobfile = exp.paths.jobfiles / f"cycle_{cycle}_advance_member_{i}.job.sh"

        jobfile.write_text(
            templates.generate(
                "slurm_job.sh.j2",
                slurm_directives=exp.cfg.slurm.directives_large
                | {"job-name": job_name},
                env_modules=exp.cfg.slurm.env_modules,
                commands=[f"{base_cmd} {i}"],
            )
        )

        logger.info(f"Jobfile for member {i} written to {jobfile}")
        files.append(jobfile)
    return files


def generate_make_analysis_jobfile(
    exp: experiment.Experiment,
    cycle: Optional[int] = None,
    queue_next_cycle: bool = False,
    compute_statistics: bool = False,
    delete_members: bool = False,
):
    """
    Generates a jobfile for the `filter`, `analysis` and `cycle` steps. At runtime, the
    script will check whether observations exist for the current cycle. If they do, all
    steps (filter, analysis, cycle) are run. If they don't, only the cycle step is run
    with the `--use-forecast` flag.

    Args:
        exp: The experiment
        cycle: The cycle for which to run the analysis command. If None, all cycles will be processed.
        queue_next_cycle: Whether to queue the next cycle after the current one is done.
        compute_statistics: Whether to compute statistics after the analysis step.
        delete_members: Whether to delete the members' forecasts after processing them.

    Returns:
        A Path object to the jobfile
    """

    exp.paths.jobfiles.mkdir(parents=True, exist_ok=True)

    obs_file = exp.paths.obs / f"cycle_{cycle}.obs_seq"
    obs_file = obs_file.resolve()
    if not obs_file.exists():
        logger.warning(
            f"Observation file {obs_file} does not exist! Filter won't run if it is not created for cycle {cycle}"
        )

    job_name = f"{exp.cfg.metadata.name}_analysis_cycle_{cycle}"
    jobfile = exp.paths.jobfiles / f"cycle_{cycle}_make_analysis.job.sh"

    base_cmd = f"{exp.cfg.slurm.python_command} -m wrf_ensembly ensemble %SUBCOMMAND% {exp.paths.experiment_path}"
    commands = [
        f"if [ -f {obs_file} ]; then",
        base_cmd.replace("%SUBCOMMAND%", "filter"),
        base_cmd.replace("%SUBCOMMAND%", "analysis"),
        base_cmd.replace("%SUBCOMMAND%", "cycle"),
        "else",
        base_cmd.replace("%SUBCOMMAND%", "cycle") + " --use-forecast",
        "fi",
    ]

    if queue_next_cycle:
        args = ""
        if compute_statistics:
            args += " --compute-statistics"
            if delete_members:
                args += " --delete-members"

        commands.append(
            f"{exp.cfg.slurm.python_command} -m wrf_ensembly slurm run-experiment {exp.paths.experiment_path} {args}"
        )

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_directives=exp.cfg.slurm.directives_small | {"job-name": job_name},
            env_modules=exp.cfg.slurm.env_modules,
            commands=commands,
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")

    return jobfile


def generate_statistics_jobfile(
    exp: experiment.Experiment,
    cycle: Optional[int] = None,
    delete_members: bool = False,
) -> Path:
    """
    Generates a jobfile to run the `statistics` step.

    Args:
        exp: The experiment
        cycle: The cycle for which to run the statistics command. If None, all cycles will be processed.
        delete_members: Whether to delete the members' forecasts after processing them.

    Returns:
        A Path object to the jobfile
    """

    exp.paths.jobfiles.mkdir(parents=True, exist_ok=True)

    if cycle is not None:
        job_name = f"{exp.cfg.metadata.name}_statistics_cycle_{cycle}"
        jobfile = exp.paths.jobfiles / f"cycle_{cycle}_statistics.job.sh"
    else:
        job_name = f"{exp.cfg.metadata.name}_statistics"
        jobfile = exp.paths.jobfiles / "statistics.job.sh"

    cmd = f"{exp.cfg.slurm.python_command} -m wrf_ensembly ensemble statistics {exp.paths.experiment_path}"
    if cycle is not None:
        cmd += f" {cycle}"
    if delete_members:
        cmd += " --remove-member-forecasts --remove-member-analysis"

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_directives=exp.cfg.slurm.directives_small | {"job-name": job_name},
            env_modules=exp.cfg.slurm.env_modules,
            commands=[cmd],
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")

    return jobfile
