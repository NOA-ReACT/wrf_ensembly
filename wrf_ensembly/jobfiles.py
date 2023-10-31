from typing import Optional
from pathlib import Path

from wrf_ensembly.console import logger
from wrf_ensembly import templates, config


def generate_advance_jobfiles(experiment_path: Path, cfg: config.Config, cycle: int):
    """
    Generates a SLURM jobfile to advance a given member in a given cycle.

    Returns:
        A list of Path objects to the jobfiles
    """

    jobfile_directory = experiment_path / "jobfiles"
    jobfile_directory.mkdir(parents=True, exist_ok=True)

    # Write one jobfile for each member
    base_cmd = f"{cfg.slurm.python_command} -m wrf_ensembly ensemble advance-member {experiment_path.resolve()}"

    files = []
    for i in range(cfg.assimilation.n_members):
        job_name = f"{cfg.metadata.name}_cycle_{cycle}_member_{i}"
        jobfile = jobfile_directory / f"cycle_{cycle}_advance_member_{i}.job.sh"

        jobfile.write_text(
            templates.generate(
                "slurm_job.sh.j2",
                slurm_directives=cfg.slurm.directives_large | {"job-name": job_name},
                env_modules=cfg.slurm.env_modules,
                commands=[f"{base_cmd} {i}"],
            )
        )

        logger.info(f"Jobfile for member {i} written to {jobfile}")
        files.append(jobfile)
    return files


def generate_make_analysis_jobfile(
    experiment_path: Path,
    cfg: config.Config,
    cycle: Optional[int] = None,
    queue_next_cycle: bool = False,
):
    """
    Generates a jobfile for the `filter`, `analysis` and `cycle` steps. At runtime, the
    script will check whether observations exist for the current cycle. If they do, all
    steps (filter, analysis, cycle) are run. If they don't, only the cycle step is run
    with the `--use-forecast` flag.

    Returns:
        A Path object to the jobfile
    """

    jobfile_directory = experiment_path / "jobfiles"
    jobfile_directory.mkdir(parents=True, exist_ok=True)

    obs_file = experiment_path / "obs" / f"cycle_{cycle}.obs_seq"
    if not obs_file.exists():
        logger.warning(
            f"Observation file {obs_file} does not exist! Filter won't run if it is not created for cycle {cycle}"
        )

    job_name = f"{cfg.metadata.name}_analysis_cycle_{cycle}"
    jobfile = jobfile_directory / f"cycle_{cycle}_make_analysis.job.sh"

    base_cmd = f"{cfg.slurm.python_command} -m wrf_ensembly ensemble %SUBCOMMAND% {experiment_path.resolve()}"
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
        commands.append(
            f"{cfg.slurm.python_command} -m wrf_ensembly slurm run-experiment {experiment_path.resolve()} --only-next-cycle --resume --in-waves"
        )

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_directives=cfg.slurm.directives_small | {"job-name": job_name},
            env_modules=cfg.slurm.env_modules,
            commands=commands,
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")

    return jobfile
