from pathlib import Path
from typing import Any, Optional

from wrf_ensembly import experiment, templates
from wrf_ensembly.console import logger


def _key_value_to_argument(key: str, value: Any) -> str:
    key = key.replace("_", "-")
    if isinstance(value, bool) and value:
        return f"--{key}" if value else ""
    return f"--{key} {value}"


def _build_command(base_cmd: str, subcommand: str, **kwargs) -> str:
    """
    Helper to construct command strings with consistent formatting.

    Args:
        base_cmd: The base command string the {{subcommand}} placeholder.
        subcommand: The subcommand to be inserted into the base command.
        **kwargs: Additional keyword arguments to be appended as command options (`--key value`). Any underscores in the key will be replaced with hyphens.
    """

    cmd = base_cmd.format(subcommand=subcommand)
    if kwargs:
        cmd += " " + " ".join(_key_value_to_argument(k, v) for k, v in kwargs.items())
    return cmd


def generate_preprocess_jobfile(exp: experiment.Experiment) -> Path:
    """
    Generate a SLURM jobfile to run the preprocessing steps (WPS and real).

    Returns:
        A Path object to the jobfile
    """

    exp.paths.jobfiles.mkdir(parents=True, exist_ok=True)

    base_cmd = f"{exp.cfg.slurm.command_prefix} wrf-ensembly {exp.paths.experiment_path.resolve()} preprocess {{subcommand}}"
    commands = [
        _build_command(base_cmd, "setup"),
        _build_command(base_cmd, "geogrid"),
    ]

    if exp.cfg.data.per_member_meteorology:
        commands.append(f"for MEMBER in {{0..{len(exp.members) - 1}}}; do")
        commands.append(_build_command(base_cmd, "ungrib", member="$MEMBER"))
        commands.append(_build_command(base_cmd, "metgrid"))
        commands.extend(
            [
                _build_command(base_cmd, "real", cycle=cycle, member="$MEMBER")
                for cycle in range(len(exp.cycles))
            ]
        )
        commands.append(_build_command(base_cmd, "interpolate-chem", member="$MEMBER"))
        commands.append("done")
    else:
        commands.extend(
            [
                _build_command(base_cmd, "ungrib"),
                _build_command(base_cmd, "metgrid"),
            ]
            + [
                _build_command(base_cmd, "real", cycle=cycle)
                for cycle in range(len(exp.cycles))
            ]
            + [
                _build_command(base_cmd, "interpolate-chem"),
            ]
        )

    jobfile = exp.paths.jobfiles / "preprocess.sh"
    jobfile.parent.mkdir(parents=True, exist_ok=True)

    dynamic_directives = {
        "job-name": f"{exp.cfg.metadata.name}_preprocess",
        "output": f"{exp.paths.logs_slurm.resolve()}/%j-preprocess.out",
    }

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_directives=exp.cfg.slurm.directives_large | dynamic_directives,
            env_modules=exp.cfg.slurm.env_modules,
            commands=commands,
            pre_commands=exp.cfg.slurm.pre_commands,
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")

    return jobfile


def generate_advance_jobfiles(exp: experiment.Experiment) -> list[Path]:
    """
    Generates a SLURM jobfile to advance a given member in a given cycle.

    Returns:
        A list of Path objects to the jobfiles
    """

    exp.paths.jobfiles.mkdir(parents=True, exist_ok=True)

    # Write one jobfile for each member
    base_cmd = f"{exp.cfg.slurm.command_prefix} wrf-ensembly {exp.paths.experiment_path.resolve()} ensemble advance-member"

    files = []
    for member in exp.members:
        if member.advanced:
            logger.info(f"Member {member.i} already advanced. Skipping...")
            continue

        i = member.i
        jobfile = exp.paths.jobfiles / f"advance_member_{i}.job.sh"

        dynamic_directives = {
            "job-name": f"{exp.cfg.metadata.name}_advance_member_{i}",
            "output": f"{exp.paths.logs_slurm.resolve()}/%j-advance_member_{i}.out",
        }

        jobfile.write_text(
            templates.generate(
                "slurm_job.sh.j2",
                slurm_directives=exp.cfg.slurm.directives_large | dynamic_directives,
                env_modules=exp.cfg.slurm.env_modules,
                commands=[_build_command(base_cmd, "", member=i)],
                pre_commands=exp.cfg.slurm.pre_commands,
            )
        )

        logger.info(f"Jobfile for member {i} written to {jobfile}")
        files.append(jobfile)
    return files


def generate_make_analysis_jobfile(
    exp: experiment.Experiment,
    cycle: Optional[int] = None,
    queue_next_cycle: bool = False,
    compute_postprocess: bool = False,
    clean_scratch: bool = False,
    run_until: int | None = None,
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
        compute_postprocess: Whether to compute postprocess after the analysis step.
        delete_members: Whether to delete the members' forecasts after processing them.
        run_until: --run-until argument to pass to `run-experiment`

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

    pert_file = exp.paths.data_diag / "perturbations" / f"perts_cycle_{cycle}.nc"
    pert_file = pert_file.resolve()
    if not pert_file.exists():
        logger.warning(
            f"Perturbation file {pert_file} does not exist! Apply perturbations won't run if it is not created for cycle {cycle}"
        )

    jobfile = exp.paths.jobfiles / f"cycle_{cycle}_make_analysis.job.sh"

    dynamic_directives = {
        "job-name": f"{exp.cfg.metadata.name}_analysis_cycle_{cycle}",
        "output": f"{exp.paths.logs_slurm.resolve()}/%j-analysis_cycle_{cycle}.out",
    }

    base_cmd = f"{exp.cfg.slurm.command_prefix} wrf-ensembly {exp.paths.experiment_path} ensemble {{subcommand}}"
    commands = [
        f"if [ -f {obs_file} ]; then",
        _build_command(base_cmd, "filter"),
        _build_command(base_cmd, "analysis"),
        _build_command(base_cmd, "cycle"),
        "else",
        _build_command(base_cmd, "cycle", use_forecast=True),
        "fi",
    ]
    if exp.cfg.perturbations.apply_perturbations_every_cycle:
        commands += [
            f"if [ -f {pert_file} ]; then",
            _build_command(base_cmd, "apply-perturbations"),
            "fi",
        ]
    commands += [
        _build_command(base_cmd, "update-bc"),
    ]

    if queue_next_cycle:
        args = ""
        if compute_postprocess:
            args += " --run-postprocess"
            if clean_scratch:
                args += " --clean-scratch"
        if run_until is not None:
            args += f" --run-until {run_until}"

        commands.append(
            f"{exp.cfg.slurm.command_prefix} wrf-ensembly {exp.paths.experiment_path} slurm run-experiment {args}"
        )

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_directives=exp.cfg.slurm.directives_small | dynamic_directives,
            env_modules=exp.cfg.slurm.env_modules,
            commands=commands,
            pre_commands=exp.cfg.slurm.pre_commands,
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")

    return jobfile


def generate_postprocess_jobfile(
    exp: experiment.Experiment,
    cycle: int,
    clean: bool = False,
) -> Path:
    """
    Generates a jobfile to run the `postprocessing` steps.

    Args:
        exp: The experiment
        cycle: The cycle for which to run the postprocess commands.
        clean: Whether to clean the scratch directory after postprocessing.

    Returns:
        A Path object to the jobfile
    """

    exp.paths.jobfiles.mkdir(parents=True, exist_ok=True)

    jobs = exp.cfg.slurm.directives_postprocess.get("ntasks", -1)
    if jobs == -1:
        logger.warning(
            "ntasks not set in `slurm.directives_postprocess``. Using default value of 1"
        )
        jobs = 1

    jobfile = exp.paths.jobfiles / f"cycle_{cycle}_postprocess.job.sh"
    dynamic_directives = {
        "job-name": f"{exp.cfg.metadata.name}_postprocess_cycle_{cycle}",
        "output": f"{exp.paths.logs_slurm.resolve()}/%j-postprocess.out",
    }

    base_cmd = f"{exp.cfg.slurm.command_prefix} wrf-ensembly {exp.paths.experiment_path.resolve()} postprocess {{subcommand}}"
    commands = [
        _build_command(
            base_cmd,
            "process-pipeline",
            cycle=cycle,
            jobs=exp.cfg.postprocess.processor_cores,
        ),
    ]
    if exp.cfg.postprocess.compute_ensemble_statistics_in_job:
        commands.append(
            _build_command(
                base_cmd,
                "statistics",
                cycle=cycle,
                jobs=exp.cfg.postprocess.statistics_cores,
            )
        )
    commands.append(
        _build_command(
            base_cmd,
            "concatenate",
            cycle=cycle,
            jobs=exp.cfg.postprocess.concatenate_cores,
        )
    )

    if clean:
        commands.append(_build_command(base_cmd, "clean"))

    max_used_jobs = max(
        exp.cfg.postprocess.processor_cores,
        exp.cfg.postprocess.statistics_cores,
        exp.cfg.postprocess.concatenate_cores,
    )
    if max_used_jobs < int(jobs):
        logger.warning(
            f"Number of SLURM tasks ({jobs}) is less than the maximum number of jobs used ({max_used_jobs})."
        )
    if max_used_jobs > int(jobs):
        logger.warning(
            f"Number of SLURM tasks ({jobs}) is greater than the maximum number of jobs used ({max_used_jobs})."
        )

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_directives=exp.cfg.slurm.directives_postprocess | dynamic_directives,
            env_modules=exp.cfg.slurm.env_modules,
            commands=commands,
            pre_commands=exp.cfg.slurm.pre_commands,
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")

    return jobfile
