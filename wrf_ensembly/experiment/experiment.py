import datetime as dt
from pathlib import Path

from wrf_ensembly import config, cycling, external, observations, utils, wrf
from wrf_ensembly.console import logger

from .dataclasses import ExperimentStatus, MemberStatus, RuntimeStatistics
from .paths import ExperimentPaths


class Experiment:
    """
    An ensemble assimilation experiment
    """

    cfg: config.Config
    cycles: list[cycling.CycleInformation]
    current_cycle_i: int
    filter_run: bool
    analysis_run: bool
    paths: ExperimentPaths
    members: list[MemberStatus] = []

    status_file_path: Path

    def __init__(self, experiment_path: Path):
        self.cfg = config.read_config(experiment_path / "config.toml")
        self.cycles = cycling.get_cycle_information(self.cfg)

        # Read experiment status from status.toml
        self.status_file_path = experiment_path / "status.toml"
        if self.status_file_path.exists():
            self.read_status()
        else:
            # If the file does not exist, maybe this is a fresh experiment. Assume we are at cycle 0
            self.current_cycle_i = 0
            self.filter_run = False
            self.analysis_run = False
            self.members = [
                MemberStatus(i=i, advanced=False, runtime_statistics=[])
                for i in range(self.cfg.assimilation.n_members)
            ]

        self.paths = ExperimentPaths(experiment_path, self.cfg)

    def read_status(self):
        """Read the status of the experiment from status.toml"""

        status = ExperimentStatus.from_toml(self.status_file_path.read_text())
        self.current_cycle_i = status.current_cycle
        self.filter_run = status.filter_run
        self.analysis_run = status.analysis_run
        self.members = [
            MemberStatus(
                i=member_status.i,
                advanced=member_status.advanced,
                runtime_statistics=member_status.runtime_statistics,
            )
            for member_status in status.members
        ]

        # Make sure list is of the correct length and sorted correctly
        if len(self.members) != self.cfg.assimilation.n_members:
            raise ValueError(
                f"Number of members in status file ({len(self.members)}) does not match configuration ({self.cfg.assimilation.n_members})"
            )
        self.members.sort(key=lambda m: m.i)

    def write_status(self):
        """
        Write the current status of the experiment to status.toml
        """

        status = ExperimentStatus(
            self.current_cycle_i, self.filter_run, self.analysis_run, self.members
        )

        self.status_file_path.write_text(status.to_toml())

    def set_next_cycle(self):
        """
        Update status to the next cycle
        """

        self.current_cycle_i += 1
        if self.current_cycle_i >= len(self.cycles):
            raise ValueError("No more cycles to run")
        self.filter_run = False
        self.analysis_run = False
        for member in self.members:
            member.advanced = False

    def advance_member(self, member_idx: int, cores: int) -> bool:
        """
        Run WRF to advance a member to the next cycle.
        Initial and boundary condition files must already be present in the member directory.
        Will generate the appropriate namelist. Will move forecasts to the output directory.

        Args:
            member: Index of the member to advance
            cores: Number of cores to use

        Returns:
            True if the member was advanced successfully
        """

        member = self.members[member_idx]
        member_path = self.paths.member_path(member_idx)
        cycle = self.cycles[self.current_cycle_i]

        # Refuse to run model if already advanced
        if member.advanced:
            logger.error(f"Member {member_idx} already advanced")
            return False

        # Locate WRF executable, icbc, ensure they all exist
        wrf_exe_path = (member_path / "wrf.exe").resolve()
        if not wrf_exe_path.exists():
            logger.error(
                f"Member {member_idx}: WRF executable not found at {wrf_exe_path}"
            )
            return False

        ic_path = (member_path / "wrfinput_d01").resolve()
        bc_path = (member_path / "wrfbdy_d01").resolve()
        if not ic_path.exists() or not bc_path.exists():
            logger.error(
                f"Member {member_idx}: Initial/boundary conditions not found at {ic_path} or {bc_path}"
            )
            return False

        # Generate namelist
        wrf_namelist_path = member_path / "namelist.input"
        wrf.generate_wrf_namelist(
            self.cfg, cycle, True, wrf_namelist_path, member_idx, self.paths
        )

        # Clean old log files
        for f in member_path.glob("rsl.*"):
            f.unlink()

        # Run WRF
        logger.info(f"Running WRF for member {member_idx}...")
        cmd = [self.cfg.slurm.mpirun_command, "-n", str(cores), str(wrf_exe_path)]

        start_time = dt.datetime.now()
        res = external.runc(cmd, cwd=member_path)
        end_time = dt.datetime.now()

        # Check output logs
        for f in member_path.glob("rsl.*"):
            logger.add_log_file(f)
        rsl_file = member_path / "rsl.out.0000"
        if not rsl_file.exists():
            logger.error(f"Member {member_idx}: RSL file not found at {rsl_file}")
            return False

        rsl_content = rsl_file.read_text()

        if "SUCCESS COMPLETE WRF" not in rsl_content:
            logger.error(
                f"Member {member_idx}: wrf.exe failed with exit code {res.returncode}"
            )
            return False

        # Update member status, take some basic precautions against other processes
        # doing the same.
        with utils.LockFile(self.status_file_path):
            # First read the status file again to make sure it hasn't changed
            self.read_status()

            # Add info about the current run
            self.members[member_idx].advanced = True
            self.members[member_idx].runtime_statistics.append(
                RuntimeStatistics(
                    cycle=self.current_cycle_i,
                    start=start_time,
                    end=end_time,
                    duration_s=int((end_time - start_time).total_seconds()),
                )
            )

            # Write to disk
            self.write_status()

        return True

    def filter(self) -> bool:
        """
        Run the Kalman Filter for the current cycle

        Returns:
            True if the filter was run successfully
        """

        if self.filter_run:
            logger.error("Filter already run for current cycle")
            return False
        if not self.all_members_advanced:
            logger.error("Not all members have been advanced")
            return False

        dart_dir = self.cfg.directories.dart_root / "models" / "wrf" / "work"

        # Grab observations
        obs_seq = dart_dir / "obs_seq.out"
        obs_seq.unlink(missing_ok=True)

        obs_file = self.paths.obs / f"cycle_{self.current_cycle_i}.obs_seq"
        if not obs_file.exists():
            logger.error(f"Observation file for current cycle not found at {obs_file}")
            return False
        utils.copy(obs_file, obs_seq)

        # Write lists of input and output files
        # The input list is the latest forecast for each member
        wrfout_name = "wrfout_d01_" + self.current_cycle.end.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        priors = [
            self.paths.scratch_forecasts_path(self.current_cycle_i, member_i)
            / wrfout_name
            for member_i in range(0, self.cfg.assimilation.n_members)
        ]
        posterior = [
            self.paths.scratch_dart_path(self.current_cycle_i)
            / f"dart_{prior.parent.name}.nc"
            for prior in priors
        ]

        dart_input_txt = dart_dir / "input_list.txt"
        dart_input_txt.write_text("\n".join(str(prior.resolve()) for prior in priors))
        logger.info(f"Wrote {dart_input_txt}")
        dart_output_txt = dart_dir / "output_list.txt"
        dart_output_txt.write_text("\n".join(str(post.resolve()) for post in posterior))
        logger.info(f"Wrote {dart_output_txt}")

        self.paths.scratch_dart_path(self.current_cycle_i).mkdir(exist_ok=True)

        # Link wrfinput, required by filter to read coordinates
        wrfinput_path = dart_dir / "wrfinput_d01"
        wrfinput_path.unlink(missing_ok=True)
        wrfinput_cur_cycle_path = (
            self.paths.data_icbc / f"wrfinput_d01_cycle_{self.current_cycle_i}"
        )
        wrfinput_path.symlink_to(wrfinput_cur_cycle_path)
        logger.info(f"Linked {wrfinput_path} to {wrfinput_cur_cycle_path}")

        # Run filter
        if self.cfg.assimilation.filter_mpi_tasks == 1:
            logger.info("Running filter w/out MPI")
            cmd = ["./filter"]
        else:
            logger.info(
                f"Using MPI to run filter, n={self.cfg.assimilation.filter_mpi_tasks}"
            )
            cmd = [
                self.cfg.slurm.mpirun_command,
                "-n",
                str(self.cfg.assimilation.filter_mpi_tasks),
                "./filter",
            ]
        res = external.runc(cmd, dart_dir, log_filename="filter.log")
        if res.returncode != 0 or "Finished ... at" not in res.output:
            logger.error(f"filter failed with exit code {res.returncode}")
            return False

        # Keep obs_seq.final for diagnostics, convert to netcdf
        obs_seq_final = dart_dir / "obs_seq.final"
        utils.copy(
            obs_seq,
            self.paths.data_diag / f"cycle_{self.current_cycle_i}.obs_seq.final",
        )
        obs_seq_final_nc = self.paths.data_diag / f"cycle_{self.current_cycle_i}.nc"
        observations.obs_seq_to_nc(
            self.cfg.directories.dart_root, obs_seq_final, obs_seq_final_nc
        )

        self.filter_run = True
        return True

    @property
    def current_cycle(self) -> cycling.CycleInformation:
        """
        Get the current cycle
        """

        return self.cycles[self.current_cycle_i]

    @property
    def all_members_advanced(self) -> bool:
        """
        Check if all ensemble members have been advanced
        """

        return all(m.advanced for m in self.members)
