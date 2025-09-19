import datetime as dt
import os
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from wrf_ensembly import (
    config,
    cycling,
    external,
    obs_sequence,
    perturbations,
    update_bc,
    utils,
    wrf,
)
from wrf_ensembly.console import logger

from .database import ExperimentDatabase
from .dataclasses import MemberStatus, RuntimeStatistics
from .paths import ExperimentPaths
from .observations import ExperimentObservations


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

    db: ExperimentDatabase
    obs: ExperimentObservations

    def __init__(self, experiment_path: Path):
        self.cfg = config.read_config(experiment_path / "config.toml")
        self.cycles = cycling.get_cycle_information(self.cfg)

        # Initialize database
        db_path = experiment_path / "status.db"
        self.db = ExperimentDatabase(db_path)

        # Initialize members in database
        self.db.initialize_members(self.cfg.assimilation.n_members)

        # Read experiment status from database
        self.load_status_from_db()

        self.paths = ExperimentPaths(experiment_path, self.cfg)
        self.obs = ExperimentObservations(self.cfg, self.cycles, self.paths)

    def load_status_from_db(self):
        """Load the status of the experiment from the database"""
        # Get experiment state
        self.current_cycle_i, self.filter_run, self.analysis_run = (
            self.db.get_experiment_state()
        )

        # Get member status
        members_data = self.db.get_all_members_status()
        self.members = []

        for member_i, advanced in members_data:
            # Get runtime statistics for this member
            runtime_stats = self.db.get_member_runtime_statistics(member_i)

            self.members.append(
                MemberStatus(
                    i=member_i, advanced=advanced, runtime_statistics=runtime_stats
                )
            )

        # Ensure we have the right number of members
        while len(self.members) < self.cfg.assimilation.n_members:
            member_i = len(self.members)
            self.members.append(
                MemberStatus(i=member_i, advanced=False, runtime_statistics=[])
            )

        self.members.sort(key=lambda m: m.i)

    def save_status_to_db(self):
        """Save the current status of the experiment to the database"""
        # Save experiment state
        self.db.set_experiment_state(
            self.current_cycle_i, self.filter_run, self.analysis_run
        )

        # Save member status
        for member in self.members:
            self.db.set_member_advanced(member.i, member.advanced)

    def set_next_cycle(self):
        """
        Update status to the next cycle
        """
        self.current_cycle_i += 1
        if self.current_cycle_i >= len(self.cycles):
            raise ValueError("No more cycles to run")
        self.filter_run = False
        self.analysis_run = False

        # Reset all members' advanced status in database
        self.db.reset_members_advanced()

        # Update local member status
        for member in self.members:
            member.advanced = False

        # Save to database
        self.save_status_to_db()

    def generate_perturbations(self, cycle_i: int):
        """
        Generates perturbations for a given cycle and stores them in `data/diag/perturbations`.
        """

        if len(self.cfg.perturbations.variables) == 0:
            logger.warning("No perturbations defined in config, skipping")
            return

        if self.cfg.perturbations.seed is not None:
            logger.warning(f"Setting random seed to {self.cfg.perturbations.seed}")
            np.random.seed(self.cfg.perturbations.seed)

        # Open the first wrfinput file to get the shape of each variable
        wrfinput = xr.open_dataset(self.paths.data_icbc / "wrfinput_d01_cycle_0")

        pert_file = self.paths.data_diag / "perturbations" / f"perts_cycle_{cycle_i}.nc"
        pert_file.parent.mkdir(parents=True, exist_ok=True)
        pert_file.unlink(missing_ok=True)

        # First, generate the perturbation field for each variable and member, and put them in a dataset
        perts = xr.Dataset()
        for var, pert_config in self.cfg.perturbations.variables.items():
            arr = np.stack(
                [
                    perturbations.generate_perturbation_field(
                        shape=wrfinput[var].shape,
                        mean=pert_config.mean,
                        sd=pert_config.sd,
                        rounds=pert_config.rounds,
                        boundary=pert_config.boundary,
                        min_value=pert_config.min_value,
                        max_value=pert_config.max_value,
                    )
                    for _ in range(self.cfg.assimilation.n_members)
                ],
                axis=0,
            )
            perts[var] = xr.DataArray(
                arr,
                dims=("member", *wrfinput[var].dims),
                coords={"member": range(self.cfg.assimilation.n_members)},
            )

        # Write the dataset to a netCDF file
        encoding = {
            var: {"zlib": True, "complevel": 5, "dtype": "float32"}
            for var in perts.data_vars
        }
        # Set chunksize for all variables: 1 on member and time, full size on rest
        for var in perts.data_vars:
            chunks = [1, 1] + [s for s in perts[var].shape[2:]]
            encoding[var]["chunksizes"] = chunks
        perts.attrs["experiment_name"] = self.cfg.metadata.name
        perts.attrs["cycle"] = cycle_i

        logger.info(f"Writing perturbations for {cycle_i} to {pert_file}")
        perts.to_netcdf(pert_file, encoding=encoding)

    def apply_perturbations(self, member_i: int):
        """
        Apply perturbations to the initial conditions of a member.
        Must be generated with `generate_perturbations` first.
        You should call this function once for every member, after the initial conditions
        are copied in the member directory (either during `ensemble setup` or `ensemble cycle`).
        """

        if member_i >= self.cfg.assimilation.n_members:
            raise ValueError(
                f"Member index {member_i} is out of bounds for {self.cfg.assimilation.n_members} members"
            )

        pert_file = (
            self.paths.data_diag
            / "perturbations"
            / f"perts_cycle_{self.current_cycle_i}.nc"
        )
        if not pert_file.exists():
            raise FileNotFoundError(
                f"Perturbation file for cycle {self.current_cycle_i} not found at {pert_file}"
            )

        perts = xr.open_dataset(pert_file).sel(member=member_i)

        with netCDF4.Dataset(
            self.paths.member_path(member_i) / "wrfinput_d01", "r+"
        ) as member_icbc:  # type: ignore
            for var, pert_config in self.cfg.perturbations.variables.items():
                logger.debug(f"Applying perturbation to {var} for member {member_i}")
                if var not in member_icbc.variables:
                    raise ValueError(f"Variable {var} not found in member IC/BC file.")

                field = perts[var].values
                if pert_config.operation == "add":
                    member_icbc[var][:] += field
                elif pert_config.operation == "multiply":
                    member_icbc[var][:] *= field
                else:
                    raise ValueError(
                        f"Unknown perturbation operation: {pert_config.operation}"
                    )

        logger.info(f"Applied perturbations to member {member_i}")

    def update_bc(self, member_i: int):
        """
        Run `update_wrf_bc` to update the boundary conditions of a member to match the initial conditions.
        """

        member_path = self.paths.member_path(member_i)
        icbc_target_file = member_path / "wrfinput_d01"
        bdy_target_file = member_path / "wrfbdy_d01"

        res = update_bc.update_wrf_bc(
            self.cfg,
            icbc_target_file,
            bdy_target_file,
            log_filename=f"update_bc_member_{member_i}.log",
        )
        if (
            res.returncode != 0
            or "update_wrf_bc Finished successfully" not in res.output
        ):
            logger.error(
                f"Member {member_i}: update_wrf_bc failed with exit code {res.returncode}"
            )
            logger.error(res.output)
            raise external.ExternalProcessFailed(res)

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
        cmd = [
            *self.cfg.slurm.mpirun_command.split(" "),
            "-n",
            str(cores),
            str(wrf_exe_path),
        ]

        start_time = dt.datetime.now()
        res = external.runc(cmd, cwd=member_path)
        end_time = dt.datetime.now()

        # Check output logs
        rsl_file = member_path / "rsl.out.0000"
        if not rsl_file.exists():
            logger.error(f"Member {member_idx}: RSL file not found at {rsl_file}")
            return False

        logger.add_log_file(rsl_file)
        rsl_content = rsl_file.read_text()

        if "SUCCESS COMPLETE WRF" not in rsl_content:
            logger.error(
                f"Member {member_idx}: wrf.exe failed with exit code {res.returncode}"
            )
            return False

        # Store logs in a zip file
        if logger.log_dir is not None:
            rsl_files = sorted(member_path.glob("rsl.*"))
            utils.zip_files(rsl_files, logger.log_dir / "rsl.zip")

        # Delete first output file
        first_output = (
            self.paths.scratch_forecasts_path(self.current_cycle_i, member_idx)
            / f"wrfout_d01_{cycle.start.strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        if first_output.exists():
            logger.info(f"Removing first output file {first_output}")
            first_output.unlink()

        # Update member status using database instead of file locking
        # The database handles concurrency internally
        self.members[member_idx].advanced = True

        # Add runtime statistics to database
        self.db.add_runtime_statistics(
            member_idx,
            self.current_cycle_i,
            start_time,
            end_time,
            int((end_time - start_time).total_seconds()),
        )

        # Update member status in database
        self.db.set_member_advanced(member_idx, True)

        # Update local runtime statistics
        self.members[member_idx].runtime_statistics.append(
            RuntimeStatistics(
                cycle=self.current_cycle_i,
                start=start_time,
                end=end_time,
                duration_s=int((end_time - start_time).total_seconds()),
            )
        )

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
        if self.cfg.data.per_member_meteorology:
            wrfinput_cur_cycle_path = (
                self.paths.data_icbc
                / "member_00"
                / f"wrfinput_d01_member_00_cycle_{self.current_cycle_i}"
            )
        else:
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
                *self.cfg.slurm.mpirun_command.split(" "),
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
        obs_sequence.obs_seq_to_nc(
            self.cfg.directories.dart_root, obs_seq_final, obs_seq_final_nc
        )

        self.filter_run = True
        self.save_status_to_db()
        return True

    def cycle_member(self, member_i: int, use_forecast: bool):
        """
        Merge the IC/BC of a member with the analysis from the previous cycle.
        Must be done for all members after finishing a set of forward runs and running
        the filter. If you have no observations and filter is skipped, run this step with
        use_forecast=True to use the forecast from the previous cycle as the analysis.

        Args:
            member_i: Index of the member to cycle
            use_forecast: Use the forecast from the previous cycle as the analysis
        """

        member_path = self.paths.member_path(member_i)
        next_cycle_i = self.current_cycle_i + 1

        # Find analysis/forecast file to use
        wrfout_name = "wrfout_d01_" + self.current_cycle.end.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        if use_forecast:
            analysis_file = (
                self.paths.scratch_forecasts_path(self.current_cycle_i)
                / f"member_{member_i:02d}"
                / wrfout_name
            )
        else:
            analysis_file = (
                self.paths.scratch_analysis_path(self.current_cycle_i)
                / f"member_{member_i:02d}"
                / wrfout_name
            )

        if not analysis_file.exists():
            raise FileNotFoundError(analysis_file)
        logger.info(f"Using {analysis_file} as analysis for member {member_i}")

        # Copy the initial & boundary condition files for the next cycle, as is
        # First check if there is a member-specific IC/BC file for the next cycle,
        # otherwise use the ensemble default
        icbc_file = (
            self.paths.data_icbc
            / f"member_{member_i:02d}"
            / f"wrfinput_d01_member_{member_i:02d}_cycle_{next_cycle_i}"
        )
        if icbc_file.exists():
            logger.info(f"Using member-specific IC/BC file {icbc_file}")
        else:
            icbc_file = self.paths.data_icbc / f"wrfinput_d01_cycle_{next_cycle_i}"

        # First check if there is a member-specific BC file for the next cycle,
        # otherwise use the ensemble default
        bdy_file = (
            self.paths.data_icbc
            / f"member_{member_i:02d}"
            / f"wrfbdy_d01_member_{member_i:02d}_cycle_{next_cycle_i}"
        )
        if bdy_file.exists():
            logger.info(f"Using member-specific BC file {bdy_file}")
        else:
            bdy_file = self.paths.data_icbc / f"wrfbdy_d01_cycle_{next_cycle_i}"

        icbc_target_file = member_path / "wrfinput_d01"
        bdy_target_file = member_path / "wrfbdy_d01"

        utils.copy(icbc_file, icbc_target_file)
        utils.copy(bdy_file, bdy_target_file)

        # Copy cycled variables from the analysis file to the IC file
        with (
            netCDF4.Dataset(analysis_file, "r") as nc_analysis,  # type: ignore
            netCDF4.Dataset(icbc_target_file, "r+") as nc_icbc,  # type: ignore
        ):
            for name in self.cfg.assimilation.cycled_variables:
                if name not in nc_analysis.variables:
                    logger.warning(f"Member {member_i}: {name} not in analysis file")
                    continue
                logger.info(f"Member {member_i}: Copying {name}")
                nc_icbc[name][:] = nc_analysis[name][:]

            # Add experiment name to attributes
            nc_icbc.experiment_name = self.cfg.metadata.name

        # Remove forecast files, log files
        logger.info(f"Cleaning up member directory {member_path}")
        for f in member_path.glob("wrfout*"):
            logger.debug(f"Removing forecast file {f}")
            f.unlink()
        for f in member_path.glob("rsl*"):
            logger.debug(f"Removing log file {f}")
            f.unlink()

    def set_wrf_environment(self):
        """
        Adds the environment variables from config's `environment.wrf` to the current environment
        """

        for key, value in self.cfg.environment.wrf.items():
            os.environ[key] = value

    def set_dart_environment(self):
        """
        Adds the environment variables from config's `environment.dart` to the current environment
        """

        for key, value in self.cfg.environment.dart.items():
            os.environ[key] = value

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
