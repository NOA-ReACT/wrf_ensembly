import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

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
from wrf_ensembly.fortran_namelists import write_namelist

from .database import ExperimentDatabase
from .dataclasses import MemberStatus, RuntimeStatistics
from .inflation import InflationConfig
from .observations import ExperimentObservations
from .paths import ExperimentPaths
from .state_machine import (
    CycleState,
    ExperimentStateError,
    ExperimentStateMachine,
    StateTransition,
)


def _generate_perturbations_for_cycle(
    cycle_i: int,
    perturbations_cfg: config.PerturbationsConfig,
    n_members: int,
    experiment_name: str,
    ic_path: Path,
    diag_dir: Path,
):
    """
    Generates perturbations for a given cycle and stores them in the diag directory.
    This is a standalone function (no Experiment instance) so it can be safely pickled
    for multiprocessing.

    For cycles > 0 where all reused variables have `different_field_every_cycle=False`
    and no new fields need generating, a symlink to cycle 0's file is created instead.
    The midcycle taper is applied at perturbation application time rather than here.
    """

    if len(perturbations_cfg.variables) == 0:
        logger.warning("No perturbations defined in config, skipping")
        return

    if perturbations_cfg.seed is not None:
        logger.warning(f"Setting random seed to {perturbations_cfg.seed}")
        np.random.seed(perturbations_cfg.seed)

    pert_dir = diag_dir / "perturbations"
    pert_dir.mkdir(parents=True, exist_ok=True)
    pert_file = pert_dir / f"perts_cycle_{cycle_i}.nc"
    pert_file.unlink(missing_ok=True)

    if cycle_i > 0:
        # Check which variables need new fields vs reusing cycle 0
        needs_new_field = any(
            pert_config.perturb_every_cycle and pert_config.different_field_every_cycle
            for pert_config in perturbations_cfg.variables.values()
        )
        needs_reuse = any(
            pert_config.perturb_every_cycle
            and not pert_config.different_field_every_cycle
            for pert_config in perturbations_cfg.variables.values()
        )

        if needs_reuse and not needs_new_field:
            # All reused variables come from cycle 0 — symlink instead of copying
            first_cycle_pert_file = pert_dir / "perts_cycle_0.nc"
            if not first_cycle_pert_file.exists():
                raise FileNotFoundError(
                    f"Perturbation file for first cycle not found at {first_cycle_pert_file}"
                )
            pert_file.symlink_to(first_cycle_pert_file.name)
            logger.info(
                f"Symlinked perturbations for cycle {cycle_i} to cycle 0 (same fields)"
            )
            return

    # Open the first wrfinput file to get the shape of each variable
    wrfinput = xr.open_dataset(ic_path)

    # Generate the perturbation fields
    perts = xr.Dataset()
    for var, pert_config in perturbations_cfg.variables.items():
        if cycle_i > 0 and (
            not pert_config.perturb_every_cycle
            or not pert_config.different_field_every_cycle
        ):
            continue

        logger.debug(f"\tVariable: {var}, config: {pert_config}")
        arr = perturbations.generate_perturbation_ensemble(
            n_members=n_members,
            shape=wrfinput[var].shape,
            mean=pert_config.mean,
            sd=pert_config.sd,
            sigma=pert_config.gaussian_sigma,
            boundary=pert_config.boundary,
            min_value=pert_config.min_value,
            max_value=pert_config.max_value,
        )
        perts[var] = xr.DataArray(
            arr,
            dims=("member", *wrfinput[var].dims),
            coords={"member": range(n_members)},
            attrs={"cfg": json.dumps(pert_config.__dict__)},
        )

    # For cycles > 0, also include reused variables from cycle 0
    if cycle_i > 0:
        first_cycle_pert_file = pert_dir / "perts_cycle_0.nc"
        if not first_cycle_pert_file.exists():
            raise FileNotFoundError(
                f"Perturbation file for first cycle not found at {first_cycle_pert_file}"
            )
        first_cycle_perts = xr.open_dataset(first_cycle_pert_file)
        for var, pert_config in perturbations_cfg.variables.items():
            if (
                not pert_config.perturb_every_cycle
                or pert_config.different_field_every_cycle
            ):
                continue

            logger.info(f"Reusing perturbation field from first cycle for {var}")
            if var not in first_cycle_perts:
                raise ValueError(
                    f"Variable {var} not found in perturbation file for first cycle"
                )
            perts[var] = first_cycle_perts[var]
            perts[var].attrs = first_cycle_perts[var].attrs

    # Write the dataset to a netCDF file
    encoding = {
        var: {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32",
            "least_significant_digit": 3,
        }
        for var in perts.data_vars
    }
    # Set chunksize for all variables: 1 on member and time, full size on rest
    for var in perts.data_vars:
        chunks = [1, 1] + [s for s in perts[var].shape[2:]]
        encoding[var]["chunksizes"] = chunks
    perts.attrs["experiment_name"] = experiment_name
    perts.attrs["cycle"] = cycle_i

    logger.info(f"Writing perturbations for {cycle_i} to {pert_file}")
    perts.to_netcdf(pert_file, encoding=encoding)

    wrfinput.close()


class Experiment:
    """
    An ensemble assimilation experiment
    """

    cfg: config.Config
    cycles: list[cycling.CycleInformation]
    current_cycle_i: int
    paths: ExperimentPaths
    members: list[MemberStatus] = []

    db: ExperimentDatabase
    obs: ExperimentObservations
    inflation: InflationConfig
    state_machine: ExperimentStateMachine

    def __init__(self, experiment_path: Path):
        self.cfg = config.read_config(experiment_path / "config.toml")
        self.cycles = cycling.get_cycle_information(self.cfg)

        # Initialize database
        db_path = experiment_path / "status.db"
        self.db = ExperimentDatabase(db_path)

        # Initialize members in database and load status
        with self.db as db_conn:
            db_conn.initialize_members(self.cfg.assimilation.n_members)

        # Read experiment status from database
        self.load_status_from_db()

        self.paths = ExperimentPaths(experiment_path, self.cfg)
        self.obs = ExperimentObservations(self.cfg, self.cycles, self.paths)
        self.inflation = InflationConfig.from_config(
            self.cfg, self.paths.data_inflation, self.paths.dart_work_dir
        )

    def load_status_from_db(self):
        """Load the status of the experiment from the database"""

        with self.db as db_conn:
            # Get experiment state (ignore legacy filter_run/analysis_run)
            self.current_cycle_i, _, _, cycle_state_str = db_conn.get_experiment_state()

            # Get member status
            members_data = db_conn.get_all_members_status()
            self.members = []

            for member_i, advanced in members_data:
                # Get runtime statistics for this member
                runtime_stats = db_conn.get_member_runtime_statistics(member_i)

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

        # Initialize state machine
        self.state_machine = ExperimentStateMachine(
            n_cycles=len(self.cycles), current_cycle_idx=self.current_cycle_i
        )

        # Load the current cycle state
        try:
            cycle_state = CycleState(cycle_state_str)
        except ValueError:
            logger.warning(
                f"Invalid cycle state '{cycle_state_str}' in database, using INITIALIZED"
            )
            cycle_state = CycleState.INITIALIZED

        self.state_machine.current_cycle.current_state = cycle_state

    @property
    def filter_run(self) -> bool:
        """Check if filter has been run for the current cycle (derived from state machine)."""
        state = self.state_machine.current_cycle.current_state
        return state in (
            CycleState.FILTER_COMPLETE,
            CycleState.ANALYSIS_COMPLETE,
            CycleState.CYCLE_COMPLETE,
        )

    @property
    def analysis_run(self) -> bool:
        """Check if analysis has been run for the current cycle (derived from state machine)."""
        state = self.state_machine.current_cycle.current_state
        return state in (CycleState.ANALYSIS_COMPLETE, CycleState.CYCLE_COMPLETE)

    def save_status_to_db(self):
        """Save the current status of the experiment to the database"""

        with self.db as db_conn:
            # Get current cycle state from state machine
            cycle_state_str = self.state_machine.current_cycle.current_state.value

            # Save experiment state (filter_run/analysis_run derived from state machine)
            db_conn.set_experiment_state(
                self.current_cycle_i,
                self.filter_run,  # Property, derived from state
                self.analysis_run,  # Property, derived from state
                cycle_state_str,
            )

            # Save member status in batch
            member_statuses = [(member.i, member.advanced) for member in self.members]
            db_conn.set_members_advanced_batch(member_statuses)

    def set_next_cycle(self):
        """
        Update status to the next cycle
        """

        self.current_cycle_i += 1
        if self.current_cycle_i >= len(self.cycles):
            raise ValueError("No more cycles to run")

        # Advance state machine to next cycle and reset to INITIALIZED
        self.state_machine.advance_to_next_cycle()
        self.state_machine.current_cycle.current_state = CycleState.INITIALIZED

        with self.db as db_conn:
            # Reset all members' advanced status in database
            db_conn.reset_members_advanced()

            # Update local member status
            for member in self.members:
                member.advanced = False

            # Save experiment state to database (filter_run/analysis_run derived from state)
            db_conn.set_experiment_state(
                self.current_cycle_i, False, False, "initialized"
            )

    def setup_dart(self):
        """Prepare DART working directory by writing namelist and linking files"""

        dart_dir = self.paths.dart_work_dir

        # Prepare inflation: restore restart files and apply namelist overrides
        filter_namelist_path = dart_dir / "input.nml"
        logger.info(f"Writing DART filter namelist to {filter_namelist_path}")
        dart_namelist = self.inflation.prepare_cycle(
            self.cfg.dart_namelist, self.current_cycle_i
        )
        dart_namelist["filter_nml"]["ens_size"] = self.cfg.assimilation.n_members
        write_namelist(dart_namelist, filter_namelist_path)

        # Copy files if defined in config
        for file_cfg in self.cfg.extra_dart_files:
            source_path = Path(file_cfg.source).resolve()
            if file_cfg.destination_name is None:
                target_path = dart_dir / source_path.name
            else:
                target_path = dart_dir / file_cfg.destination_name
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Extra DART file not found at {source_path}"
                )
            utils.copy(source_path, target_path)
            logger.info(f"Copied extra DART file from {source_path} to {target_path}")

    def generate_perturbations(self, cycle_i: int):
        """
        Generates perturbations for a given cycle and stores them in `data/diag/perturbations`.
        """
        _generate_perturbations_for_cycle(
            cycle_i=cycle_i,
            perturbations_cfg=self.cfg.perturbations,
            n_members=self.cfg.assimilation.n_members,
            experiment_name=self.cfg.metadata.name,
            ic_path=self.paths.ic_path(0, 0),
            diag_dir=self.paths.data_diag,
        )

    def apply_perturbations(self, member_i: int):
        """
        Apply perturbations to the initial conditions of a member.
        Must be generated with `generate_perturbations` first.
        You should call this function once for every member, after the initial conditions
        are copied in the member directory (either during `ensemble setup` or `ensemble cycle`).

        This function will not check the configuration, rather it will assume that the
        contents of the perturbation files are correct. This allows you to modify them
        outside of this program if needed.
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
        wrfinput_path = self.paths.member_path(member_i) / "wrfinput_d01"
        if not wrfinput_path.exists():
            raise FileNotFoundError(
                f"Initial conditions file for member {member_i} not found at {wrfinput_path}"
            )

        perts = xr.open_dataset(pert_file).sel(member=member_i)
        pert_config = {
            name: json.loads(var.attrs["cfg"]) for name, var in perts.data_vars.items()
        }

        with netCDF4.Dataset(wrfinput_path, "r+") as member_icbc:  # type: ignore
            for name, cfg in pert_config.items():
                if "operation" not in cfg:
                    raise ValueError(
                        f"Perturbation config for {name} does not contain 'operation', parsed from netCDF attribute `cfg`"
                    )
                operation = cfg["operation"]

                logger.debug(f"Applying perturbation to {name} for member {member_i}")
                logger.debug(f"Perturbation config: {pert_config}")
                if name not in member_icbc.variables:
                    raise ValueError(f"Variable {name} not found in member IC/BC file.")

                field = perts[name].to_numpy()

                # Apply midcycle taper if configured (for non-first cycles using
                # reused perturbation fields)
                midcycle_taper_width = cfg.get("midcycle_taper_width", 0)
                if midcycle_taper_width > 0 and self.current_cycle_i > 0:
                    logger.debug(
                        f"Applying mid-cycle taper (width={midcycle_taper_width}) to {name}"
                    )
                    taper = perturbations.edge_taper(
                        field.shape[-2], field.shape[-1], midcycle_taper_width
                    )
                    if operation == "add":
                        field = field * taper
                    elif operation == "multiply":
                        field = 1 + (field - 1) * taper
                    else:
                        raise ValueError(
                            "Cannot use 'assign' operation with midcycle taper"
                        )

                match operation:
                    case "add":
                        member_icbc[name][:] += field
                    case "multiply":
                        member_icbc[name][:] *= field
                    case "assign":
                        member_icbc[name][:] = field
                    case _:
                        raise ValueError(f"Unknown perturbation operation: {operation}")

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

        # Update member status using database context manager
        # The database handles concurrency internally
        self.members[member_idx].advanced = True

        # Add runtime statistics and update member status in single transaction
        with self.db as db_conn:
            db_conn.add_runtime_statistics(
                member_idx,
                self.current_cycle_i,
                start_time,
                end_time,
                int((end_time - start_time).total_seconds()),
            )

            db_conn.set_member_advanced(member_idx, True)

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
        Run the Kalman Filter for the current cycle.

        Checks and updates the experiment state machine. Raises ExperimentStateError if
        the experiment is not in a valid state to run the filter.

        Returns:
            True if the filter was run successfully
        """

        n_advanced = sum(1 for m in self.members if m.advanced)

        # Auto-recover: if in ADVANCING_MEMBERS and all members done, transition
        if (
            self.state_machine.current_cycle.current_state
            == CycleState.ADVANCING_MEMBERS
        ):
            if n_advanced == self.cfg.assimilation.n_members:
                self.state_machine.current_cycle.transition(
                    StateTransition.ALL_MEMBERS_ADVANCED,
                    n_advanced,
                    self.cfg.assimilation.n_members,
                )
                self.save_status_to_db()
                logger.info("All members advanced - starting filter")
            else:
                raise ExperimentStateError(
                    f"Only {n_advanced}/{self.cfg.assimilation.n_members} members advanced"
                )

        # Validate state
        can_run, error = self.state_machine.can_run_filter(
            self.current_cycle_i, n_advanced, self.cfg.assimilation.n_members
        )
        if not can_run:
            req_actions = self.state_machine.current_cycle.get_required_actions()
            raise ExperimentStateError(
                f"Cannot run filter: {error}\nNext required action: {req_actions}"
            )

        # Skip filter if no observation file and no inflation enabled
        obs_file = self.paths.obs / f"cycle_{self.current_cycle_i:03}.obs_seq"
        if not obs_file.exists():
            logger.error(f"No observation file found at {obs_file}, cannot run filter!")
            return False

        self.setup_dart()

        # Grab observations
        dart_dir = self.paths.dart_work_dir
        obs_seq = dart_dir / "obs_seq.out"
        obs_seq.unlink(missing_ok=True)
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
        wrfinput_cur_cycle_path = self.paths.ic_path(0, self.current_cycle_i)
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
            obs_seq_final,
            self.paths.data_diag / f"cycle_{self.current_cycle_i}.obs_seq.final",
        )
        obs_seq_final_nc = self.paths.data_diag / f"cycle_{self.current_cycle_i}.nc"
        obs_sequence.obs_seq_to_nc(
            self.cfg.directories.dart_root, obs_seq_final, obs_seq_final_nc
        )

        # Stash inflation files so we can use them next cycle
        self.inflation.stash_restart_files(self.current_cycle_i)

        # Transition state
        self.state_machine.current_cycle.transition(StateTransition.FILTER_COMPLETE)
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
        icbc_file = self.paths.ic_path(member_i, next_cycle_i)
        bdy_file = self.paths.bc_path(member_i, next_cycle_i)

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
