"""
WRF (and WRF-Chem) behind the `Model` interface.

There are two ways to carry a member from one cycle to the next
(`assimilation.cycling_mode`):

- `wrfinput`: real.exe makes a wrfinput and wrfbdy for every cycle, and the
  `cycled_variables` of the analysis are copied into the next cycle's wrfinput.
- `restart`: real.exe runs once for the whole experiment. Each member continues from
  its own restart file, into which the `state_variables` of the analysis are written
  (`restart.py`, `rebalance.py`). Only this mode can run several cycles as one WRF run
  (`[segments]`, see `segments.py` and `checkpoints.py`).
"""

import contextlib
import dataclasses
import datetime as dt
import json
import shutil
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import xarray as xr

from wrf_ensembly import (
    checkpoints,
    cycling,
    external,
    perturbations,
    rebalance,
    restart,
    update_bc,
    utils,
    wrf,
)
from wrf_ensembly.console import logger

from .base import Model, ModelRun

TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
"""Times in the names of wrfout and wrfrst files"""


class WrfModel(Model):
    name = "wrf"

    # Setting up

    @property
    def environment(self) -> dict[str, str]:
        return self.cfg.environment.wrf

    def install(self, force: bool) -> None:
        """
        Copies WRF and WPS from `directories.wrf_root`/`wps_root` into the experiment's
        work directory, and WRF into every member's directory.
        """

        # Check if WRF/WPS directories already exist
        if self.paths.work_wrf.exists():
            if force:
                logger.info(f"Removing existing WRF directory {self.paths.work_wrf}")
                shutil.rmtree(self.paths.work_wrf)
            else:
                raise FileExistsError(f"WRF directory {self.paths.work_wrf} already exists")
        if self.paths.work_wps.exists():
            if force:
                logger.info(f"Removing existing WPS directory {self.paths.work_wps}")
                shutil.rmtree(self.paths.work_wps)
            else:
                raise FileExistsError(f"WPS directory {self.paths.work_wps} already exists")

        # Copy WRF/WPS in the work directory
        # Depending on whether WRF was built using the traditional buildscripts or CMake, the binaries will be either in
        # `run/` with the exe suffix (old scripts) or inside `install/bin` without the suffix. CMake also installs the
        # data files in `install/run`. When both builds exist, the CMake one wins: a classic build left over in `run/`
        # is usually stale and would otherwise shadow it.
        wrf_required_binaries = ["wrf", "real", "tc", "ndown"]
        wrf_root = self.cfg.directories.wrf_root
        cmake_install = wrf_root / "install"
        if (cmake_install / "bin" / "wrf").exists():
            logger.info(f"Using the CMake build of WRF in {cmake_install}")
            run_dir = cmake_install / "run"
            binary_sources = {b: cmake_install / "bin" / b for b in wrf_required_binaries}
        else:
            logger.info(f"Using the classic build of WRF in {wrf_root / 'run'}")
            run_dir = wrf_root / "run"
            binary_sources = {b: run_dir / f"{b}.exe" for b in wrf_required_binaries}

        # Binaries are copied separately below, skip them (and the CMake symlinks to them) here
        binary_names = {
            *wrf_required_binaries,
            *(f"{b}.exe" for b in wrf_required_binaries),
        }
        shutil.copytree(
            run_dir,
            self.paths.work_wrf,
            symlinks=False,  # Maybe fix symlinks so that they are valid after getting copied?
            ignore=lambda d, names: (
                [n for n in names if n in binary_names] if Path(d) == run_dir else []
            ),
        )

        for binary_name, source in binary_sources.items():
            if source.exists():
                utils.copy(source, self.paths.work_wrf / f"{binary_name}.exe")
                logger.info(f"- {binary_name}: {source}")
            else:
                logger.warning(f"- {binary_name}: not found at {source}")

        logger.info(f"Copied WRF to {self.paths.work_wrf}")

        # For WPS the situation is similar, the binaries are either places at the root directory
        # or at `install/bin`. In this case we only need the binaries, geogrid's tables and
        # ungrib variable tables.
        self.paths.work_wps.mkdir(parents=True, exist_ok=True)
        geogrid_source = self.cfg.directories.wps_root / "geogrid"
        geogrid_target = self.paths.work_wps / "geogrid"
        geogrid_target.mkdir(exist_ok=True)
        for table in geogrid_source.glob("*TBL*"):
            utils.copy(table, geogrid_target / table.name)

        ungrib_vtable_source = self.cfg.directories.wps_root / "ungrib" / "Variable_Tables"
        ungrib_vtable_target = self.paths.work_wps / "ungrib" / "Variable_Tables"
        shutil.copytree(ungrib_vtable_source, ungrib_vtable_target)

        metgrid_table_source = self.cfg.directories.wps_root / "metgrid" / "METGRID.TBL.ARW"
        metgrid_table_target = self.paths.work_wps / "metgrid" / "METGRID.TBL.ARW"
        metgrid_table_target.parent.mkdir(exist_ok=True)
        utils.copy(metgrid_table_source, metgrid_table_target)
        # metgrid.exe opens METGRID.TBL, which WPS links to the ARW table
        (metgrid_table_target.parent / "METGRID.TBL").symlink_to(metgrid_table_target.name)

        wps_required_binaries = ["geogrid", "ungrib", "metgrid"]
        for binary_name in wps_required_binaries:
            target_path = self.paths.work_wps / f"{binary_name}.exe"
            classic_source_path = self.cfg.directories.wps_root / f"{binary_name}.exe"
            cmake_source_path = (
                self.cfg.directories.wps_root / "install" / "bin" / binary_name
            )
            # Same as WRF, prefer the CMake build if both exist
            if cmake_source_path.exists():
                utils.copy(cmake_source_path, target_path)
            elif classic_source_path.exists():
                utils.copy(classic_source_path, target_path)
            else:
                logger.error(f"WPS Binary {binary_name} not found")

        logger.info(f"Copied WPS to {self.paths.work_wps}")

        for j in range(self.cfg.assimilation.n_members):
            member_dir = self.paths.member_path(j)

            if member_dir.exists():
                if force:
                    logger.info(f"Removing existing member directory {member_dir}")
                    shutil.rmtree(member_dir)
                else:
                    raise FileExistsError(f"Member directory {member_dir} already exists")

            shutil.copytree(self.paths.work_wrf, member_dir, dirs_exist_ok=True)
            logger.info(f"Copied WRF to {member_dir}")

    def setup_member(self, member_i: int) -> None:
        utils.copy(
            self.paths.ic_path(member_i, 0),
            self.paths.member_path(member_i) / "wrfinput_d01",
        )
        self.prepare_member_boundaries(member_i, 0)
        logger.info(f"Member {member_i}: Copied initial and boundary conditions")

    def preprocess_job(self, command: Callable[..., str]) -> list[str]:
        commands = [command("setup"), command("geogrid")]

        # real.exe runs once per cycle, or once for the whole experiment in restart mode
        real_runs: list[dict[str, int]]
        if self.cfg.assimilation.cycling_mode == "restart":
            real_runs = [{}]
        else:
            real_runs = [{"cycle": cycle} for cycle in range(len(self.cycles))]

        if self.cfg.data.per_member_meteorology:
            n_members = self.cfg.assimilation.n_members
            commands.append(f"for MEMBER in {{0..{n_members - 1}}}; do")
            commands.append(command("ungrib", member="$MEMBER"))
            commands.append(command("metgrid"))
            commands.extend(
                command("real", member="$MEMBER", **real_args)
                for real_args in real_runs
            )
            commands.append("done")
            # Interpolates the chemistry for all members at once, in parallel
            commands.append(command("interpolate-chem"))
        else:
            commands.extend([command("ungrib"), command("metgrid")])
            commands.extend(command("real", **real_args) for real_args in real_runs)
            commands.append(command("interpolate-chem"))
        return commands

    # Running

    def run_member(
        self,
        member_idx: int,
        segment: range,
        cycle: cycling.CycleInformation,
        cores: int,
    ) -> ModelRun | None:
        """
        Runs wrf.exe for a member, to the next cycle or with segments to the end of
        `segment`. Initial and boundary condition files must already be in the member
        directory. Generates the namelist and sorts the output into the cycles.
        """

        member_path = self.paths.member_path(member_idx)

        # Locate WRF executable, icbc, ensure they all exist
        wrf_exe_path = (member_path / "wrf.exe").resolve()
        if not wrf_exe_path.exists():
            logger.error(
                f"Member {member_idx}: WRF executable not found at {wrf_exe_path}"
            )
            return None

        ic_path = self.member_initial_state(member_idx, self.exp.current_cycle_i).resolve()
        bc_path = (member_path / "wrfbdy_d01").resolve()
        if not ic_path.exists() or not bc_path.exists():
            logger.error(
                f"Member {member_idx}: Initial/boundary conditions not found at {ic_path} or {bc_path}"
            )
            return None

        # With segments, continue from the newest checkpoint of an earlier attempt, if
        # there is one. The member keeps its wrfbdy, which has the boundary increment of
        # the segment's start.
        restart_interval = None
        resume: tuple[dt.datetime, Path] | None = None
        plan = self.exp.segment_of(segment[0])
        if plan is not None:
            restart_interval = plan.checkpoint_interval_min
            resume = self.find_resume_point(member_idx, segment)

        if resume is not None and resume[0] == cycle.end:
            logger.info(
                f"Member {member_idx}: An earlier attempt already ran to the end, "
                "only filing its output"
            )
            start_time = end_time = None
        else:
            if resume is not None:
                resume_time, checkpoint = resume
                logger.info(f"Member {member_idx}: Resuming from {checkpoint}")
                # Copies left by earlier resumes, not the segment's initial state
                for f in member_path.glob("wrfrst_d01_*"):
                    if f != self.member_initial_state(member_idx, segment[0]):
                        f.unlink()
                utils.copy(checkpoint, member_path / checkpoint.name)
                cycle = dataclasses.replace(cycle, start=resume_time)

            ok, start_time, end_time = self._run_wrf(
                member_idx,
                cycle,
                cores,
                restart_interval,
                restart=True if resume is not None else None,
                watch_checkpoints=plan is not None,
            )
            if not ok:
                return None

        self.file_segment_outputs(member_idx, segment)

        simulated_s = int((cycle.end - cycle.start).total_seconds())
        return ModelRun(start_time, end_time, simulated_s)

    def _run_wrf(
        self,
        member_idx: int,
        cycle: cycling.CycleInformation,
        cores: int,
        restart_interval: int | None,
        restart: bool | None,
        watch_checkpoints: bool,
    ) -> tuple[bool, dt.datetime, dt.datetime]:
        """
        Writes the namelist and runs wrf.exe for a member from `cycle.start` to
        `cycle.forecast_end`. With `watch_checkpoints`, confirms and prunes the restart
        files it writes on the way (see `checkpoints.py`).

        Returns:
            Whether WRF succeeded, and when it started and ended
        """

        member_path = self.paths.member_path(member_idx)
        wrf_exe_path = (member_path / "wrf.exe").resolve()

        # Generate namelist
        wrf_namelist_path = member_path / "namelist.input"
        wrf.generate_wrf_namelist(
            self.cfg,
            cycle,
            True,
            wrf_namelist_path,
            member_idx,
            self.paths,
            restart_interval=restart_interval,
            restart=restart,
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

        rsl_file = member_path / "rsl.out.0000"
        watcher = contextlib.nullcontext()
        if watch_checkpoints:
            interval = (
                restart_interval
                if restart_interval is not None
                else int((cycle.end - cycle.start).total_seconds() // 60)
            )
            watcher = checkpoints.CheckpointWatcher(
                self.paths.scratch_restart_path(cycle.index, member_idx),
                rsl_file,
                cycle.start,
                dt.timedelta(minutes=interval),
                keep=(
                    None
                    if self.cfg.assimilation.keep_restart_files
                    else self.cfg.segments.keep_checkpoints
                ),
            )

        start_time = dt.datetime.now()
        with watcher:
            res = external.runc(cmd, cwd=member_path)
        end_time = dt.datetime.now()

        # Check output logs
        if not rsl_file.exists():
            logger.error(f"Member {member_idx}: RSL file not found at {rsl_file}")
            return False, start_time, end_time

        logger.add_log_file(rsl_file)
        rsl_content = rsl_file.read_text()

        if "SUCCESS COMPLETE WRF" not in rsl_content:
            logger.error(
                f"Member {member_idx}: wrf.exe failed with exit code {res.returncode}"
            )
            return False, start_time, end_time

        # Store logs in a zip file
        if logger.log_dir is not None:
            rsl_files = sorted(member_path.glob("rsl.*"))
            utils.zip_files(rsl_files, logger.log_dir / "rsl.zip")

        return True, start_time, end_time

    def find_resume_point(
        self, member_i: int, segment: range
    ) -> tuple[dt.datetime, Path] | None:
        """
        The newest checkpoint of an earlier attempt at a segment that the member can
        continue from, with its time: confirmed (WRF finished writing it), readable, and
        inside the segment. If it is at the segment's end, the earlier attempt finished
        and only its output needs filing. None to start from the beginning.
        """

        first, last = segment[0], segment[-1]
        start, end = self.cycles[first].start, self.cycles[last].end

        # Moved there only after an earlier attempt succeeded
        end_name = f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
        moved = self.paths.scratch_restart_path(last, member_i) / end_name
        if last != first and moved.exists():
            return end, moved

        directory = self.paths.scratch_restart_path(first, member_i)
        confirmed = checkpoints.confirmed_times(directory)
        for time, f in reversed(checkpoints.list_checkpoints(directory)):
            if not start < time <= end:
                continue
            if time not in confirmed:
                logger.info(
                    f"Member {member_i}: Not resuming from {f.name}, WRF didn't confirm "
                    "writing it"
                )
                continue
            if not checkpoints.is_complete(f, time):
                logger.warning(f"Member {member_i}: Not resuming from unreadable {f}")
                continue
            return time, f
        return None

    def file_segment_outputs(self, member_i: int, segment: range):
        """
        After a member ran a segment: WRF wrote all output and restart files into the
        first cycle's scratch directories. Move each wrfout into the cycle it belongs to
        (`start < time <= end`), delete the one at the segment start (as for a single
        cycle), and move the restart file at the segment end into the last cycle's
        restart directory, where `cycle` looks for it.

        Safe to run again on files that were already moved.
        """

        first, last = segment[0], segment[-1]
        first_dir = self.paths.scratch_forecasts_path(first, member_i)
        first_output = (
            first_dir / f"wrfout_d01_{self.cycles[first].start:%Y-%m-%d_%H:%M:%S}"
        )
        if first_output.exists():
            logger.info(f"Removing first output file {first_output}")
            first_output.unlink()

        if first == last:
            return

        for f in sorted(first_dir.glob("wrfout_d01_*")):
            time = wrf.wrfout_time(f.name)
            cycle_i = next(
                (
                    i
                    for i in segment
                    if self.cycles[i].start < time <= self.cycles[i].end
                ),
                None,
            )
            if cycle_i is None or cycle_i == first:
                continue
            target = self.paths.scratch_forecasts_path(cycle_i, member_i) / f.name
            target.parent.mkdir(parents=True, exist_ok=True)
            f.rename(target)
        logger.info(f"Member {member_i}: Sorted the output into cycles {first}-{last}")

        if self.cfg.assimilation.cycling_mode == "restart":
            end = self.cycles[last].end
            name = f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
            source = self.paths.scratch_restart_path(first, member_i) / name
            target = self.paths.scratch_restart_path(last, member_i) / name
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                logger.info(f"Member {member_i}: Moved {name} to {target.parent}")
            elif not target.exists():
                raise FileNotFoundError(f"WRF wrote no restart file at {source}")

    def remove_cycle_output(self, cycle_i: int, member_i: int) -> None:
        for f in self.paths.scratch_forecasts_path(cycle_i, member_i).glob("wrfout_*"):
            f.unlink()
        # Checkpoints and their confirmations
        restart_dir = self.paths.scratch_restart_path(cycle_i, member_i)
        if restart_dir.is_dir():
            for f in restart_dir.iterdir():
                if f.is_file():
                    f.unlink()

    # Output files

    @property
    def forecast_glob(self) -> str:
        return "wrfout_d01_*"

    def time_stamp(self, time: dt.datetime) -> str:
        return time.strftime(TIME_FORMAT)

    def forecast_file_name(self, time: dt.datetime) -> str:
        return f"wrfout_d01_{self.time_stamp(time)}"

    def first_processor(self):
        # Imported here, xwrf is only needed for the postprocessing
        from wrf_ensembly.processors import XWRFPostProcessor

        return XWRFPostProcessor()

    # Assimilation and cycling

    def prepare_filter(self) -> list[Path]:
        cycle_i = self.exp.current_cycle_i
        wrfout_name = self.forecast_file_name(self.exp.current_cycle.end)

        # Link wrfinput, required by filter to read coordinates
        wrfinput_path = self.paths.dart_work_dir / "wrfinput_d01"
        wrfinput_path.unlink(missing_ok=True)
        # Any wrfinput of the domain works, the grid doesn't change between cycles.
        # In restart mode, only the first cycle has one.
        wrfinput_cur_cycle_path = self.paths.ic_path(0, 0)
        wrfinput_path.symlink_to(wrfinput_cur_cycle_path)
        logger.info(f"Linked {wrfinput_path} to {wrfinput_cur_cycle_path}")

        # The prior is the latest forecast of each member
        return [
            self.paths.scratch_forecasts_path(cycle_i, member_i) / wrfout_name
            for member_i in range(self.cfg.assimilation.n_members)
        ]

    def make_analysis(self, member: int) -> None:
        """
        The analysis of a member is its wrfout at the end of the cycle, with the
        `state_variables` replaced by what the filter wrote.
        """

        cycle_i = self.exp.current_cycle_i
        cycle = self.exp.current_cycle

        # Copy forecasts to analysis directory
        wrfout_name = self.forecast_file_name(cycle.end)
        member_dir = f"member_{member:02d}"
        forecast_file = self.paths.scratch_forecasts_path(cycle_i) / member_dir / wrfout_name
        analysis_file = self.paths.scratch_analysis_path(cycle_i) / member_dir / wrfout_name
        utils.copy(forecast_file, analysis_file)

        dart_file = self.paths.scratch_dart_path(cycle_i) / f"dart_member_{member:02d}.nc"
        if not dart_file.exists():
            raise FileNotFoundError(f"Member {member}: {dart_file} does not exist")

        # Copy the state variables from the dart file to the analysis file
        logger.info(f"Member {member}: Copying state variables from {dart_file}")
        with (
            netCDF4.Dataset(dart_file, "r") as nc_dart,  # type: ignore
            netCDF4.Dataset(analysis_file, "r+") as nc_analysis,  # type: ignore
        ):
            for name in self.cfg.assimilation.state_variables:
                if name not in nc_dart.variables:
                    logger.warning(f"Member {member}: {name} not in dart file")
                    continue
                logger.info(f"Member {member}: Copying {name}")
                nc_analysis[name][:] = nc_dart[name][:]

            # Add experiment name and current cycle information to attributes
            # TODO Standardize this somehow? We must add metadata to all files!
            nc_analysis.experiment_name = self.cfg.metadata.name
            nc_analysis.current_cycle = cycle_i
            nc_analysis.cycle_start = cycle.start.strftime(TIME_FORMAT)
            nc_analysis.cycle_end = cycle.end.strftime(TIME_FORMAT)

    def cycle_member(self, member_i: int, use_forecast: bool) -> None:
        """
        In `wrfinput` cycling mode, the next cycle's wrfinput gets the `cycled_variables`
        from the analysis. In `restart` mode, the member continues from its restart file.
        """

        member_path = self.paths.member_path(member_i)
        if self.cfg.assimilation.cycling_mode == "restart":
            self._cycle_member_restart(member_i, use_forecast)
        else:
            self._cycle_member_wrfinput(member_i, use_forecast)

        # Remove forecast files, log files
        logger.info(f"Cleaning up member directory {member_path}")
        for f in member_path.glob("wrfout*"):
            logger.debug(f"Removing forecast file {f}")
            f.unlink()
        for f in member_path.glob("rsl*"):
            logger.debug(f"Removing log file {f}")
            f.unlink()

    def _cycle_member_wrfinput(self, member_i: int, use_forecast: bool) -> None:
        """`cycle_member` for the `wrfinput` cycling mode"""

        member_path = self.paths.member_path(member_i)
        next_cycle_i = self.exp.current_cycle_i + 1

        # Find analysis/forecast file to use
        wrfout_name = "wrfout_d01_" + self.exp.current_cycle.end.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        if use_forecast:
            analysis_file = (
                self.paths.scratch_forecasts_path(self.exp.current_cycle_i)
                / f"member_{member_i:02d}"
                / wrfout_name
            )
        else:
            analysis_file = (
                self.paths.scratch_analysis_path(self.exp.current_cycle_i)
                / f"member_{member_i:02d}"
                / wrfout_name
            )

        if not analysis_file.exists():
            raise FileNotFoundError(analysis_file)
        logger.info(f"Using {analysis_file} as analysis for member {member_i}")

        # Copy the initial & boundary condition files for the next cycle, as is
        icbc_target_file = member_path / "wrfinput_d01"
        utils.copy(self.paths.ic_path(member_i, next_cycle_i), icbc_target_file)
        self.prepare_member_boundaries(member_i, next_cycle_i)

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

    def _cycle_member_restart(self, member_i: int, use_forecast: bool) -> None:
        """
        `cycle_member` for the `restart` cycling mode: the member's restart file at the
        end of the current cycle is copied into its directory (the original is kept in
        scratch so the next cycle can be rerun). Unless cycling the forecast, the
        `state_variables` of the analysis are written into it and it is rebalanced.
        """

        member_path = self.paths.member_path(member_i)
        next_cycle_i = self.exp.current_cycle_i + 1
        end = self.exp.current_cycle.end
        source = self.paths.scratch_restart_path(self.exp.current_cycle_i, member_i) / (
            f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
        )
        if not source.exists():
            raise FileNotFoundError(source)
        logger.info(f"Member {member_i}: Continuing from {source}")

        # Only the next cycle's restart file belongs in the member directory. A stale
        # wrfinput would be ignored by wrf.exe, but could be modified by mistake.
        for f in member_path.glob("wrfrst_d01_*"):
            f.unlink()
        (member_path / "wrfinput_d01").unlink(missing_ok=True)

        initial_state = self.member_initial_state(member_i, next_cycle_i)
        utils.copy(source, initial_state)

        # Parameter perturbations are applied once and then only carried over in the
        # restart file. WRF only writes the fields its Registry marks for restart, so
        # without that a parameter would silently go back to its default.
        parameters = [
            name
            for name, var in self.cfg.perturbations.variables.items()
            if var.kind == "parameter"
        ]
        if parameters:
            with netCDF4.Dataset(initial_state, "r") as nc_restart:  # type: ignore
                missing = [
                    p for p in parameters if not restart.field_names(nc_restart, p)
                ]
            if missing:
                raise ValueError(
                    f'{", ".join(missing)} (perturbed with kind = "parameter") not in '
                    f"{source}. WRF only writes fields with the `r` flag in its Registry "
                    "to restart files, so the perturbation would be lost after the first "
                    "cycle. Add the flag and rebuild WRF."
                )

        if not use_forecast:
            analysis_file = (
                self.paths.scratch_analysis_path(self.exp.current_cycle_i)
                / f"member_{member_i:02d}"
                / f"wrfout_d01_{end:%Y-%m-%d_%H:%M:%S}"
            )
            if not analysis_file.exists():
                raise FileNotFoundError(analysis_file)
            with (
                netCDF4.Dataset(analysis_file, "r") as nc_analysis,  # type: ignore
                netCDF4.Dataset(initial_state, "r+") as nc_restart,  # type: ignore
            ):
                before = rebalance.balanced_fields(nc_restart)
                written = restart.write_state(
                    nc_restart, nc_analysis, self.cfg.assimilation.state_variables
                )
                rebalance.rebalance(nc_restart, before)
                nc_restart.experiment_name = self.cfg.metadata.name
            logger.info(
                f"Member {member_i}: Wrote the analysis ({', '.join(written)}) into "
                f"{initial_state.name} and rebalanced it"
            )

        self.prepare_member_boundaries(member_i, next_cycle_i)

    def update_bc(self, member_i: int) -> None:
        """
        Update the boundary conditions of a member with the changes made to its initial
        state (analysis, perturbations), see `member_reference_state`.
        """

        member_path = self.paths.member_path(member_i)
        reference = self.member_reference_state(member_i, self.exp.current_cycle_i)
        if not reference.exists():
            raise FileNotFoundError(
                f"Member {member_i}: The unmodified initial state is needed to update the "
                f"boundary conditions, but {reference} does not exist"
            )
        changed = update_bc.update_wrf_bc(
            self.member_initial_state(member_i, self.exp.current_cycle_i),
            reference,
            member_path / "wrfbdy_d01",
        )
        if changed:
            logger.info(f"Member {member_i}: Updated boundary conditions")
        else:
            logger.info(
                f"Member {member_i}: Initial state unchanged at the boundaries, "
                "boundary conditions left as they are"
            )

    def clean_old_state(self) -> None:
        """
        Deletes the WRF restart files in scratch that no cycle can start from anymore, in
        restart mode, unless `assimilation.keep_restart_files` is set or the cycle is in
        `assimilation.keep_restart_files_for_cycles`.

        The current cycle starts from the previous cycle's restart files, which are kept
        so it can be rerun. Older cycles' are deleted, and so are any restart files of
        the previous cycle other than the one at its end (written when a forecast
        extension reaches past the next restart time).
        """

        if (
            self.cfg.assimilation.cycling_mode != "restart"
            or self.cfg.assimilation.keep_restart_files
        ):
            return

        previous_i = self.exp.current_cycle_i - 1
        if previous_i < 0:
            return
        keep_cycles = set(self.cfg.assimilation.keep_restart_files_for_cycles)
        for cycle_dir in sorted(self.paths.scratch_restart.glob("cycle_*")):
            cycle_i = int(cycle_dir.name.removeprefix("cycle_"))
            if cycle_i < previous_i and cycle_i not in keep_cycles:
                logger.info(f"Removing restart files in {cycle_dir}")
                shutil.rmtree(cycle_dir)

        end = self.cycles[previous_i].end
        keep = f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
        for member_i in range(self.cfg.assimilation.n_members):
            member_dir = self.paths.scratch_restart_path(previous_i, member_i)
            for f in member_dir.glob("wrfrst_*"):
                if f.name != keep:
                    logger.info(f"Removing unused restart file {f}")
                    f.unlink()

    # A member's initial state and boundaries

    def member_initial_state(self, member_i: int, cycle_i: int) -> Path:
        """
        The file a member starts a cycle from, in its directory: the wrfinput, or in
        restart mode (after the first cycle) the WRF restart file at the cycle's start.
        """

        member_path = self.paths.member_path(member_i)
        if self.cfg.assimilation.cycling_mode == "restart" and cycle_i > 0:
            start = self.cycles[cycle_i].start
            return member_path / f"wrfrst_d01_{start:%Y-%m-%d_%H:%M:%S}"
        return member_path / "wrfinput_d01"

    def member_reference_state(self, member_i: int, cycle_i: int) -> Path:
        """
        The initial state of a member at a cycle before it was modified by the analysis or
        perturbations: real.exe's wrfinput, or in restart mode (after the first cycle) the
        restart file WRF wrote at the end of the previous cycle. `cycle` copies the latter
        into the member directory and keeps the original in scratch.
        """

        if self.cfg.assimilation.cycling_mode == "restart" and cycle_i > 0:
            start = self.cycles[cycle_i].start
            return self.paths.scratch_restart_path(cycle_i - 1, member_i) / (
                f"wrfrst_d01_{start:%Y-%m-%d_%H:%M:%S}"
            )
        return self.paths.ic_path(member_i, cycle_i)

    def prepare_member_boundaries(self, member_i: int, cycle_i: int) -> None:
        """
        Puts the boundary conditions for a cycle in the member's directory: the wrfbdy
        (in restart mode, the cycle's records of the experiment-long one, or the whole
        segment's if one starts at this cycle) and the lower boundary (wrflowinp, linked)
        if real.exe made one.
        """

        member_path = self.paths.member_path(member_i)
        cycle = self.cycles[cycle_i]
        bdy_target = member_path / "wrfbdy_d01"
        if self.cfg.assimilation.cycling_mode == "restart":
            end = cycle.forecast_end
            segment = self.exp.segment_of(cycle_i)
            if segment is not None and segment.first == cycle_i:
                end = self.cycles[segment.last].end
            n_records = wrf.extract_boundary_records(
                self.paths.bc_path(member_i, None),
                bdy_target,
                cycle.start,
                end,
            )
            logger.info(f"Member {member_i}: Extracted {n_records} boundary record(s)")
            lowinp = self.paths.lowinp_path(member_i, None)
        else:
            utils.copy(self.paths.bc_path(member_i, cycle_i), bdy_target)
            lowinp = self.paths.lowinp_path(member_i, cycle_i)

        lowinp_target = member_path / "wrflowinp_d01"
        lowinp_target.unlink(missing_ok=True)
        if lowinp.exists():
            lowinp_target.symlink_to(lowinp.resolve())
        elif self.cfg.wrf_namelist.get("physics", {}).get("sst_update", 0) == 1:
            raise FileNotFoundError(
                f"sst_update = 1 but there's no lower boundary file at {lowinp}"
            )

    # Perturbations

    def _perturbation_worker_args(self) -> dict[str, Any]:
        """
        The arguments `perturbations.generate_perturbations_for_cycle` needs, pulled out
        of the experiment. Kept in one place because the same set has to be handed to
        worker processes, which cannot be given `self`.
        """

        return {
            "perturbations_cfg": self.cfg.perturbations,
            "reapplied_parameters": perturbations.reapplied_parameters(self.cfg),
            "n_members": self.cfg.assimilation.n_members,
            "experiment_name": self.cfg.metadata.name,
            "ic_path": self.paths.ic_path(0, 0),
            "diag_dir": self.paths.data_diag,
        }

    def generate_perturbations(self, cycle_i: int):
        """
        Generates perturbations for a given cycle and stores them in `data/diag/perturbations`.
        """

        perturbations.generate_perturbations_for_cycle(
            cycle_i=cycle_i, **self._perturbation_worker_args()
        )

    def generate_perturbations_for_cycles(
        self, cycle_indices: Iterable[int], jobs: int
    ) -> Iterator[int]:
        """
        Generates perturbations for several cycles in parallel, yielding each cycle index
        as it finishes so the caller can record progress one cycle at a time.

        Args:
            cycle_indices: Which cycles to generate perturbations for
            jobs: How many cycles to process in parallel

        Yields:
            Each cycle index, once its perturbations have been written
        """

        cycle_indices = list(cycle_indices)
        worker = partial(
            perturbations.generate_perturbations_for_cycle,
            **self._perturbation_worker_args(),
        )

        with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as executor:
            results = executor.map(worker, cycle_indices)
            for cycle_i in cycle_indices:
                next(results)  # Raises here if the worker failed
                yield cycle_i

    def apply_perturbations(self, member_i: int):
        """
        Apply perturbations to the initial conditions of a member.
        Must be generated with `generate_perturbations` first.
        You should call this function once for every member, after the initial conditions
        are copied in the member directory (either during `ensemble setup` or `ensemble cycle`).

        This function will mostly not check the configuration, rather it will assume that
        the contents of the perturbation files are correct. This allows you to modify them
        outside of this program if needed. The exception is variables in the config that
        are not perturbed at this cycle (see `perturbations.applied_at_cycle`), which are
        skipped even if the file has them. Files generated before `kind = "parameter"`
        existed repeat the first cycle's field in every cycle, and in restart mode
        applying it again would compound it.
        """

        if member_i >= self.cfg.assimilation.n_members:
            raise ValueError(
                f"Member index {member_i} is out of bounds for {self.cfg.assimilation.n_members} members"
            )

        pert_file = (
            self.paths.data_diag
            / "perturbations"
            / f"perts_cycle_{self.exp.current_cycle_i}.nc"
        )
        if not pert_file.exists():
            raise FileNotFoundError(
                f"Perturbation file for cycle {self.exp.current_cycle_i} not found at {pert_file}"
            )
        initial_state = self.member_initial_state(member_i, self.exp.current_cycle_i)
        if not initial_state.exists():
            raise FileNotFoundError(
                f"Initial conditions file for member {member_i} not found at {initial_state}"
            )
        # In restart mode, after the first cycle, it's a restart file that needs rebalancing
        is_restart = initial_state.name.startswith("wrfrst")

        perts = xr.open_dataset(pert_file).sel(member=member_i)
        applied = perturbations.applied_at_cycle(self.cfg, self.exp.current_cycle_i)
        pert_config = {}
        for name, var in perts.data_vars.items():
            if name in self.cfg.perturbations.variables and name not in applied:
                logger.info(
                    f"Member {member_i}: Not perturbing {name}, it is not perturbed at "
                    f"cycle {self.exp.current_cycle_i}"
                )
                continue
            pert_config[name] = json.loads(var.attrs["cfg"])
        if not pert_config:
            logger.info(f"Member {member_i}: Nothing to perturb")
            return

        with netCDF4.Dataset(initial_state, "r+") as member_icbc:  # type: ignore
            before = rebalance.balanced_fields(member_icbc) if is_restart else None
            for name, cfg in pert_config.items():
                if "operation" not in cfg:
                    raise ValueError(
                        f"Perturbation config for {name} does not contain 'operation', parsed from netCDF attribute `cfg`"
                    )
                operation = cfg["operation"]

                logger.debug(f"Applying perturbation to {name} for member {member_i}")
                logger.debug(f"Perturbation config: {pert_config}")
                if not restart.field_names(member_icbc, name):
                    raise ValueError(f"Variable {name} not found in member IC/BC file.")

                field = perts[name].to_numpy()

                # Apply midcycle taper if configured (for non-first cycles using
                # reused perturbation fields)
                midcycle_taper_width = cfg.get("midcycle_taper_width", 0)
                if midcycle_taper_width > 0 and self.exp.current_cycle_i > 0:
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

                # Perturb the current state, `set_field` handles restart time levels
                current = member_icbc[restart.field_names(member_icbc, name)[-1]][:]
                match operation:
                    case "add":
                        perturbed = current + field
                    case "multiply":
                        perturbed = current * field
                    case "assign":
                        perturbed = np.broadcast_to(field, current.shape)
                    case _:
                        raise ValueError(f"Unknown perturbation operation: {operation}")
                restart.set_field(member_icbc, name, perturbed)

            if before is not None:
                rebalance.rebalance(member_icbc, before)

        logger.info(f"Applied perturbations to member {member_i}")
