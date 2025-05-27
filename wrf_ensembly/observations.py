import datetime as dt
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from mashumaro import field_options
from mashumaro.mixins.toml import DataClassTOMLMixin

from wrf_ensembly import external, fortran_namelists, utils
from wrf_ensembly.config import Config
from wrf_ensembly.console import logger


@dataclass
class Observation:
    """Represents one observation file"""

    start_date: dt.datetime = field(
        metadata=field_options(
            deserialize=lambda v: dt.datetime.strptime(v, "%Y%m%dT%H%M%S").replace(
                tzinfo=dt.timezone.utc
            )
        )
    )
    """Timestamp of the first datapoint inside the file"""

    end_date: dt.datetime = field(
        metadata=field_options(
            deserialize=lambda v: dt.datetime.strptime(v, "%Y%m%dT%H%M%S").replace(
                tzinfo=dt.timezone.utc
            )
        )
    )
    """Timestamp of the last datapoint inside the file"""

    path: Path
    """Path to the physical file"""


@dataclass
class ObservationGroup(DataClassTOMLMixin):
    """
    Represents one group of observations, in which all files share the same format, use
    the same observation operator and are of the same DART kind
    """

    kind: str
    """DART kind for the observation"""

    converter: str
    """
    Path to converter executable. Must take input and output file as first and second
    arguments. You can add any other arguments you want to pass to the converter in the
    TOML file, they will be passed to the converter as well, but before the input and
    output file.
    """

    files: list[Observation]
    """Files in this group"""

    cwd: Path = Path(".")
    """Working directory for the converter"""

    def get_files_in_window(
        self, start: dt.datetime, end: dt.datetime
    ) -> list[Observation]:
        """Returns all files that overlap with the given window"""

        return [f for f in self.files if f.start_date <= end and start <= f.end_date]

    def convert_file(self, file: Observation, output_file: Path):
        """Converts a file to DART obs_seq format by running the appropriate converter"""

        res = external.run(
            external.ExternalProcess(
                [*self.converter.split(" "), file.path, output_file],
                cwd=Path(self.converter).parent,
                log_filename=f"converter_{file.path.name}.log",
            )
        )
        if res.returncode != 0:
            raise external.ExternalProcessFailed(res)
        if not output_file.exists():
            logger.error("Converter did not produce an output file!")
            raise external.ExternalProcessFailed(res)
        return res


def read_observation_group(path: Path) -> ObservationGroup:
    """Read an observation group file from the disk"""

    return ObservationGroup.from_toml(path.read_text())


def read_observations(dir: Path) -> dict[str, ObservationGroup]:
    """Reads all observation group files in a directory"""
    return {f.stem: read_observation_group(f) for f in dir.glob("*.toml")}


def join_obs_seq(
    cfg: Config,
    obs_seq_files: list[Path],
    output_file: Path,
    obs_kinds: list[str],
    binary_obs_sequence: bool = False,
):
    """
    Runs the `obs_sequence_tool` to join a list of obs_seq files into one.
    The tool must be compiled w/ DART and be available in the model's work directory.

    TODO Add temporal and spatial cropping options

    Args:
        cfg: Experiment configuration object
        obs_seq_files: List of obs_seq files to join
        output_file: Where to write the output file
        obs_kinds: List of observation kinds to include in the output file
        binary_obs_sequence: Whether to write the output file as a binary obs_seq file, defaults to False (i.e., ASCII)
    """

    dart_work_dir = cfg.directories.dart_root / "models" / "wrf" / "work"
    obs_seq_tool = dart_work_dir / "obs_sequence_tool"

    nml = {
        "obs_sequence_tool_nml": {
            "filename_seq_list": "./input_list",
            "filename_out": output_file.resolve(),
            "gregorian_cal": True,
        },
        "obs_sequence_nml": {"write_binary_obs_sequence": binary_obs_sequence},
        "obs_kind_nml": {
            "assimilate_these_obs_types": obs_kinds,
        },
        "location_nml": {},
        "utilities_nml": {
            "TERMLEVEL": 1,
            "module_details": False,
            "logfilename": "obs_sequence_tool.out",
            "nmlfilename": "obs_sequence_tool.nml",
            "write_nml": "file",
        },
    }

    # Link obs_sequence_tool inside a temp directory, create namelist
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        obs_seq_tool_ln = tmp_dir / "obs_sequence_tool"
        obs_seq_tool_ln.symlink_to(obs_seq_tool)

        # Write the namelist file
        namelist_path = tmp_dir / "input.nml"
        fortran_namelists.write_namelist(nml, namelist_path)

        # Write all input files inside a text file
        filelist_path = tmp_dir / "input_list"
        filelist_path.write_text("\n".join(str(f.resolve()) for f in obs_seq_files))
        print(filelist_path.read_text())

        # Call obs_sequence_tool, check results
        res = external.run(
            external.ExternalProcess(
                [obs_seq_tool_ln], cwd=tmp_dir, log_filename="obs_sequence_tool.log"
            )
        )
        if res.returncode != 0:
            logger.error(f"obs_sequence_tool exited with error code {res.returncode}!")
            logger.error(res.output)
            raise RuntimeError(f"obs_sequence_tool failed with code {res.returncode}")

    return res


def obs_seq_to_nc(
    dart_root: Path,
    obs_seq: Path,
    nc: Path,
    binary_obs_sequence: bool = False,
) -> external.ExternalProcessResult:
    """
    Uses the `obs_seq_to_netcdf` program to convert the given obs_seq file to netcdf format

    Args:
        dart_root: Path to the DART root directory
        obs_seq: Path to obs_seq file
        nc: Path to output netcdf file
        binary_obs_sequence: Whether the obs_seq file is binary or not, defaults to False

    Returns:
        The result of the external process
    """

    # Locate executable
    binary = dart_root / "models" / "wrf" / "work" / "obs_seq_to_netcdf"
    binary = binary.resolve()
    if not binary.exists():
        raise RuntimeError(f"obs_seq_to_netcdf binary not found at {binary}")

    # Link obs_seq_to_netcdf inside a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        binary_ln = tmp_dir / "obs_seq_to_netcdf"
        binary_ln.symlink_to(binary)

        # Create namelist file
        nml = {
            "obs_seq_to_netcdf_nml": {
                "obs_sequence_name": str(obs_seq.resolve()),
                "obs_sequence_list": "",
                "append_to_netcdf": False,
                "lonlim1": 0.0,
                "lonlim2": 360,
                "latlim1": -90,
                "latlim2": 90,
                "verbose": False,
            },
            "obs_sequence_nml": {"write_binary_obs_sequence": binary_obs_sequence},
            "location_nml": {},
            "obs_kind_nml": {},
            "schedule_nml": {},
            "utilities_nml": {
                "TERMLEVEL": 1,
                "module_details": False,
                "logfilename": "obs_sequence_tool.out",
                "nmlfilename": "obs_sequence_tool.nml",
                "write_nml": "file",
            },
        }
        fortran_namelists.write_namelist(nml, tmp_dir / "input.nml")

        # Call obs_seq_to_netcdf
        res = external.run(external.ExternalProcess([binary_ln], cwd=tmp_dir))

        if res.returncode != 0:
            logger.error(f"obs_seq_to_netcdf exited with error code {res.returncode}!")
            logger.error(res.output)
            raise RuntimeError(f"obs_seq_to_netcdf failed with code {res.returncode}")

        # Move output file to the desired location
        obs_seq_out = tmp_dir / "obs_epoch_001.nc"
        utils.copy(obs_seq_out, nc)
        logger.info(f"Converted obs_seq file to netcdf: {nc}")

    return res


def is_obs_seq_empty(file: Path) -> bool:
    """
    Returns true if an obs_seq file is empty. To do this, we read the first couple of
    lines to get the num_obs value. The file is parsed with regex, thus this works only
    with ASCII obs_seq files.

    TODO: Do this in FORTRAN maybe?
    """

    # Read the first couple of lines (2KB)
    with file.open("r") as f:
        content = f.read(2048)

    # Use regex to find num_obs
    match = re.search(r"[^_]num_obs:\s+(\d+)", content)
    if match:
        num_obs = int(match.group(1))
        return num_obs == 0

    # If no match is found, assume the file is empty
    return True


def preprocess_for_wrf(
    dart_root: Path,
    wrfinput: Path,
    obs_seq: Path,
    assimilation_window_dart_time: tuple[int, int],
    boundary_width=0.0,
    boundary_error_factor=0.0,
    boundary_error_width=0.0,
    binary_obs_sequence=False,
):
    """
    Runs a obs_seq file through the `wrf_dart_obs_preprocess` utility that (supported features):
    - Removes all observations outside the domain
    - Inflate errors near the boundaries

    Args:
        dart_root: Path to DART root directory
        wrfinput: Path to the WRF input file
        obs_seq: Path to the obs_seq file. Will be overwritten with the preprocessed version, if successful.
        assimilation_window_dart_time: Tuple with the start and end of the assimilation window in DART time units (days since 1601-01-01, seconds since midnight)
        boundary_width: If > 0, the domain will be reduced by this amount of gridpoints.
        boundary_error_factor: If >0, the errors near the boundary will be increased by this factor.
        boundary_error_width: How near the boundary to increase the errors, in gridpoints. A ramping is applied, so the innermost points are changed by 0 and the outermost by `boundary_error_factor`.
        binary_obs_sequence: Whether the obs_seq file is binary or not, defaults to False
    """

    # Locate the executable
    binary = dart_root / "models" / "wrf" / "work" / "wrf_dart_obs_preprocess"
    binary = binary.resolve()
    if not binary.exists():
        raise RuntimeError(f"wrf_dart_obs_preprocess binary not found at {binary}")

    # Link wrf_dart_obs_preprocess inside a temp directory, alongside all input files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        binary_ln = tmp_dir / "wrf_dart_obs_preprocess"
        binary_ln.symlink_to(binary)

        # Link input files
        wrfinput_ln = tmp_dir / "wrfinput_d01"
        wrfinput_ln.symlink_to(wrfinput)

        obs_seq_ln = tmp_dir / "obs_seq.in"
        obs_seq_ln.symlink_to(obs_seq)

        # Create namelist file
        nml = {
            "wrf_obs_preproc_nml": {
                # I/O
                "file_name_input": "obs_seq.in",
                "file_name_output": "obs_seq.out",
                # Time management, not supported
                "overwrite_obs_time": False,
                # Boundary
                "obs_boundary": boundary_width,
                "increase_bdy_error": boundary_error_factor > 0,
                "maxobsfac": boundary_error_factor,
                "obsdistbdy": boundary_error_width,
                # Surface elevation, not used
                "sfc_elevation_check": False,
                "sfc_elevation_tol": 3000.0,
                "obs_pressure_top": 0.0,
                "obs_height_top": 2.0e10,
                # Radiosonde stuff, not used
                "include_sig_data": True,
                "tc_sonde_radii": -1.0,
                # Superorbing, not used
                "superob_aircraft": False,
                "aircraft_horiz_int": 800.0,
                "aircraft_pres_int": 25000.0,
                "superob_sat_winds": False,
                "sat_wind_horiz_int": 800.0,
                "sat_wind_pres_int": 25000.0,
                # Overwrite QC flags, not used
                "overwrite_ncep_satwnd_qc": False,
                "overwrite_ncep_sfc_qc": False,
            },
            "obs_sequence_nml": {"write_binary_obs_sequence": binary_obs_sequence},
            "location_nml": {},
            "obs_kind_nml": {},
            "schedule_nml": {},
            "model_nml": {},
            "ensemble_manager_nml": {},
            "utilities_nml": {
                "TERMLEVEL": 1,
                "module_details": False,
                "logfilename": "obs_sequence_tool.out",
                "nmlfilename": "obs_sequence_tool.nml",
                "write_nml": "file",
            },
        }
        fortran_namelists.write_namelist(nml, tmp_dir / "input.nml")

        # Call utility, move result back to the original location
        stdin = (
            f"{assimilation_window_dart_time[0]} {assimilation_window_dart_time[1]}\n"
        )
        res = external.run(
            external.ExternalProcess([binary_ln], cwd=tmp_dir, stdin=stdin)
        )

        if res.returncode != 0:
            logger.error(
                f"wrf_dart_obs_preprocess exited with error code {res.returncode}!"
            )
            logger.error(res.output)
            raise external.ExternalProcessFailed(res)

        # If the file is empty, just remove the original
        obs_seq_out = tmp_dir / "obs_seq.out"
        if is_obs_seq_empty(obs_seq_out):
            logger.warning("Preprocessed obs_seq file is empty (all obs removed)!")
            obs_seq.unlink()
            return

        # Move output file to the desired location
        logger.info(f"Removing {obs_seq}")
        obs_seq.unlink()

        logger.info(f"Renaming {obs_seq_out} to {obs_seq}")
        shutil.move(obs_seq_out, obs_seq)
