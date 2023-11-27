import datetime as dt
import tempfile
from pathlib import Path

import tomli
from pydantic import BaseModel

from wrf_ensembly import external, namelist, utils
from wrf_ensembly.config import Config
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import Experiment


class Observation(BaseModel):
    """Represents one observation file"""

    start_date: dt.datetime
    """Timestamp of the first datapoint inside the file"""

    end_date: dt.datetime
    """Timestamp of the last datapoint inside the file"""

    path: Path
    """Path to the physical file"""


class ObservationGroup(BaseModel):
    """
    Represents one group of observations, in which all files share the same format, use
    the same observation operator and are of the same DART kind
    """

    kind: str
    """DART kind for the observation"""

    converter: Path
    """
    Path to converter executable. Must take input and output file as first and second
    arguments
    """

    files: list[Observation]
    """Files in this group"""

    def get_files_in_window(
        self, start: dt.datetime, end: dt.datetime
    ) -> list[Observation]:
        """Returns all files that overlap with the given window"""

        return [
            f
            for f in self.files
            if (f.start_date <= start <= f.end_date)
            or (f.start_date <= end <= f.end_date)
        ]

    def convert_file(self, file: Observation, output_file: Path):
        """Converts a file to DART obs_seq format by running the appropriate converter"""

        res = external.run(
            external.ExternalProcess(
                [self.converter, file.path, output_file],
                cwd=self.converter.parent,
                log_filename=f"converter_{file.path.name}.log",
            )
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Converter {self.converter} failed with code {res.returncode}"
            )
        return res


def read_observation_group(path: Path) -> ObservationGroup:
    """Read an observation group file from the disk"""

    with open(path, "rb") as f:
        obs = tomli.load(f)
    return ObservationGroup(**obs)


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
            "filename_out": "./obs_seq.out",
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
        namelist.write_namelist(nml, namelist_path)

        # Write all input files inside a text file
        filelist_path = tmp_dir / "input_list"
        filelist_path.write_text("\n".join(str(f.resolve()) for f in obs_seq_files))

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

        # Move output file to the desired location
        obs_seq_out = tmp_dir / "obs_seq.out"
        utils.copy(obs_seq_out, output_file)

    return res


def obs_seq_to_nc(
    exp: Experiment,
    obs_seq: Path,
    nc: Path,
    binary_obs_sequence: bool = False,
) -> external.ExternalProcessResult:
    """
    Uses the `obs_seq_to_netcdf` program to convert the given obs_seq file to netcdf format

    Args:
        obs_seq: Path to obs_seq file
        nc: Path to output netcdf file

    Returns:
        The result of the external process
    """

    # Locate executable
    binary = (
        exp.cfg.directories.dart_root / "models" / "wrf" / "work" / "obs_seq_to_netcdf"
    )
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
        namelist.write_namelist(nml, tmp_dir / "input.nml")

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
