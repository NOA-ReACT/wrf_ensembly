"""
Streaming NetCDF writer for incremental time-series output.

Provides a class that manages writing to NetCDF files with an unlimited
time dimension, appending one timestep at a time.
"""

import re
from pathlib import Path

import netCDF4
import numpy as np

from wrf_ensembly.statistics import COORDINATE_VARIABLES, NetCDFFile, create_file


class StreamingNetCDFWriter:
    """
    Manages incremental writes to a NetCDF file with unlimited time dimension.

    This writer is designed for streaming postprocessing workflows where data
    is processed one timestep at a time and appended to the output file.

    Time coordinates are automatically converted from datetime64 to integer
    minutes since the reference time specified in the template's units attribute.

    Example:
        template = get_structure_from_xarray(processed_ds, reference_time)
        writer = StreamingNetCDFWriter(output_path, template)

        for timestep_data, time_value in data_stream:
            writer.append_timestep(timestep_data, time_value)

        writer.close()
    """

    def __init__(
        self,
        path: Path,
        template: NetCDFFile,
        compression: str = "zlib",
        complevel: int = 4,
        shuffle: bool = True,
        significant_digits: int | None = None,
        significant_digits_overrides: dict[str, int] | None = None,
        quantize_mode: str = "GranularBitRound",
    ):
        """
        Create output file from template structure.

        Args:
            path: Path where the output file will be created.
            template: NetCDFFile structure to use as template.
            compression: Compression algorithm ('zlib', 'zstd', 'bzip2', 'szip', or 'none').
            complevel: Compression level (0-9).
            shuffle: Whether to apply shuffle filter before compression.
            significant_digits: Default number of significant digits for quantization.
                Set to None to disable quantization.
            significant_digits_overrides: Dict mapping regex patterns to significant digits
                for per-variable overrides.
            quantize_mode: Quantization algorithm ('BitGroom', 'BitRound', 'GranularBitRound').
        """
        self.path = path
        self.template = template
        self.time_index = 0
        self.reference_time: np.datetime64 | None = None

        # Extract reference time from template's time variable units attribute
        if "t" in template.variables:
            units = template.variables["t"].attributes.get("units", "")
            # Parse "minutes since YYYY-MM-DD HH:MM:SS" format
            match = re.match(r"minutes since (.+)", units)
            if match:
                ref_time_str = match.group(1)
                self.reference_time = np.datetime64(ref_time_str.replace(" ", "T"))

        # Remove existing file if present
        path.unlink(missing_ok=True)

        # Create the file using the existing create_file function
        self.ds = create_file(
            path,
            template,
            compression=compression,
            complevel=complevel,
            shuffle=shuffle,
            significant_digits=significant_digits,
            significant_digits_overrides=significant_digits_overrides,
            quantize_mode=quantize_mode,
        )

    def append_timestep(
        self,
        data: dict[str, np.ndarray],
        time_coord: np.ndarray,
    ) -> None:
        """
        Append one timestep of data to the file.

        Args:
            data: Dictionary mapping variable names to arrays.
                  Arrays should NOT include the time dimension - they will
                  be written at the current time index.
            time_coord: Time coordinate value for this timestep. Can be datetime64
                       (will be converted to integer minutes) or numeric.
        """
        # Write time coordinate
        if "t" in self.ds.variables:
            # Convert datetime64 to integer minutes since reference time
            if np.issubdtype(np.asarray(time_coord).dtype, np.datetime64):
                if self.reference_time is None:
                    raise ValueError(
                        "Cannot convert datetime64 time coordinate: no reference time set"
                    )
                # Calculate minutes since reference time
                time_val = np.asarray(time_coord).flatten()[0]
                delta = (time_val - self.reference_time) / np.timedelta64(1, "m")
                time_coord = int(delta)

            self.ds.variables["t"][self.time_index] = time_coord

        # Write each variable
        for var_name, values in data.items():
            if var_name not in self.ds.variables:
                continue
            if var_name in COORDINATE_VARIABLES:
                continue

            var = self.ds.variables[var_name]

            # Check if variable has time dimension
            if "t" in var.dimensions:
                var[self.time_index, ...] = values
            else:
                # Non-time-varying variable, only write once
                if self.time_index == 0:
                    var[:] = values

        self.time_index += 1
        self.ds.sync()  # Flush to disk

    def close(self) -> None:
        """Close the NetCDF file."""
        self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
