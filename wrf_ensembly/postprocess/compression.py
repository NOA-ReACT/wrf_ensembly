"""
Compression utilities for netCDF4 output files.

Provides detection of available compression filters and validation
of compression configuration against system capabilities.
"""

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4

if TYPE_CHECKING:
    from wrf_ensembly.config import PostprocessConfig


# Supported compression algorithms
SUPPORTED_COMPRESSIONS = {"zlib", "zstd", "bzip2", "szip", "none"}

# Supported quantization modes
SUPPORTED_QUANTIZE_MODES = {"BitGroom", "BitRound", "GranularBitRound"}


def detect_available_filters() -> dict[str, bool]:
    """
    Detect which compression filters are available in the current netCDF4 installation.

    Returns:
        Dictionary mapping filter names to availability (True/False).
        'zlib' is always available as it's built into netCDF4.
    """
    available = {"zlib": True, "none": True}

    # Create a temporary file to test filter availability
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.nc"

        with netCDF4.Dataset(test_file, "w") as ds:
            available["zstd"] = ds.has_zstd_filter()
            available["bzip2"] = ds.has_bzip2_filter()
            available["szip"] = ds.has_szip_filter()

    return available


def check_significant_digits_support() -> tuple[bool, str]:
    """
    Check if the netcdf-c library version supports significant_digits.

    The significant_digits parameter requires netcdf-c >= 4.9.0.

    Returns:
        Tuple of (is_supported, version_string).
    """
    nc_version_full = netCDF4.__netcdf4libversion__
    # Strip development suffix (e.g., "4.9.4-development" -> "4.9.4")
    nc_version = nc_version_full.split("-")[0]

    try:
        parts = nc_version.split(".")
        major, minor = int(parts[0]), int(parts[1])
        is_supported = (major, minor) >= (4, 9)
    except (ValueError, IndexError):
        # If we can't parse the version, assume not supported
        is_supported = False

    return is_supported, nc_version_full


class CompressionConfigError(Exception):
    """Raised when compression configuration is invalid or unsupported."""

    pass


def validate_compression_config(cfg: "PostprocessConfig") -> None:
    """
    Validate compression settings against available system capabilities.

    Checks that:
    - The requested compression filter is available
    - significant_digits is supported if quantization is enabled
    - quantize_mode is valid

    Args:
        cfg: PostprocessConfig to validate.

    Raises:
        CompressionConfigError: If the configuration is invalid or unsupported.
    """
    # Check compression algorithm
    if cfg.compression not in SUPPORTED_COMPRESSIONS:
        raise CompressionConfigError(
            f"Unknown compression algorithm '{cfg.compression}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_COMPRESSIONS))}"
        )

    available = detect_available_filters()

    if cfg.compression != "none" and not available.get(cfg.compression, False):
        available_list = [k for k, v in available.items() if v and k != "none"]
        raise CompressionConfigError(
            f"Compression filter '{cfg.compression}' is not available. "
            f"Available filters: {', '.join(sorted(available_list))}.\n"
            f"To enable {cfg.compression}, rebuild netCDF4-python with a "
            f"{cfg.compression}-enabled HDF5 library."
        )

    # Check compression level
    if not 0 <= cfg.compression_level <= 9:
        raise CompressionConfigError(
            f"compression_level must be between 0 and 9, got {cfg.compression_level}"
        )

    if cfg.significant_digits != 0:
        is_supported, version = check_significant_digits_support()
        if not is_supported:
            raise CompressionConfigError(
                f"Quantization (significant_digits) requires netcdf-c >= 4.9.0. "
                f"Current version: {version}. "
                f"Set significant_digits to null/None to disable quantization."
            )

        if cfg.significant_digits < 1:
            raise CompressionConfigError(
                f"significant_digits must be >= 1, got {cfg.significant_digits}"
            )

        if cfg.quantize_mode not in SUPPORTED_QUANTIZE_MODES:
            raise CompressionConfigError(
                f"Unknown quantize_mode '{cfg.quantize_mode}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_QUANTIZE_MODES))}"
            )

        if cfg.significant_digits_overrides:
            for pattern, digits in cfg.significant_digits_overrides.items():
                # Check pattern is valid regex
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise CompressionConfigError(
                        f"Invalid regex pattern in significant_digits_overrides: "
                        f"'{pattern}': {e}"
                    )

                # Check digits value
                if digits < 1:
                    raise CompressionConfigError(
                        f"significant_digits_overrides values must be >= 1, "
                        f"got {digits} for pattern '{pattern}'"
                    )
