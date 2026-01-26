"""
Postprocessing module for ensemble WRF output.

This module provides streaming postprocessing capabilities for ensemble
forecasts, including:
- Streaming NetCDF writer for incremental time-series output
- Streaming pipeline for memory-efficient ensemble statistics computation
- Compression configuration validation and filter detection
"""

from wrf_ensembly.postprocess.compression import (
    CompressionConfigError,
    check_significant_digits_support,
    detect_available_filters,
    validate_compression_config,
)
from wrf_ensembly.postprocess.streaming_pipeline import (
    process_cycle_single_member,
    process_cycle_streaming,
    process_members_for_timestep,
)
from wrf_ensembly.postprocess.streaming_writer import StreamingNetCDFWriter

__all__ = [
    "StreamingNetCDFWriter",
    "process_cycle_streaming",
    "process_cycle_single_member",
    "process_members_for_timestep",
    "CompressionConfigError",
    "detect_available_filters",
    "check_significant_digits_support",
    "validate_compression_config",
]
