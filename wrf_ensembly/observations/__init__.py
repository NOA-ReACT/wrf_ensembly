"""
This module contains some functions & commands for handling WRF-Ensembly observation
files, independenly from an experiment

Structure:
- io.py: Core I/O functions and schema validation, use this to read/write files
- cli.py: CLI entry point for the `wrf-ensembly-obs` command
- operations.py: CLI commands for interacting with observation files
- utils.py: Pure functions for handling observation dataframes
- converters/: Individual converter modules for instruments, each containing both conversion functions and CLI commands
"""

from . import converters, dart, io, superorbing, plotting
from .utils import project_locations_to_wrf

__all__ = [
    "io",
    "converters",
    "project_locations_to_wrf",
    "dart",
    "superorbing",
    "plotting",
]
