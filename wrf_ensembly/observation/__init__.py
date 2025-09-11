"""
This module contains some functions & commands for handling WRF-Ensembly observation
files, independenly from an experiment

Structure:
- io.py: Core I/O functions and schema validation, use this to read/write files
- cli.py: CLI entry point for the `wrf-ensembly-obs` command
- converters/: Individual converter modules for instruments, each containing both conversion functions and CLI commands
"""

from . import converters, io

__all__ = ["io", "converters"]
