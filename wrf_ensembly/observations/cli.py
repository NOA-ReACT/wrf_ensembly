"""Main CLI for WRF-Ensembly observation converters.

This module provides a CLI for converting various observation data formats
to the standardized WRF-Ensembly observation format.

Adding a new converter:
1. Create a converter function in wrf_ensembly/observation/converters/your_format.py
   - The function should take input parameters and return a pandas DataFrame
   - The DataFrame must conform to the WRF-Ensembly observation schema
   - Use validate_schema() from wrf_ensembly.observation.io to check compliance

2. Add a CLI command in the same file using click decorators:
   - Use click decorators to define arguments and options
   - Call your converter function and save results using obs_io.write_obs()
   - See the aeronet.py file for an example structure

3. Import and add your command to this file:
   - from wrf_ensembly.observation.converters.your_format import your_format_cli
   - cli.add_command(your_format_cli)

4. Update the __init__.py files to include your new modules in __all__

The resulting CLI will be: wrf-ensembly-obs-convert your_format [args...]
"""

import click

from wrf_ensembly.observations.converters import aeronet_cli, remotap_spexone_cli
from wrf_ensembly.observations.operations import (
    dump_info,
    filter_obs,
    join_files,
    to_obs_seq,
)


@click.group()
def cli():
    pass


@cli.group()
def convert_group():
    """Convert raw observation files to WRF-Ensembly observation format.

    This tool provides converters for various observation data formats.
    Each converter takes raw observation files and converts them to the
    standardized WRF-Ensembly observation format (parquet files).
    """
    pass


convert_group.add_command(aeronet_cli)
convert_group.add_command(remotap_spexone_cli)


@cli.group()
def operations_group():
    """Perform operations on WRF-Ensembly observation files.

    This tool provides various operations for WRF-Ensembly observation files,
    such as joining multiple files, filtering observations, and dumping file info.
    """
    pass


operations_group.add_command(join_files)
operations_group.add_command(dump_info)
operations_group.add_command(filter_obs)
operations_group.add_command(to_obs_seq)

if __name__ == "__main__":
    cli()
