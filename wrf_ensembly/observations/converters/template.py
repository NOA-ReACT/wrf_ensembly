"""Template for creating new observation converters.

This file shows the pattern for creating a new converter with both
the conversion function and CLI command in the same file.
"""

from pathlib import Path
from typing import List

import click
import pandas as pd

from wrf_ensembly.observations import io as obs_io


def convert_template(
    input_paths: List[Path], some_option: str = "default"
) -> pd.DataFrame:
    """Convert template format files to WRF-Ensembly Observation format.

    Args:
        input_paths: List of paths to template format files.
        some_option: Example optional parameter.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    # This is where you would implement your actual conversion logic
    # Example structure:

    all_observations = []

    for input_path in input_paths:
        # 1. Read the raw observation file
        # raw_data = pd.read_csv(input_path)  # or whatever format you need

        # 2. Process and transform to WRF-Ensembly format
        # processed_df = transform_data(raw_data, some_option)

        # 3. Ensure all required columns are present and correctly formatted
        # processed_df = format_for_wrf_ensembly(processed_df, input_path)

        # all_observations.append(processed_df)
        pass

    # 4. Combine all files if multiple input files
    # combined_df = pd.concat(all_observations, ignore_index=True)

    # 5. Validate schema and return
    # obs_io.validate_schema(combined_df)
    # return combined_df

    # For template purposes, return empty DataFrame with correct schema
    return pd.DataFrame(columns=obs_io.REQUIRED_COLUMNS)


@click.command()
@click.argument(
    "input_paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--some-option",
    default="default_value",
    help="Example optional parameter.",
)
def template(input_paths: List[Path], output_path: Path, some_option: str):
    """Template converter for demonstration.

    INPUT_PATHS: One or more input files (this example accepts multiple files)
    OUTPUT_PATH: Path where to save the converted observations (will be saved as parquet)
    """

    click.echo(f"Converting {len(input_paths)} files")
    click.echo(f"Input files: {[str(p) for p in input_paths]}")
    click.echo(f"Output path: {output_path}")
    click.echo(f"Option value: {some_option}")

    # Convert the data
    converted_df = convert_template(list(input_paths), some_option=some_option)

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    click.echo(f"Successfully converted {len(converted_df)} observations")
    click.echo(f"Saved to: {output_path}")


# Note: To add this converter to the CLI, you would:
# 1. Import this command in observation/cli.py:
#    from wrf_ensembly.observation.converters.template import template
# 2. Add it to the CLI:
#    cli.add_command(template)
# 3. Update the __init__.py files to include the new module
