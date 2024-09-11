import datetime as dt
import re
from itertools import chain
from pathlib import Path

import click
import tomli_w


@click.command()
@click.argument("dir_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def create_aeolus_l2b_obsgroup(dir_path: Path, output_path: Path):
    """
    Creates the observation group file for the given directory of AEOLUS L2B DBL files
    """

    prefix = "AE_OPER_ALD_U_N_2B_"
    date_regex = re.compile(r"(\d{8}T\d{6})_(\d{8}T\d{6})")

    files = []
    for dbl in chain(dir_path.glob("*.dbl"), dir_path.glob("*.DBL")):
        click.echo(f"Processing {dbl}")

        # Sample file name: AE_OPER_ALD_U_N_2B_20210901T003156_20210901T015956_0001.DBL
        name = dbl.name
        if name[: len(prefix)] != prefix:
            click.echo(f"File prefix is not {prefix}!")
            continue

        start, end = date_regex.findall(name)[0]
        start = dt.datetime.strptime(start, "%Y%m%dT%H%M%S").replace(
            tzinfo=dt.timezone.utc
        )
        end = dt.datetime.strptime(end, "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)

        files.append({"start_date": start, "end_date": end, "path": str(dbl.resolve())})

    if not files:
        click.echo("No files found!")
        return

    # Sort files by start date, easier to browse the file this way
    files = sorted(files, key=lambda f: f["start_date"])

    obsgroup = {
        "kind": "AEOLUS_L2B_HLOS",
        "converter": "/path/to/convert_aeolus_l2b",
        "files": files,
    }
    with open(output_path, "wb") as f:
        tomli_w.dump(obsgroup, f)
    click.echo(f"Wrote info about {len(files)} files to {output_path}")


if __name__ == "__main__":
    create_aeolus_l2b_obsgroup()
