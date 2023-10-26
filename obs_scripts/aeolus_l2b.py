from pathlib import Path
import re
import datetime as dt
from itertools import chain
import tomli_w

import typer


def create_aeolus_l2b_obsgroup(dir: Path, output: Path):
    """
    Creates the observation group file for the given directory of AEOLUS L2B DBL files
    """

    prefix = "AE_OPER_ALD_U_N_2B_"
    date_regex = re.compile(r"(\d{8}T\d{6})_(\d{8}T\d{6})")

    files = []
    for dbl in chain(dir.glob("*.dbl"), dir.glob("*.DBL")):
        print(f"Processing {dbl}")

        # Sample file name: AE_OPER_ALD_U_N_2B_20210901T003156_20210901T015956_0001.DBL
        name = dbl.name
        if name[: len(prefix)] != prefix:
            print(f"File prefix is not {prefix}!")
            continue

        start, end = date_regex.findall(name)[0]
        start = dt.datetime.strptime(start, "%Y%m%dT%H%M%S").replace(
            tzinfo=dt.timezone.utc
        )
        end = dt.datetime.strptime(end, "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)

        files.append({"start_date": start, "end_date": end, "path": str(dbl.resolve())})

    if not files:
        print("No files found!")
        return

    # Sort files by start date, easier to browse the file this way
    files = sorted(files, key=lambda f: f["start_date"])

    obsgroup = {
        "kind": "AEOLUS_L2B",
        "converter": "/path/to/convert_aeolus_l2b",
        "files": files,
    }
    with open(output, "wb") as f:
        tomli_w.dump(obsgroup, f)
    print(f"Wrote info about {len(files)} files to {output}")


if __name__ == "__main__":
    typer.run(create_aeolus_l2b_obsgroup)
