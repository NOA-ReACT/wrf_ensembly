import re
from pathlib import Path
from typing import Union


def convert_to_dtype(
    x: list[str] | str,
) -> list[int | float | str | bool | list] | int | float | str | bool:
    """
    Convert a namelist value to the appropriate data type.
    Can handle integers and floats, and lists of either.
    If the value is a string, it is returned as-is.
    """
    if isinstance(x, list):
        return [convert_to_dtype(v) for v in x]
    x = x.strip()
    if x == ".true.":
        return True
    if x == ".false.":
        return False
    if re.match(r"^[Ff]alse$", x):
        return False
    if re.match(r"^-?[0-9]+$", x):
        return int(x)
    if re.match(r"^-?[0-9]+\.[0-9]*e?[-+]?[0-9]*$", x):
        return float(x)
    if re.match(r"^'.*'$", x):
        return x[1:-1]
    return x


def parse(buf: str) -> dict[str, Union[str, dict, list, str, int, float]]:
    """
    Parse a namelist into a dictionary.

    Args:
        buf: The namelist to parse, in string format.
    """
    groups = {}
    group = None
    for line in buf.splitlines():
        line = line.strip()
        if line.startswith("&"):
            group = line[1:].strip()
            groups[group] = {}
        elif line.startswith("/"):
            group = None
        elif "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.endswith(","):
                value = value[:-1]
            if "," in value:
                value = [v.strip() for v in value.split(",")]
            groups[group][key] = convert_to_dtype(value)
    return groups


def read(path: str | Path):
    """
    Read a namelist from a file.

    Args:
        path: The path to read the namelist from.
    """
    with open(path, "r") as f:
        return parse(f.read())


def convert_to_string(
    x: Union[str, bool, int, float, list[Union[str, bool, int, float]]],
) -> str:
    """
    Convert a namelist value to a string, properly formatting it.
    """

    if isinstance(x, list):
        return ", ".join([convert_to_string(v) for v in x])
    if isinstance(x, bool):
        return ".true." if x else ".false."
    if isinstance(x, int) or isinstance(x, float):
        return str(x)
    return f"'{x}'"


def write_namelist(namelist: dict, path: str | Path):
    """
    Write a namelist to a file.

    Args:
        namelist: The namelist to write.
        path: The path to write the namelist to.
    """
    with open(path, "w") as f:
        for group, values in namelist.items():
            print(f"&{group}", file=f)
            for key, value in values.items():
                if value is None:
                    continue
                value = convert_to_string(value)
                print(f"    {key} = {value},", file=f)
            print("/", file=f)
