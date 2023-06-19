from datetime import datetime, timedelta


def datetime_to_namelist_items(dt: datetime, prefix: str) -> dict[str, int]:
    """
    Converts a datetime to a set of namelist items, as required by WRF.

    Args:
        dt: The datetime to convert
        prefix: Which prefix to use for the namelist items (e.g. "start" or "end")

    Returns:
        The converted namelist items in a dictionary
    """

    return {
        f"{prefix}_year": dt.year,
        f"{prefix}_month": dt.month,
        f"{prefix}_day": dt.day,
        f"{prefix}_hour": dt.hour,
        f"{prefix}_minute": dt.minute,
        f"{prefix}_second": dt.second,
    }


def timedelta_to_namelist_items(td: timedelta, prefix: str = "run") -> dict[str, int]:
    """
    Converts a timedelta to a set of namelist items, as required by WRF
    (for example, the `run_*` items).

    Args:
        td: The timedelta to convert
        prefix: Prefix of items, defaults to "run".

    Returns:
        The converted namelist items in a dictionary
    """

    return {
        f"{prefix}_days": td.days,
        f"{prefix}_hours": td.seconds // 3600,
        f"{prefix}_minutes": (td.seconds // 60) % 60,
        f"{prefix}_seconds": td.seconds % 60,
    }
