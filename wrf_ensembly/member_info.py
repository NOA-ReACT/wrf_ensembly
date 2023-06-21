from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
import tomli
import tomli_w


class MemberSection(BaseModel):
    i: int
    """Index of the member"""

    current_cycle: int
    """Current cycle of the member, i.e. which cycle should now run"""


class CycleSection(BaseModel):
    runtime: datetime
    """When the cycle was processed"""

    walltime_s: int
    """Walltime in seconds"""


class MemberInfo(BaseModel):
    metadata: dict[str, str] = {}

    member: MemberSection

    cycle: dict[int, CycleSection] = {}


def read_member_info(path: Path) -> MemberInfo:
    """
    Reads a TOML member info file and returns a MemberInfo object.

    Args:
        path: Path to the TOML member info file
    """
    with open(path, "rb") as f:
        minfo = tomli.load(f)

    minfo = MemberInfo(**minfo)

    return minfo


def write_member_info(path: Path, minfo: MemberInfo):
    """
    Writes a MemberInfo object to a TOML file.

    Args:
        path: Path to the TOML configuration file
        minfo: MemberInfo object to write
    """
    with open(path, "wb") as f:
        tomli_w.dump(minfo.dict(), f)
