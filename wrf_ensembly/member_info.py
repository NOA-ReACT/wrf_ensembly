"""
Model for the member_info.toml file, which stores information about the state of each
ensemble member (i.e., which cycle are they in, what has been run, etc.).

Mainly used in `experiment.py` to validate the toml file after reading.
"""

from datetime import datetime

from pydantic import BaseModel


class MemberSection(BaseModel):
    i: int
    """Index of the member"""

    current_cycle: int
    """Current cycle of the member, i.e. which cycle should now run"""


class CycleSection(BaseModel):
    runtime: datetime | None
    """When the cycle was processed"""

    walltime_s: int | None
    """Walltime in seconds"""

    advanced: bool
    """Whether the cycle was advanced or not"""

    filter: bool
    """Whether the filter was run or not"""

    analysis: bool
    """Whether the posterior was postprocessed or not"""


class MemberInfo(BaseModel):
    metadata: dict[str, str] = {}

    member: MemberSection

    cycle: dict[int, CycleSection] = {}
