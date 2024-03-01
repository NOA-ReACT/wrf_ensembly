"""
Model for the member_info.toml file, which stores information about the state of each
ensemble member (i.e., which cycle are they in, what has been run, etc.).

Mainly used in `experiment.py` to validate the toml file after reading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel


@dataclass
class MemberSection:
    i: int
    """Index of the member"""

    current_cycle: int
    """Current cycle of the member, i.e. which cycle should now run"""


@dataclass
class CycleSection:
    runtime: Optional[datetime]
    """When the cycle was processed"""

    walltime_s: Optional[int]
    """Walltime in seconds"""

    advanced: bool
    """Whether the cycle was advanced or not"""

    filter: bool
    """Whether the filter was run or not"""

    analysis: bool
    """Whether the posterior was postprocessed or not"""

    def to_dict(self) -> dict[str, Union[datetime, int, bool, None]]:
        return {
            "runtime": self.runtime,
            "walltime_s": self.walltime_s,
            "advanced": self.advanced,
            "filter": self.filter,
            "analysis": self.analysis,
        }


class MemberInfo(BaseModel):
    metadata: dict[str, str] = field(default_factory=dict)

    member: MemberSection

    cycle: dict[int, CycleSection] = field(default_factory=dict)
