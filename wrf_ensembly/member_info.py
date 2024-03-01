"""
Model for the member_info.toml file, which stores information about the state of each
ensemble member (i.e., which cycle are they in, what has been run, etc.).

Mainly used in `experiment.py` to validate the toml file after reading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

from mashumaro.mixins.toml import DataClassTOMLMixin


@dataclass
class MemberSection:
    i: int
    """Index of the member"""

    current_cycle: int
    """Current cycle of the member, i.e. which cycle should now run"""


@dataclass
class CycleSection:
    advanced: bool
    """Whether the cycle was advanced or not"""

    filter: bool
    """Whether the filter was run or not"""

    analysis: bool
    """Whether the posterior was postprocessed or not"""

    runtime: Optional[datetime] = None
    """When the cycle was processed"""

    walltime_s: Optional[int] = None
    """Walltime in seconds"""

    def to_dict(self) -> dict[str, Union[datetime, int, bool, None]]:
        return {
            "runtime": self.runtime,
            "walltime_s": self.walltime_s,
            "advanced": self.advanced,
            "filter": self.filter,
            "analysis": self.analysis,
        }


@dataclass
class MemberInfo(DataClassTOMLMixin):
    member: MemberSection

    metadata: dict[str, str] = field(default_factory=dict)

    cycle: dict[str, CycleSection] = field(default_factory=dict)
