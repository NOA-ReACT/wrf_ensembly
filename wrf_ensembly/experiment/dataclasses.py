import datetime as dt
from dataclasses import dataclass


@dataclass
class RuntimeStatistics:
    """Contains runtime information for one cycle"""

    cycle: int
    """Which cycle are these statistics for"""

    start: dt.datetime
    """Start of model execution"""

    end: dt.datetime
    """End of model execution"""

    duration_s: int
    """Duration of model execution in seconds"""


@dataclass
class MemberStatus:
    """Status of a single ensemble member"""

    i: int
    """The member ID"""

    advanced: bool
    """Has WRF been run for the current cycle?"""

    runtime_statistics: list[RuntimeStatistics]
