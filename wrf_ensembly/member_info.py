from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
import tomli
import tomli_w

from wrf_ensembly.utils import filter_none_from_dict


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

    def get_current_cycle(self) -> CycleSection:
        """
        Returns the current cycle of the member.

        Returns:
            CycleSection object
        """
        if self.member.current_cycle not in self.cycle:
            raise ValueError(
                f"Member {self.member.i} has no cycle {self.member.current_cycle}, but current_cycle is set to it."
            )

        return self.cycle[self.member.current_cycle]

    def set_current_cycle(self, cycle: CycleSection):
        """
        Sets the current cycle of the member.

        Args:
            cycle: CycleSection object
        """

        self.cycle[self.member.current_cycle] = cycle


def read_member_info(experiment_path: Path, member_id: int) -> MemberInfo:
    """
    Reads the member info file for a given member.

    Args:
        experiment_path: Path to the experiment directory
        member_id: ID of the member

    Returns:
        MemberInfo object
    """

    toml_path = experiment_path / "work" / "ensemble" / f"member_{member_id:02d}.toml"
    return read_member_info_toml(toml_path)


def read_all_member_info(experiment_path: Path) -> dict[int, MemberInfo]:
    """
    Reads all member info files for a given experiment.

    Args:
        experiment_path: Path to the experiment directory

    Returns:
        Dictionary of member info objects ({id: MemberInfo})
    """

    member_info = {}
    for member_path in (experiment_path / "work" / "ensemble").glob("member_*.toml"):
        member_id = int(member_path.stem.split("_")[1])
        member_info[member_id] = read_member_info_toml(member_path)
    return member_info


def read_member_info_toml(path: Path) -> MemberInfo:
    """
    Reads a TOML member info file and returns a MemberInfo object.

    Args:
        path: Path to the TOML member info file
    """
    with open(path, "rb") as f:
        minfo = tomli.load(f)

    minfo = MemberInfo(**minfo)

    return minfo


def write_member_info(experiment_path: Path, minfo: MemberInfo) -> Path:
    """
    Writes a MemberInfo object to a TOML file.

    Args:
        experiment_path: Path to the experiment directory
        minfo: MemberInfo object to write

    Returns:
        Path to the member info file
    """

    toml_path = (
        experiment_path / "work" / "ensemble" / f"member_{minfo.member.i:02d}.toml"
    )
    write_member_info_toml(toml_path, minfo)
    return toml_path


def write_member_info_toml(path: Path, minfo: MemberInfo):
    """
    Writes a MemberInfo object to a TOML file.

    Args:
        path: Path to the TOML configuration file
        minfo: MemberInfo object to write
    """

    cycle = {str(k): filter_none_from_dict(v.dict()) for k, v in minfo.cycle.items()}

    with open(path, "wb") as f:
        tomli_w.dump(minfo.dict() | {"cycle": cycle}, f)


def ensure_same_cycle(minfos: dict[int, MemberInfo]):
    """
    Ensures that all members have the same current cycle. Raises a ValueError otherwise.

    Args:
        minfos: Dictionary of MemberInfo objects
    """

    current_cycle = None
    for minfo in minfos.values():
        if current_cycle is None:
            current_cycle = minfo.member.current_cycle
        elif current_cycle != minfo.member.current_cycle:
            raise ValueError(
                f"Member {minfo.member.i} has a different current cycle than member 0"
            )


def ensure_current_cycle_state(minfos: dict[int, MemberInfo], state: dict[str, any]):
    """
    Ensures that all member infos are at the given state.
    """

    curr_cycles = {i: minfo.member.current_cycle for i, minfo in minfos.items()}
    if len(set(curr_cycles.values())) != 1:
        raise ValueError("Not all members are at the same current cycle", curr_cycles)

    c = curr_cycles[0]
    for minfo in minfos.values():
        cycle_info = minfo.cycle[c].dict()
        for k, v in state.items():
            if k not in cycle_info or cycle_info[k] != v:
                raise ValueError(
                    f"Member {minfo.member.i} has a different {k} than expected"
                )
