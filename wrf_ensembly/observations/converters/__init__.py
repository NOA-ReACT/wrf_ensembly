"""Observation format converters for WRF-Ensembly."""

from .aeronet import aeronet as aeronet_cli
from .remotap_spexone import remotap_spexone as remotap_spexone_cli
from .earthcare_ebd import earthcare_atl_ebd as earthcare_ebd_cli
from .aeolus_l2b import aeolus_l2b as aeolus_l2b_cli
from .aeolus_l2a import aeolus_l2a as aeolus_l2a_cli

__all__ = [
    "aeronet_cli",
    "remotap_spexone_cli",
    "earthcare_ebd_cli",
    "aeolus_l2b_cli",
    "aeolus_l2a_cli",
]
