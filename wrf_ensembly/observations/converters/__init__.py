"""Observation format converters for WRF-Ensembly."""

from .aeolus_l2a import aeolus_l2a as aeolus_l2a_cli
from .aeolus_l2b import aeolus_l2b as aeolus_l2b_cli
from .aeronet import aeronet as aeronet_cli
from .earthcare_ebd import earthcare_atl_ebd as earthcare_ebd_cli
from .modis import modis as modis_cli
from .msg_seviri import msg_seviri as msg_seviri_cli
from .remotap_spexone import remotap_spexone as remotap_spexone_cli
from .viirs import viirs as viirs_cli

__all__ = [
    "aeronet_cli",
    "remotap_spexone_cli",
    "earthcare_ebd_cli",
    "viirs_cli",
    "modis_cli",
    "msg_seviri_cli",
    "aeolus_l2a_cli",
    "aeolus_l2b_cli",
]
