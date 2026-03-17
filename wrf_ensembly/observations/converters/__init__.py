"""Observation format converters for WRF-Ensembly."""

from .aeronet import aeronet as aeronet_cli
from .earthcare_ebd import earthcare_atl_ebd as earthcare_ebd_cli
from .modis import modis as modis_cli
from .remotap_spexone import remotap_spexone as remotap_spexone_cli
from .viirs import viirs as viirs_cli

# Try to import AEOLUS converters (requires optional 'coda' dependency)
HAS_AEOLUS_CONVERTERS = False
try:
    from .aeolus_l2a import aeolus_l2a as aeolus_l2a_cli
    from .aeolus_l2b import aeolus_l2b as aeolus_l2b_cli

    HAS_AEOLUS_CONVERTERS = True
except (ImportError, OSError, TypeError):
    pass

__all__ = [
    "aeronet_cli",
    "remotap_spexone_cli",
    "earthcare_ebd_cli",
    "viirs_cli",
    "modis_cli",
    "HAS_AEOLUS_CONVERTERS",
]

if HAS_AEOLUS_CONVERTERS:
    __all__ += ["aeolus_l2a_cli", "aeolus_l2b_cli"]
