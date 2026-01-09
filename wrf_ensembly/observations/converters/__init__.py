"""Observation format converters for WRF-Ensembly."""

from .aeronet import aeronet as aeronet_cli
from .remotap_spexone import remotap_spexone as remotap_spexone_cli
from .earthcare_ebd import earthcare_atl_ebd as earthcare_ebd_cli
from .viirs import viirs as viirs_cli
from .modis import modis as modis_cli

# Try to import AEOLUS converters (requires optional 'coda' dependency)
# We need to check if the module can be imported without triggering the coda import
aeolus_l2b_cli = None
aeolus_l2a_cli = None
HAS_AEOLUS_CONVERTERS = False

try:
    # This will only succeed if coda can be loaded
    from .aeolus_l2b import HAS_CODA as has_coda_l2b
    from .aeolus_l2a import HAS_CODA as has_coda_l2a

    if has_coda_l2b and has_coda_l2a:
        from .aeolus_l2b import aeolus_l2b as aeolus_l2b_cli
        from .aeolus_l2a import aeolus_l2a as aeolus_l2a_cli

        HAS_AEOLUS_CONVERTERS = True
except (ImportError, OSError):
    # Either the module can't be imported or coda can't be loaded
    pass

__all__ = [
    "aeronet_cli",
    "remotap_spexone_cli",
    "earthcare_ebd_cli",
    "viirs_cli",
    "modis_cli",
    "aeolus_l2b_cli",
    "aeolus_l2a_cli",
    "HAS_AEOLUS_CONVERTERS",
]
