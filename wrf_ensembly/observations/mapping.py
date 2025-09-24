"""
This file contains the dictionaries to map between `quantity`, model variable names and DART observation types.
"""

QUANTITY_TO_DART_OBS_TYPE = {
    "AOD_500nm": "AIRSENSE_AOD",
    "AOD_550nm": "AIRSENSE_AOD",
}

QUANTITY_TO_WRF_VAR = {
    "AOD_500nm": "AOD_500",
    "AOD_550nm": "AOD_550",
}
