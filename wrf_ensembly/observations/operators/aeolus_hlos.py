import numpy as np


def hlos_wind_operator(
    model_fields: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute HLOS wind from earth-relative U/V and Aeolus azimuth angle.

    Follows Aeolus sign convention where positive HLOS is directed away
    from the satellite. Expects earth-relative wind components (i.e.
    already rotated from WRF grid-relative using COSALPHA/SINALPHA).

    Args:
        model_fields: Must contain "wind_east" and "wind_north", each (n_obs,).
        metadata: Must contain "azimuth" in degrees, (n_obs,).

    Returns:
        (n_obs,) HLOS wind values in m/s.
    """

    u_earth = model_fields["wind_east"]
    v_earth = model_fields["wind_north"]
    azimuth_rad = np.deg2rad(metadata["azimuth"])

    return -u_earth * np.sin(azimuth_rad) - v_earth * np.cos(azimuth_rad)
