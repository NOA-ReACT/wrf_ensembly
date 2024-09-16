from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter


def set_boundaries(arr: np.ndarray, boundary_size: int, value: float) -> np.ndarray:
    """Sets the boundaries of an array to a given value."""

    slices = [slice(None)] * arr.ndim
    for dim in range(arr.ndim):
        if arr.shape[dim] < 2 * boundary_size:
            continue

        slices[dim] = slice(None, boundary_size)
        arr[tuple(slices)] = value
        slices[dim] = slice(-boundary_size, None)
        arr[tuple(slices)] = value
        slices[dim] = slice(None)  # Reset slice for next dimension
    return arr


def generate_pertubation_field(
    shape: Tuple[int, ...], mean: float, sd: float, rounds=10, boundary=0
):
    """
    Generates a random pseudo-spatially-collerated field. Based on the technique
    described in Tsikerdekis et. al. 2021 (https://doi.org/10.5194/acp-21-2637-2021).

    The field is generated by creating an array of random numbers (sampled from N(1, 10))
    and applying consequent rounds of gaussian smoothing. The field is then normalized
    to have a mean of `mean` and a standard deviation of `sd`.

    Args:
        shape: The shape of the random field to create
        mean: Desired mean value of the field, after normalization
        sd: Desired standard deviation of the field, after normalization
        rounds: Number of rounds of gaussian smoothing to apply

    Returns:
        The generated field as a numpy array.
    """
    x = np.random.normal(1, 10, shape)

    if boundary > 0:
        x = set_boundaries(x, boundary, 1)

    for _ in range(rounds):
        x = uniform_filter(x, size=3)
    x = (x - x.mean()) / x.std()
    x = x * sd + mean
    return x
