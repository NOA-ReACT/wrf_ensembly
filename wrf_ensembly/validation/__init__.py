"""Validation module for analyzing experiment results against observations."""

from wrf_ensembly.validation.model_interpolation import ModelInterpolation
from wrf_ensembly.validation.first_departures import FirstDeparturesAnalysis

__all__ = ["ModelInterpolation", "FirstDeparturesAnalysis"]
