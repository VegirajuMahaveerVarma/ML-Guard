"""Validation helpers shared by MLGuard checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlguard.exceptions import InputValidationError


def as_array(values: Any, *, name: str) -> np.ndarray:
    """Convert values to a NumPy array and reject empty inputs."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} cannot be converted to an array.") from exc
    if array.ndim == 0 or array.shape[0] == 0:
        raise InputValidationError(f"{name} must contain at least one sample.")
    return array


def validate_same_sample_count(reference: np.ndarray, candidate: np.ndarray) -> None:
    """Ensure two arrays contain the same number of samples."""
    if reference.shape[0] != candidate.shape[0]:
        raise InputValidationError(
            "The arrays must contain the same number of samples; "
            f"got {reference.shape[0]} and {candidate.shape[0]}."
        )


def require_numeric(array: np.ndarray, *, name: str) -> np.ndarray:
    """Convert an input array to a numeric dtype, raising a clear error on failure."""
    try:
        result = array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must contain numeric values.") from exc
    if not np.isfinite(result).all():
        raise InputValidationError(f"{name} must contain only finite numeric values.")
    return result
