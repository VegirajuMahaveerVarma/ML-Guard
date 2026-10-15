"""Exception types raised by MLGuard."""


class MLGuardError(Exception):
    """Base exception for expected MLGuard errors."""


class InputValidationError(MLGuardError, ValueError):
    """Raised when model inputs or configuration are invalid."""


class ModelExecutionError(MLGuardError, RuntimeError):
    """Raised when a wrapped model cannot produce predictions."""
