"""Custom exception hierarchy for Tether.

All Tether exceptions inherit from TetherError so callers can catch
the entire family with a single except clause when desired.
"""

from __future__ import annotations


class TetherError(Exception):
    """Base class for all Tether errors."""


class StorageError(TetherError):
    """Raised when a SQLite read or write operation fails.

    Args:
        message: Human-readable description of what went wrong and how to fix it.
        original_error: The underlying exception that caused this error, if any.
    """

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class CaptureError(TetherError):
    """Raised when recording an LLM call fails.

    Args:
        message: Human-readable description of what went wrong.
        original_error: The underlying exception that caused this error, if any.
    """

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class RecoveryError(TetherError):
    """Raised when all recovery strategies for a failed LLM call are exhausted.

    Args:
        message: Human-readable description of what was attempted.
        original_error: The underlying exception that caused this error, if any.
    """

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class ConfigurationError(TetherError):
    """Raised when Tether is misconfigured.

    Args:
        message: Human-readable description of the misconfiguration and how to fix it.
        original_error: The underlying exception that caused this error, if any.
    """

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class RateLimitError(CaptureError):
    """Raised when the provider returns a 429 Too Many Requests response.

    Tether will automatically retry with exponential backoff when recovery
    is enabled. If you see this error, either the retry budget was exhausted
    or recovery is disabled for this run.

    Args:
        message: Human-readable description including provider and retry details.
        original_error: The 429 HTTP error from the provider SDK.
    """


class ProviderError(CaptureError):
    """Raised when the provider returns a 5xx server error.

    Args:
        message: Human-readable description including provider and status code.
        original_error: The HTTP error from the provider SDK.
    """


class BudgetExceededError(TetherError):
    """Raised when a run's estimated cost exceeds the configured monthly_budget.

    Set a budget via TetheredOpenAI(budget_usd=10.0) to enable enforcement.

    Args:
        message: Human-readable description including the budget and actual spend.
        original_error: Always None for this error type.
    """

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error
