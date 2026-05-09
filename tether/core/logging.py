"""Structured logging setup for Tether using structlog.

Usage:
    from tether.core.logging import get_logger, bound_context

    log = get_logger(__name__)
    log.info("step_recorded", step_id=str(step_id), latency_ms=42.1)

    with bound_context(run_id="abc-123"):
        log.info("inside_run")  # automatically includes run_id
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog

_configured = False


def _configure() -> None:
    """Configure structlog once at import time (idempotent)."""
    global _configured
    if _configured:
        return

    is_dev = os.getenv("TETHER_ENV", "production").lower() in ("dev", "development", "local")

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_dev:
        processors: list[Any] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_get_log_level()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def _get_log_level() -> int:
    """Return the integer log level from the TETHER_LOG_LEVEL env var."""
    import logging

    level_name = os.getenv("TETHER_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_logger(name: str) -> Any:
    """Return a structlog bound logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured structlog BoundLogger instance with ``logger`` pre-bound.
    """
    _configure()
    # Bind the name directly since PrintLoggerFactory doesn't expose .name
    # the way stdlib logging does.
    return structlog.get_logger().bind(logger=name)


@contextmanager
def bound_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager that binds key-value pairs to all log events within the block.

    This is the correct way to attach run_id or step_id to a group of log
    statements without threading them through every function call.

    Args:
        **kwargs: Arbitrary key-value pairs to bind (e.g. run_id="...", step_id="...").

    Example:
        with bound_context(run_id=str(run.id)):
            log.info("starting")   # automatically includes run_id
            do_work()
            log.info("finished")   # still includes run_id
    """
    _configure()
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(*kwargs.keys())
