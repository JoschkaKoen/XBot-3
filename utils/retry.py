"""
Retry utilities — decorator and inline helper with configurable delay.

Usage:
    # As decorator:
    @with_retry(max_attempts=3, base_delay=0.1, backoff=1.0, label="my_call")
    def my_fn(...): ...

    # Inline (no decorator needed):
    result = retry_call(some_fn, arg1, arg2, max_attempts=3, label="my_call")

backoff=1.0 → fixed delay between retries (default).
backoff=2.0 → exponential back-off (delay doubles after each failure).
"""

import time
import functools
import logging
from typing import Callable, Type, Tuple

logger = logging.getLogger("xbot.retry")


def with_retry(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    backoff: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = ""
) -> Callable:
    """
    Decorator: retry with configurable delay strategy.

    Args:
        max_attempts: total number of tries (including the first)
        base_delay:   wait in seconds between attempts
        backoff:      multiplier applied to delay after each failure.
                      1.0 = fixed delay (default), 2.0 = exponential back-off
        exceptions:   tuple of exception types to catch and retry
        label:        human-readable name used in log messages
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s", name, max_attempts, exc
                        )
                        raise
                    exc_str = str(exc)
                    if len(exc_str) > 120:
                        exc_str = exc_str[:120] + " …"
                    logger.warning(
                        "%s attempt %d/%d failed (%s). Retrying in %.1fs …",
                        name, attempt, max_attempts, exc_str, delay
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


def retry_call(
    fn: Callable,
    *args,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    backoff: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    label: str = "",
    **kwargs,
):
    """
    Inline retry helper (no decorator needed).

    Example:
        result = retry_call(some_fn, arg1, arg2, max_attempts=4, label="DeepL")
    """
    decorated = with_retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        backoff=backoff,
        exceptions=exceptions,
        label=label or fn.__name__,
    )(fn)
    return decorated(*args, **kwargs)
