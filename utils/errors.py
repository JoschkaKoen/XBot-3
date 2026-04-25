"""
Bot-wide exception types.

FatalProviderError — raised when an external API returns a non-retryable
HTTP status (401 Unauthorised, 402 Payment Required, 403 Forbidden).

Catching this in the main loop stops the bot immediately so no further
API calls are made after billing/auth failures.
"""


class FatalProviderError(RuntimeError):
    """
    Non-retryable provider failure (billing expired, invalid key, forbidden).

    Raise this instead of letting requests.raise_for_status() bubble up when
    the HTTP status indicates that retrying will never succeed and will only
    waste money on upstream calls (e.g. Grok content generation) that
    precede the failing provider in the pipeline.
    """
