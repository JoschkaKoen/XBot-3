"""
utils/io — safe file I/O helpers.

atomic_json_write(path, data, **json_kwargs)
    Writes JSON to a temp file in the same directory, then renames it over
    the target. On Linux os.replace() is atomic, so readers always see either
    the old complete file or the new complete file — never a half-written one.

safe_json_read(path, default=None, *, logger=None)
    Reads JSON from *path*. If the file is missing or corrupt, returns
    *default* and (if *logger* is supplied) emits a warning. Use this instead
    of hand-rolled try/except blocks at every call site.
"""

import json
import logging
import os
import tempfile
from typing import Any


def atomic_json_write(path: str, data, **json_kwargs) -> None:
    """Write *data* as JSON to *path* atomically (write-then-rename)."""
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)
        os.replace(tmp, path)
    except Exception:
        # Bare-except is intentional here: we re-raise immediately, and the
        # only purpose of this block is to clean up the temp file. We must
        # not swallow whatever exception triggered the cleanup.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def safe_json_read(
    path: str,
    default: Any = None,
    *,
    logger: logging.Logger | None = None,
) -> Any:
    """
    Read JSON from *path*. Return *default* (or {} if default is None) when
    the file is missing or unreadable. When *logger* is given, log a warning
    on parse/IO errors so silent corruption is at least visible.
    """
    fallback = {} if default is None else default
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        if logger:
            logger.warning("Could not read %s: %s — using default.", path, exc)
        return fallback
