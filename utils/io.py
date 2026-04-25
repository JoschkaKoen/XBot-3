"""
utils/io  — safe file-write helpers

atomic_json_write(path, data, **json_kwargs)
    Writes JSON to a temp file in the same directory, then renames it over
    the target. On Linux os.replace() is atomic, so readers always see either
    the old complete file or the new complete file — never a half-written one.
"""

import json
import os
import tempfile


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
