"""Owner-checked mkdir locks with kernel-serialized acquisition/reclamation.

The persistent sibling guard is never unlinked: all contenders flock the same inode.
Only dead owners are reclaimed; elapsed time alone cannot steal a live operation.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return pid > 0


def operate(action: str, path: Path, pid: int, token: str = "") -> str | None:
    with Path(str(path) + ".guard").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        owner = {}
        if path.exists():
            try:
                owner = json.loads((path / "owner.json").read_text())
            except (OSError, ValueError):
                pass
        if action == "release":
            if owner.get("pid") != pid or owner.get("token") != token:
                return None
        elif path.exists():
            try:
                holder = int((path / "pid").read_text().strip())
            except (OSError, ValueError):
                holder = 0
            if holder and _alive(holder):
                return None
            # A legacy holder may still be between mkdir and writing its PID.
            if not holder and time.time() - path.stat().st_mtime < 60:
                return None
        if path.exists():
            # Unknown files prevent reclamation rather than being recursively deleted.
            if any(p.name not in {"pid", "owner.json"} for p in path.iterdir()):
                return None
            for name in ("pid", "owner.json"):
                (path / name).unlink(missing_ok=True)
            path.rmdir()
        if action == "release":
            return "released"
        path.mkdir()
        token = uuid.uuid4().hex
        (path / "pid").write_text(str(pid))
        (path / "owner.json").write_text(json.dumps({"pid": pid, "token": token}))
        return token


def main() -> int:
    action, name, pid, *rest = sys.argv[1:]
    if action not in {"acquire", "release"} or int(pid) <= 0:
        return 2
    result = operate(action, Path(name), int(pid), rest[0] if rest else "")
    if result is None:
        return 1
    if action == "acquire":
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
