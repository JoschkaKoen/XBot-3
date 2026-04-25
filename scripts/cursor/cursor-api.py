#!/usr/bin/env python3
"""
cursor-api — instantly switch Cursor's OpenAI-compatible endpoint + API key.

Usage:
    python cursor-api.py zai        # switch to z.ai / GLM
    python cursor-api.py minimax    # switch to MiniMax
    python cursor-api.py default    # switch back to Cursor Pro (no override)
    python cursor-api.py status     # show current setting

Config file: ~/.cursor-api-presets.json
  {
    "zai":     { "key": "YOUR_ZAI_KEY",     "base_url": "https://api.z.ai/api/coding/paas/v4" },
    "minimax": { "key": "YOUR_MINIMAX_KEY", "base_url": "https://api.minimax.chat/v1" }
  }

The script automatically closes Cursor, writes the settings, then relaunches it.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
DB       = Path.home() / ".config/Cursor/User/globalStorage/state.vscdb"
CFG_FILE = Path.home() / ".cursor-api-presets.json"

REACTIVE_KEY = (
    "src.vs.platform.reactivestorage.browser.reactiveStorageServiceImpl"
    ".persistentStorage.applicationUser"
)
AUTH_KEY = "cursorAuth/openAIKey"

# ── built-in presets (key/base_url filled from config file) ───────────────────
BUILTIN = {
    "zai": {
        "label":    "z.ai  (GLM)",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
    },
    "minimax": {
        "label":    "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
    },
    "default": {
        "label":    "Cursor Pro  (no override)",
        "base_url": None,
        "key":      "",
    },
}

CURSOR_BIN = "/usr/share/cursor/cursor"

# ── helpers ────────────────────────────────────────────────────────────────────
def cursor_pids() -> list[int]:
    """Return PIDs of all running Cursor main processes."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", CURSOR_BIN], text=True
        ).strip()
        return [int(p) for p in out.splitlines() if p.strip()]
    except subprocess.CalledProcessError:
        return []


def kill_cursor() -> bool:
    """Gracefully close Cursor; wait up to 8 s; force-kill if needed.
    Returns True if Cursor was running."""
    pids = cursor_pids()
    if not pids:
        return False

    # Find the main process (lowest PID = parent)
    main_pid = min(pids)
    print(f"Closing Cursor (pid {main_pid}) …", flush=True)
    os.kill(main_pid, signal.SIGTERM)

    for _ in range(16):          # wait up to 8 s
        time.sleep(0.5)
        if not cursor_pids():
            return True

    print("Force-killing Cursor …", flush=True)
    for pid in cursor_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    return True


def launch_cursor() -> None:
    """Start Cursor in the background."""
    subprocess.Popen(
        [CURSOR_BIN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print("Cursor relaunched.")


def load_config() -> dict:
    if not CFG_FILE.exists():
        return {}
    with open(CFG_FILE) as f:
        return json.load(f)


def read_current() -> tuple[str, str]:
    """Return (current_key, current_base_url)."""
    conn = sqlite3.connect(DB)
    cur  = conn.cursor()
    cur.execute("SELECT value FROM ItemTable WHERE key=?", (AUTH_KEY,))
    key_row = cur.fetchone()
    cur.execute("SELECT value FROM ItemTable WHERE key=?", (REACTIVE_KEY,))
    blob_row = cur.fetchone()
    conn.close()
    key      = key_row[0]  if key_row  else ""
    base_url = None
    if blob_row:
        try:
            data     = json.loads(blob_row[0])
            base_url = data.get("openAIBaseUrl")
        except Exception:
            pass
    return key or "", base_url or ""


def apply(api_key: str, base_url: str | None) -> None:
    """Write key + base_url into Cursor's SQLite state DB."""
    conn = sqlite3.connect(DB)
    cur  = conn.cursor()

    # 1. API key
    cur.execute("UPDATE ItemTable SET value=? WHERE key=?", (api_key, AUTH_KEY))

    # 2. base_url + useOpenAIKey flag in reactive storage blob
    cur.execute("SELECT value FROM ItemTable WHERE key=?", (REACTIVE_KEY,))
    row = cur.fetchone()
    if row:
        data = json.loads(row[0])
        data["openAIBaseUrl"] = base_url
        data["useOpenAIKey"]  = bool(api_key)
        cur.execute(
            "UPDATE ItemTable SET value=? WHERE key=?",
            (json.dumps(data, ensure_ascii=False), REACTIVE_KEY),
        )

    conn.commit()
    conn.close()


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "status":
        key, url = read_current()
        if not url:
            print("Current: Cursor Pro  (no override)")
        else:
            cfg = load_config()
            label = url
            for name, p in {**BUILTIN, **cfg}.items():
                if isinstance(p, dict) and p.get("base_url") == url:
                    label = BUILTIN.get(name, {}).get("label", name)
                    break
            print(f"Current: {label}")
            print(f"  base_url : {url}")
            print(f"  key      : {key[:12]}…" if len(key) > 12 else f"  key      : {key}")
        return

    presets = {**BUILTIN}
    user_cfg = load_config()
    for name, vals in user_cfg.items():
        if name in presets:
            presets[name]["key"] = vals.get("key", "")
            if "base_url" in vals:
                presets[name]["base_url"] = vals["base_url"]
        else:
            presets[name] = vals

    if cmd not in presets:
        print(f"Unknown preset '{cmd}'.  Available: {', '.join(presets)}")
        sys.exit(1)

    preset = presets[cmd]
    key     = preset.get("key", "")
    url     = preset.get("base_url")

    if cmd != "default" and not key:
        print(
            f"No API key found for '{cmd}'.\n"
            f"Add it to {CFG_FILE}:\n"
            f'  {{""{cmd}"": {{"key": "YOUR_KEY", "base_url": "{url}"}}}}'
        )
        sys.exit(1)

    was_running = kill_cursor()

    apply(key, url)

    label = preset.get("label", cmd)
    print(f"Switched to: {label}")
    if url:
        print(f"  base_url : {url}")
        print(f"  key      : {key[:12]}…" if len(key) > 12 else f"  key      : {key}")

    if was_running:
        time.sleep(0.5)   # let the DB flush fully
        launch_cursor()
    else:
        print("Cursor was not running — start it manually.")


if __name__ == "__main__":
    main()
