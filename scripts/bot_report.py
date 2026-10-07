"""Read-only bot health, fixed-age reach, and storage inventory; no API calls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.age_metrics import report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    args = p.parse_args()
    root = Path(args.root)
    output = {"free_disk_bytes": shutil.disk_usage(root).free, "accounts": {}, "health": {}}
    health = root / "data/health.json"
    if health.exists():
        output["health"] = json.loads(health.read_text())
        due = output["health"].get("next_post_at")
        output["overdue_seconds"] = max(0, round(time.time() - due)) if isinstance(due, (int, float)) else None
    for f in sorted((root / "data").glob("post_history*.json")):
        output["accounts"][f.stem] = report(json.loads(f.read_text()))
    # Inventory only. Source photos, published media and DBs are never removed.
    output["retention_inventory"] = {}
    for folder in ("Images", "Videos", "Voices", "data"):
        files = [f for f in (root / folder).rglob("*") if f.is_file()]
        output["retention_inventory"][folder] = {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
