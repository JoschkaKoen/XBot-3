"""Fixed-age observations using the existing per-cycle request budget."""
from datetime import datetime, timezone
import statistics

WINDOWS = {"24h": (24, 36), "7d": (168, 180)}


def age_hours(row, now=None):
    try:
        ts = datetime.fromisoformat(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ((now or datetime.now(timezone.utc)) - ts).total_seconds() / 3600
    except (ValueError, KeyError, TypeError):
        return None


def due_records(rows, limit, now=None):
    due = []
    for row in rows:
        if not row.get("tweet_id") or row.get("deleted_on_x"):
            continue
        age = age_hours(row, now)
        if age is None:
            continue
        for label, (target, end) in WINDOWS.items():
            if target <= age <= end and label not in row.get("age_metrics", {}):
                due.append((age - target, row))
                break
    return [row for _, row in sorted(due, key=lambda item: item[0])[:max(0, limit)]]


def snapshot(row, metrics, now=None):
    now = now or datetime.now(timezone.utc)
    age = age_hours(row, now)
    result = {**row, "metrics": metrics, "metrics_fetched_at": now.isoformat()}
    observations = dict(row.get("age_metrics", {}))
    if age is not None:
        for label, (target, end) in WINDOWS.items():
            if target <= age <= end and label not in observations:
                observations[label] = {"age_hours": round(age, 3),
                    "fetched_at": now.isoformat(), "metrics": dict(metrics)}
    result["age_metrics"] = observations
    return result


def report(rows):
    out = {"records": len(rows), "delivered_x": sum(bool(r.get("tweet_id")) for r in rows),
           "delivered_website": sum(bool(r.get("website_pushed")) for r in rows)}
    for label in WINDOWS:
        obs = [r["age_metrics"][label] for r in rows if label in r.get("age_metrics", {})]
        out[label] = {"n": len(obs), "median_impressions": statistics.median(
            [o["metrics"].get("impression_count", 0) for o in obs]) if obs else None,
            "observed_ages_hours": [o["age_hours"] for o in obs]}
    return out
