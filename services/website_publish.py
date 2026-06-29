"""
services/website_publish — push compliant secondary posts to the eXercise site.

China context: X (Twitter) is blocked in mainland China, so the English→Chinese
posts are mirrored to the eXercise website (Singapore, reachable from China) with
the video + text embedded directly — no x.com resources. After each live secondary
cycle, :func:`sync` uploads any history record not yet on the site.

  sync(spec):
    • read spec.history_file
    • for each record with website_pushed != True whose local video still exists,
      POST (multipart: meta JSON + video mp4 [+ png poster]) to EXERCISE_INGEST_URL
      with a Bearer EXERCISE_INGEST_TOKEN header
    • on success set website_pushed=True and atomically rewrite the history file

Design:
  • NEVER raises — a website outage must not disturb the bot (the per-record
    website_pushed flag means a failed push self-heals on the next cycle).
  • Idempotent by record ``id`` (the server overwrites by id), so re-pushes /
    catch-ups are safe.
  • Pushes DIRECT by default (requests Session with trust_env=False) so the upload
    bypasses the local GFW/X proxy — exercises.lol is not blocked and the proxy is
    only for blocked sites. Set EXERCISE_INGEST_DIRECT=off to use the env proxy.
  • Retries only TRANSIENT failures (timeout / connection error / 5xx / HTTP 425
    early-data); never retries 400/401/413 (those are config errors → logged loudly).
"""

from __future__ import annotations

import json
import logging
import os
import time

import requests

import config
from utils.io import atomic_json_write, safe_json_read
from utils.ui import info as ui_info, warn as ui_warn

logger = logging.getLogger("xbot.website")

_TIMEOUT = (10, 60)                       # (connect, read) seconds — generous for the high-RTT link
_RETRY_STATUS = {425, 500, 502, 503, 504}  # 425 = early-data (nginx ssl_early_data); transient 5xx
_MAX_ATTEMPTS = 3
# Poster MIME by extension. The base image is usually PNG, but z-image-turbo can
# preserve a ComfyUI .jpg/.webp output — accept those instead of silently
# dropping the poster (which a .png-only check would do).
_POSTER_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _meta_for(rec: dict) -> str:
    """Build the meta JSON the eXercise page renders (keys mirror english.html)."""
    return json.dumps(
        {
            "id": rec.get("id", ""),
            "timestamp": rec.get("timestamp", ""),
            "source_word": rec.get("source_word", ""),
            "audience_word": rec.get("audience_word", ""),
            "source_sentence": rec.get("source_sentence", ""),
            "audience_sentence": rec.get("audience_sentence", ""),
            "cefr": rec.get("cefr_level", ""),
            "full_tweet": rec.get("full_tweet", ""),
            "tweet_url": rec.get("tweet_url", ""),
        },
        ensure_ascii=False,
    )


def _push_record(session: requests.Session, url: str, token: str, rec: dict) -> tuple[bool, str]:
    """POST one record. Returns (ok, message). Retries only transient failures."""
    video_path = rec.get("video_path") or ""
    if not video_path or not os.path.isfile(video_path):
        return False, "local video missing"          # can't push without media (e.g. legacy record)

    image_path = rec.get("image_path") or ""
    poster_ext = os.path.splitext(image_path)[1].lower()
    has_poster = bool(image_path) and os.path.isfile(image_path) and poster_ext in _POSTER_MIME
    meta = _meta_for(rec)
    headers = {"Authorization": f"Bearer {token}"}

    backoff = 2.0
    last = "failed"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        handles = []
        try:
            vf = open(video_path, "rb")
            handles.append(vf)
            files = {"video": (os.path.basename(video_path), vf, "video/mp4")}
            if has_poster:
                pf = open(image_path, "rb")
                handles.append(pf)
                files["poster"] = (os.path.basename(image_path), pf, _POSTER_MIME[poster_ext])
            resp = session.post(url, headers=headers, data={"meta": meta}, files=files, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return True, "ok"
            if resp.status_code in _RETRY_STATUS:
                last = f"HTTP {resp.status_code}"      # transient → retry
            else:
                # 400/401/413/etc — config error, not worth retrying. Log loudly.
                return False, f"HTTP {resp.status_code} {resp.text[:160]}"
        except requests.RequestException as exc:
            last = type(exc).__name__                  # timeout / connection reset → retry
        finally:
            for h in handles:
                try:
                    h.close()
                except OSError:
                    pass
        if attempt < _MAX_ATTEMPTS:
            time.sleep(backoff)
            backoff *= 2
    return False, last


def sync(spec) -> dict:
    """Push all not-yet-published records for *spec* to the eXercise site.

    Returns a summary dict ``{pushed, failed, pending, skipped}``. Never raises.
    """
    url = (getattr(config, "EXERCISE_INGEST_URL", "") or "").strip()
    token = (getattr(config, "EXERCISE_INGEST_TOKEN", "") or "").strip()
    if not url or not token:
        return {"skipped": "not configured", "pushed": 0, "failed": 0, "pending": 0}

    try:
        hist = safe_json_read(spec.history_file, default=[], logger=logger)
        if not isinstance(hist, list):
            return {"pushed": 0, "failed": 0, "pending": 0}
        pending = [r for r in hist if r.get("id") and not r.get("website_pushed")]
        if not pending:
            return {"pushed": 0, "failed": 0, "pending": 0}

        session = requests.Session()
        if getattr(config, "EXERCISE_INGEST_DIRECT", True):
            session.trust_env = False     # bypass HTTP(S)_PROXY (the GFW/X proxy)

        pushed = failed = 0
        changed = False
        for rec in pending:
            ok_push, msg = _push_record(session, url, token, rec)
            if ok_push:
                rec["website_pushed"] = True
                changed = True
                pushed += 1
                ui_info(f"   ↳ [{spec.id}] web ✓ {rec.get('id')}")
            else:
                failed += 1
                ui_warn(f"   ↳ [{spec.id}] web ✗ {rec.get('id')}: {msg}")

        if changed:
            atomic_json_write(spec.history_file, hist, ensure_ascii=False, indent=2)
        return {"pushed": pushed, "failed": failed, "pending": len(pending)}
    except Exception as exc:                    # failure-isolated: never disturb the bot
        logger.exception("website_publish.sync failed for '%s': %s", spec.id, exc)
        ui_warn(f"   ↳ [{spec.id}] web sync error ({exc}) — base bot continues.")
        return {"pushed": 0, "failed": 0, "pending": 0, "error": str(exc)}
