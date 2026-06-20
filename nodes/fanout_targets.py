"""
Node: fanout_targets

Runs AFTER the base German→English tweet has been posted + scored. For each
configured secondary target (data/secondary_targets.json), it re-uses the SAME
already-generated image to produce and post a transcreated tweet in another
language pair (first: English→Chinese) to that target's own X account.

Per the single-loop design, everything here runs sequentially in one process, so
there is no concurrency / GPU contention with the base cycle.

CRITICAL — failure isolation: each target is wrapped in try/except so NOTHING
propagates to main.py. A secondary TTS error can raise FatalProviderError, which
main.py treats as fatal (it stops the bot); we must swallow it here so a flaky
Chinese-side post can never take down the working German bot.

================================================================================
 STATE CONTRACT
================================================================================
  Reads from state:   source_word, example_sentence_source, example_sentence_target,
                      cefr_level, full_tweet, midjourney_prompt, image_path,
                      image_subject_gender, cycle
  Writes to state:    secondary_results [{target_id, tweet_id, tweet_url, id, status}]
  Side effects:       posts tweets on secondary X accounts; writes per-target
                      audio/video + data/post_history.<id>.json; mirrors compliant
                      posts to the eXercise website (services/website_publish)
================================================================================
"""

import logging
import os
import uuid
from datetime import datetime, timezone

import config
from utils.io import safe_json_read, atomic_json_write
from utils.ui import ok, info as ui_info, warn as ui_warn, section_banner

logger = logging.getLogger("xbot.fanout")


def _already_posted_this_cycle(spec, cycle: int) -> bool:
    """Resume-safety: True if this target already has ANY record for *cycle*.

    Records are now written even when the X post fails (the website is the primary
    China channel), so the resume guard keys on the record's existence — not on a
    tweet_id — to avoid re-running (and double-mirroring) a cycle after a crash.
    """
    hist = safe_json_read(spec.history_file, default=[], logger=logger)
    if not isinstance(hist, list):
        return False
    return any(r.get("cycle") == cycle for r in hist)


def _record_history(spec, state: dict, tc: dict, tweet_id: str, tweet_url: str,
                    video_path: str, image_path: str) -> dict:
    """Append a structured record (everything the website needs) and return it.

    ``id`` is stable: the X tweet_id when present, else a uuid4 — so re-pushes /
    catch-ups overwrite the same eXercise entry (idempotent). ``website_pushed``
    starts False; services/website_publish flips it once the site has the post.
    """
    stable_id = (str(tweet_id).strip() or uuid.uuid4().hex)
    rec = {
        "id": stable_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_id": spec.id,
        "tweet_id": tweet_id,
        "tweet_url": tweet_url,
        "full_tweet": tc["full_tweet"],
        "source_word": tc.get("source_word", ""),
        "source_sentence": tc.get("source_sentence", ""),
        "audience_word": tc.get("audience_word", ""),
        "audience_sentence": tc.get("audience_sentence", ""),
        "cefr_level": tc.get("cefr", ""),
        "video_file": os.path.basename(video_path) if video_path else "",
        "video_path": video_path or "",
        "image_file": os.path.basename(image_path) if image_path else "",
        "image_path": image_path or "",
        "cycle": state.get("cycle", 0),
        "website_pushed": False,
    }
    hist = safe_json_read(spec.history_file, default=[], logger=logger)
    if not isinstance(hist, list):
        hist = []
    hist.append(rec)
    atomic_json_write(spec.history_file, hist, ensure_ascii=False, indent=2)
    return rec


def _run_one_target(spec, state: dict) -> dict:
    """Transcreate → re-TTS → re-video (shared image) → publish for one target."""
    from services.transcreate import transcreate
    from services.targets import target_config
    from nodes.generate_audio import generate_audio
    from nodes.create_video import create_video
    from nodes.publish import post_tweet_with_video

    if not state.get("image_path"):
        raise RuntimeError("no image_path — cannot build a secondary video")
    base = {
        "source_word": state.get("source_word", ""),
        "article": state.get("article", ""),
        "example_sentence_source": state.get("example_sentence_source", ""),
        "example_sentence_target": state.get("example_sentence_target", ""),
        "full_tweet": state.get("full_tweet", ""),
        "cefr_level": state.get("cefr_level", ""),
        "midjourney_prompt": state.get("midjourney_prompt", ""),
    }
    if not base["example_sentence_target"]:
        raise RuntimeError("base has no target-language sentence to transcreate from")

    cycle = state.get("cycle", 0)
    section_banner(
        "🌐",
        f"SECONDARY POST — {spec.source_language} → {spec.target_language}",
        f"target: {spec.id}",
    )
    tc = transcreate(spec, base, cycle=cycle)   # prints candidates, pick, EN/ZH breakdown, tweet_box

    # Content-compliance gate (fail-closed): publish only on a positive verdict.
    # Runs before any media so a block/unverified wastes no TTS/video. On a
    # technical failure it retries; if it can't verify, it skips this cycle (the
    # next loop generates fresh content and re-checks) — never posts unverified.
    compliance_status = ""   # "" (no policy) | "ok" — surfaced to the China watchdog as "verified"
    if getattr(spec, "content_policy", ""):
        from services.content_safety import check_compliance
        ui_info(f"[{spec.id}] checking {spec.content_policy} compliance ({config.CONTENT_SAFETY_MODEL}) …")
        ok_to_post, status, reason = check_compliance(tc["full_tweet"], spec.content_policy)
        if not ok_to_post:
            if status == "blocked":
                ui_warn(f"[{spec.id}] BLOCKED by {spec.content_policy} compliance: {reason} — not posting")
            else:
                ui_warn(f"[{spec.id}] compliance UNVERIFIED ({reason}) — skipping; next loop will retry")
            return {"target_id": spec.id, "tweet_id": "", "tweet_url": "",
                    "status": status, "reason": reason}
        ui_info(f"[{spec.id}] {spec.content_policy} compliance: OK")
        compliance_status = "ok"

    # Sub-state: NEVER merged into the shared state (keeps base artifacts intact).
    # The taught-language sentence becomes example_sentence_source → spoken + KTV-subtitled.
    sub = {
        "example_sentence_source": tc["source_sentence"],
        "full_tweet": tc["full_tweet"],
        "image_path": state["image_path"],            # shared, flag-free
        "midjourney_prompt": state.get("midjourney_prompt", ""),
        "image_subject_gender": state.get("image_subject_gender", "neutral"),
        "cycle": cycle,
    }
    with target_config(spec):
        sub = generate_audio(sub)     # English voice (config.SOURCE_LANGUAGE_CODE overridden)
        sub = create_video(sub)       # shared image, target flags, English KTV, static (ENABLE_VIDEO=off)

    video_path = sub.get("video_path")
    if not video_path:
        raise RuntimeError("secondary video was not produced")
    image_path = state.get("image_path", "")     # shared, flag-free base image → website poster

    if config.FANOUT_DRY_RUN:
        ui_info(f"   [{spec.id}] DRY RUN — not posting. video={video_path}")
        return {"target_id": spec.id, "tweet_id": "", "tweet_url": "",
                "video_path": video_path, "status": compliance_status, "dry_run": True}

    # X post and website mirror are DECOUPLED. The eXercise website is the PRIMARY
    # channel for China (X is blocked there), so a failed/absent X post must NOT stop
    # the (already compliance-passed) content from being recorded + later mirrored.
    tweet_id, tweet_url = "", ""
    if not spec.account.is_complete():
        ui_warn(
            f"[{spec.id}] X creds incomplete (TWITTER_*_{spec.account_env_prefix}) — "
            f"skipping X post; content is still recorded + mirrored to the website."
        )
    else:
        try:
            tweet_id, tweet_url = post_tweet_with_video(tc["full_tweet"], video_path, creds=spec.account)
            ok(f"[{spec.id}] posted → {tweet_url}")
        except Exception as exc:
            logger.exception("X post failed for '%s': %s", spec.id, exc)
            ui_warn(f"[{spec.id}] X post failed ({exc}) — content still recorded + mirrored to website.")

    rec = _record_history(spec, state, tc, tweet_id, tweet_url, video_path, image_path)
    return {"target_id": spec.id, "tweet_id": tweet_id, "tweet_url": tweet_url,
            "id": rec["id"], "status": compliance_status, "dry_run": False}


def fanout_targets(state: dict) -> dict:
    targets = list(getattr(config, "SECONDARY_TARGETS", []) or [])
    if not targets:
        return state

    logger.info("Node: fanout_targets (%d target(s))", len(targets))
    ui_info(f"🌐 Secondary targets: {len(targets)}")

    results = list(state.get("secondary_results", []) or [])
    # A prior dry-run result only counts as "handled" if dry-run is still on.
    # If FANOUT_DRY_RUN was turned off, we must re-run those targets for real.
    handled = {
        r.get("target_id") for r in results
        if r.get("tweet_id") or (r.get("dry_run") and config.FANOUT_DRY_RUN)
    }
    cycle = state.get("cycle", 0)

    this_cycle: list = []
    for spec in targets:
        if spec.id in handled:
            ui_info(f"[{spec.id}] already handled this cycle — skipping.")
            continue
        if not config.FANOUT_DRY_RUN and _already_posted_this_cycle(spec, cycle):
            ui_info(f"[{spec.id}] already posted for cycle {cycle} (resume) — skipping.")
            continue
        try:
            res = _run_one_target(spec, state)
            results.append(res)
            this_cycle.append((spec, res))
        except Exception as exc:
            # Includes FatalProviderError (RuntimeError subclass). Must NOT bubble
            # up — the base cycle has already posted; a secondary failure must not
            # stop the bot or trip the consecutive-failure / fatal-error paths.
            logger.exception("Fan-out target '%s' failed: %s", spec.id, exc)
            ui_warn(f"[{spec.id}] fan-out failed ({exc}) — base bot continues.")
            this_cycle.append((spec, {"target_id": spec.id, "error": str(exc)}))

    # Compliance watchdog for content_policy targets. Runs OUTSIDE the per-target
    # try/except so it CAN stop the bot: if the China check is UNVERIFIED (couldn't
    # run) too many cycles in a row, register_cycle raises FatalProviderError →
    # main.py stops the bot. A "blocked" verdict counts as verified (resets it).
    china = [(s, r) for s, r in this_cycle if getattr(s, "content_policy", "")]
    if china:
        from services.content_safety import register_cycle
        register_cycle(
            unverified=any(r.get("status") == "unverified" for _, r in china),
            verified=any(
                r.get("status") in ("blocked", "ok") or r.get("tweet_id") or r.get("dry_run")
                for _, r in china
            ),
        )

    # ── Mirror compliant posts to the eXercise website (China-facing channel) ──
    # The website is the primary channel for China (X is blocked there). sync pushes
    # this cycle's record plus any earlier un-pushed backlog whose local video still
    # exists, so a website outage self-heals next cycle. Failure-isolated (never
    # raises). Skipped in dry-run and when EXERCISE_INGEST_URL is unset.
    if not config.FANOUT_DRY_RUN and (getattr(config, "EXERCISE_INGEST_URL", "") or "").strip():
        from services.website_publish import sync as website_sync
        ui_info("🌐 eXercise website mirror:")
        for spec in targets:
            res = website_sync(spec)               # prints ✓/✗ per pushed record
            if not res.get("error") and not res.get("pending"):
                ui_info(f"   [{spec.id}] up to date")

    return {**state, "secondary_results": results}
