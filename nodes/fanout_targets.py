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
  Writes to state:    secondary_results [{target_id, tweet_id, tweet_url}]
  Side effects:       posts tweets on secondary X accounts; writes per-target
                      audio/video + data/post_history.<id>.json
================================================================================
"""

import logging
from datetime import datetime, timezone

import config
from utils.io import safe_json_read, atomic_json_write
from utils.ui import ok, info as ui_info, warn as ui_warn, section_banner

logger = logging.getLogger("xbot.fanout")


def _already_posted_this_cycle(spec, cycle: int) -> bool:
    """Resume-safety: True if this target already has a posted record for *cycle*."""
    hist = safe_json_read(spec.history_file, default=[], logger=logger)
    if not isinstance(hist, list):
        return False
    return any(r.get("cycle") == cycle and r.get("tweet_id") for r in hist)


def _record_history(spec, state: dict, tc: dict, tweet_id: str, tweet_url: str) -> None:
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_id": spec.id,
        "tweet_id": tweet_id,
        "tweet_url": tweet_url,
        "full_tweet": tc["full_tweet"],
        "source_word": tc["source_word"],
        "cefr_level": tc.get("cefr", ""),
        "cycle": state.get("cycle", 0),
    }
    hist = safe_json_read(spec.history_file, default=[], logger=logger)
    if not isinstance(hist, list):
        hist = []
    hist.append(rec)
    atomic_json_write(spec.history_file, hist, ensure_ascii=False, indent=2)


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

    if config.FANOUT_DRY_RUN:
        ui_info(f"   [{spec.id}] DRY RUN — not posting. video={video_path}")
        return {"target_id": spec.id, "tweet_id": "", "tweet_url": "",
                "video_path": video_path, "dry_run": True}

    if not spec.account.is_complete():
        raise RuntimeError(
            f"account creds for '{spec.id}' incomplete — set "
            f"TWITTER_CONSUMER_KEY/SECRET + TWITTER_ACCESS_TOKEN/SECRET with suffix "
            f"_{spec.account_env_prefix} in .env"
        )

    tweet_id, tweet_url = post_tweet_with_video(tc["full_tweet"], video_path, creds=spec.account)
    ok(f"[{spec.id}] posted → {tweet_url}")
    _record_history(spec, state, tc, tweet_id, tweet_url)
    return {"target_id": spec.id, "tweet_id": tweet_id, "tweet_url": tweet_url, "dry_run": False}


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
                r.get("status") == "blocked" or r.get("tweet_id") or r.get("dry_run")
                for _, r in china
            ),
        )

    return {**state, "secondary_results": results}
