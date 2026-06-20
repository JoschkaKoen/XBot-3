"""
services/targets — secondary language-pair targets for the in-cycle fan-out.

The bot runs ONE loop (German→English). After the base tweet + image are made,
``nodes/fanout_targets`` re-uses the SAME image to produce, for each configured
secondary target, a transcreated tweet in another language pair (e.g. English→
Chinese) and posts it to that target's own X account.

Targets are declared in ``data/secondary_targets.json`` (public config: flags,
codes, length limits) — adding a pair is one JSON entry plus that account's
``TWITTER_*_<PREFIX>`` keys in ``.env`` (gitignored).

This module deliberately does NOT ``import config`` at module top: ``config``
imports ``load_targets()`` at the end of its own initialisation, so a top-level
``import config`` here would be a partial-import foot-gun. ``config`` is imported
lazily inside :func:`target_config` (only ever called at runtime).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

# NB: keep module-level imports stdlib-only. ``config`` imports load_targets() at
# the end of its own initialisation; importing utils/config here would create a
# circular import (utils/__init__ → utils.ui → config). utils.io is imported
# lazily inside load_targets(), and config/nodes lazily inside target_config().

_TARGETS_FILE = "data/secondary_targets.json"
_SENTINEL = object()


@dataclass(frozen=True)
class AccountCreds:
    """OAuth 1.0a user-context credentials for one X account (+ optional bearer)."""
    consumer_key: str = ""
    consumer_secret: str = ""
    access_token: str = ""
    access_token_secret: str = ""
    bearer_token: str = ""

    def is_complete(self) -> bool:
        """True iff the four OAuth1 values needed to POST are all present."""
        return all([
            self.consumer_key,
            self.consumer_secret,
            self.access_token,
            self.access_token_secret,
        ])


@dataclass(frozen=True)
class TargetSpec:
    """Everything language/account/path-specific for one secondary target."""
    id: str
    source_language: str          # taught + spoken + KTV-subtitled (e.g. "English")
    target_language: str          # audience / tweet-text explanation (e.g. "Chinese")
    source_language_code: str     # ISO 639-1, e.g. "en" → list_voices()
    target_language_code: str
    source_flag: str
    target_flag: str
    source_flag_colors: str
    target_flag_colors: str
    script: str                   # e.g. "simplified"
    tts_speed: float
    max_tweet_length: int         # per-account tier; X weights CJK ~2x
    account: AccountCreds
    account_env_prefix: str       # e.g. "ZH" → TWITTER_*_ZH
    voices_dir: str
    videos_dir: str
    voices_music_dir: str
    history_file: str
    content_policy: str = ""       # e.g. "china" → screen each post before publishing
    enabled: bool = True

    def hashtags(self) -> str:
        """House-style hashtags for the taught language, e.g. '#LearnEnglish #EnglishVocabulary'."""
        lang = self.source_language.replace(" ", "")
        return f"#Learn{lang} #{lang}Vocabulary"


def _creds_for(prefix: str) -> AccountCreds:
    """Read TWITTER_<FIELD>_<PREFIX> + X_BEARER_TOKEN_<PREFIX> from the environment."""
    g = lambda field_: os.getenv(f"TWITTER_{field_}_{prefix}", "").strip()
    return AccountCreds(
        consumer_key=g("CONSUMER_KEY"),
        consumer_secret=g("CONSUMER_SECRET"),
        access_token=g("ACCESS_TOKEN"),
        access_token_secret=g("ACCESS_TOKEN_SECRET"),
        bearer_token=os.getenv(f"X_BEARER_TOKEN_{prefix}", "").strip(),
    )


def _spec_from_dict(t: dict) -> TargetSpec:
    tid = str(t["id"]).strip()
    prefix = str(t.get("account_env_prefix") or tid).strip().upper()
    return TargetSpec(
        id=tid,
        source_language=t["source_language"],
        target_language=t["target_language"],
        source_language_code=t.get("source_language_code", ""),
        target_language_code=t.get("target_language_code", ""),
        source_flag=t.get("source_flag", ""),
        target_flag=t.get("target_flag", ""),
        source_flag_colors=t.get("source_flag_colors", ""),
        target_flag_colors=t.get("target_flag_colors", ""),
        script=t.get("script", "simplified"),
        # Default mirrors config.TTS_SPEED's default ("0.70") without importing config.
        tts_speed=float(t.get("tts_speed", os.getenv("TTS_SPEED", "0.70") or "0.70")),
        max_tweet_length=int(t.get("max_tweet_length", 280)),
        account=_creds_for(prefix),
        account_env_prefix=prefix,
        voices_dir=t.get("voices_dir", f"Voices/{tid}"),
        videos_dir=t.get("videos_dir", f"Videos/{tid}"),
        voices_music_dir=t.get("voices_music_dir", f"Voices with Background Music/{tid}"),
        history_file=t.get("history_file", f"data/post_history.{tid}.json"),
        content_policy=str(t.get("content_policy", "")).strip(),
        enabled=bool(t.get("enabled", True)),
    )


def load_targets() -> list[TargetSpec]:
    """Load enabled secondary targets from data/secondary_targets.json (→ [] if absent/malformed)."""
    from utils.io import safe_json_read  # lazy: avoids config import cycle
    raw = safe_json_read(_TARGETS_FILE, default=[])
    if not isinstance(raw, list):
        return []
    specs: list[TargetSpec] = []
    for entry in raw:
        try:
            spec = _spec_from_dict(entry)
        except (KeyError, TypeError, ValueError):
            continue
        if spec.enabled:
            specs.append(spec)
    return specs


@contextmanager
def target_config(spec: TargetSpec):
    """
    Temporarily swap the language/flag/dir globals so the existing
    ``generate_audio`` / ``create_video`` nodes render for *spec* instead of the
    base German→English pair, then restore everything.

    Safe because the fan-out is strictly sequential (one process, no threads
    competing on config). Patches both ``config.*`` (read at call time) AND the
    import-bound dir names inside the node modules (``from config import
    VOICES_DIR`` captures the value at import, so patching ``config`` alone would
    not redirect output). ``ENABLE_VIDEO`` is forced to ``"off"`` so secondaries
    build a static KTV video from the SHARED image — no I2V regeneration and no
    double-advance of the video-frequency gate.
    """
    import importlib
    import config
    # NB: ``import nodes.create_video`` would resolve to the create_video *function*
    # (nodes/__init__.py re-exports it, shadowing the submodule attribute), so use
    # import_module to get the actual module objects whose globals we patch.
    _ga = importlib.import_module("nodes.generate_audio")
    _cv = importlib.import_module("nodes.create_video")

    for d in (spec.voices_dir, spec.videos_dir, spec.voices_music_dir):
        os.makedirs(d, exist_ok=True)

    overrides = [
        (config, "SOURCE_LANGUAGE", spec.source_language),
        (config, "TARGET_LANGUAGE", spec.target_language),
        (config, "SOURCE_LANGUAGE_CODE", spec.source_language_code),
        (config, "TARGET_LANGUAGE_CODE", spec.target_language_code),
        (config, "SOURCE_FLAG", spec.source_flag),
        (config, "TARGET_FLAG", spec.target_flag),
        (config, "SOURCE_FLAG_COLORS", spec.source_flag_colors),
        (config, "TARGET_FLAG_COLORS", spec.target_flag_colors),
        (config, "TTS_SPEED", spec.tts_speed),
        (config, "ENABLE_VIDEO", "off"),
        (_ga, "VOICES_DIR", spec.voices_dir),
        (_cv, "VIDEOS_DIR", spec.videos_dir),
        (_cv, "VOICES_MUSIC_DIR", spec.voices_music_dir),
    ]
    saved = [(obj, attr, getattr(obj, attr, _SENTINEL)) for obj, attr, _ in overrides]
    try:
        for obj, attr, val in overrides:
            setattr(obj, attr, val)
        yield
    finally:
        for obj, attr, old in saved:
            if old is _SENTINEL:
                continue
            setattr(obj, attr, old)
