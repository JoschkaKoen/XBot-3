"""
Configuration for XBot-3 Language Learning Bot.

Most settings are read from settings.env (git-tracked) and .env (gitignored, API keys).

================================================================================
 USER-TUNABLE SETTINGS (edit in settings.env)
================================================================================

  LANGUAGE PAIR
    SOURCE_LANGUAGE     → e.g. "German", "Spanish", "French"
    TARGET_LANGUAGE     → e.g. "English"

  CONTENT GENERATION
    IMAGE_STYLE        → "photographic" (realistic) or "disney" (Pixar-style)
    TWEET_STYLE        → "funny" (jokes) or "educational" (plain)
    MAX_EXAMPLE_WORDS  → max words in example sentence (default: 13)

  VIDEO OPTIONS
    ENABLE_VIDEO       → "off" (static image) | "grok" (Grok Imagine) | "WAN2.1" (local Wan2.1)
    VIDEO_STYLE        → "ktv" (karaoke highlight) or "simple"
    KTV_FONT_SIZE      → base subtitle font size (px at ~480p frame height); bar scales with it
    ENABLE_KEN_BURNS   → true/false — slow zoom+pan on static videos
    ENABLE_BACKGROUND_MUSIC → on/off (or true/false) — mix voice with BACKGROUND_MUSIC_PATH (default off)
    VIDEO_FREQUENCY    → generate video every N tweets (1 = every tweet)

  IMAGE GENERATION
    IMAGE_PROVIDER        → "midjourney" (TTAPI), "grok" (xAI), or "z-image-turbo" (ComfyUI local)
    FLAG_OVERLAY          → true/false — show country flags on images
    Z_IMAGE_TURBO_STEPS   → denoising steps for Z-Image-Turbo (default: 8, range 8–9)
    ENABLE_INSTRUCTIR_ENHANCE → true/false — after Z-Image-Turbo, run InstructIR on each candidate (optional; see README)
    INSTRUCTIR_DIR        → path to InstructIR git clone; INSTRUCTIR_PROMPT overrides the enhancement instruction

  X/TWITTER
    USE_TRENDS         → comma cycle: trends | pool | false (or true/false aliases).
                         trends = word from X trends; pool = AI word + random curated theme (German→English);
                         false/off = AI word + strategy topic. Example: trends,trends,pool,pool,trends
    MAX_TWEET_LENGTH   → character limit (280 standard, up to 25000 premium)

  BOT BEHAVIOUR
    POST_INTERVAL_SECONDS → time between posts (18000 = 5 hours)
    AUTO_UPDATE         → true/false — pull GitHub updates after each wait
    STRATEGY_UPDATE_INTERVAL_HOURS → hours between X metrics refresh + strategy
                            re-analysis; set false/off/never/disabled to disable both

  AI MODELS (advanced)
    TWEET_MODEL, STRATEGY_MODEL, WORD_PICK_MODEL, etc.
    Valid values: "flagship", "reasoning", "non-reasoning"

================================================================================
"""

import os
import platform
import logging
import re
from dotenv import load_dotenv

_LOG = logging.getLogger(__name__)


def _parse_strategy_update_interval(raw: str | None) -> tuple[bool, int]:
    """
    Parse STRATEGY_UPDATE_INTERVAL_HOURS.

    Returns (metrics_and_strategy_updates_enabled, interval_hours).
    When enabled is False, the bot never calls the X API to refresh metrics and
    never re-runs LLM strategy analysis (same as leaving metrics_refreshed=False).

    Accepts: false / off / no / never / disabled (case-insensitive) → disabled.
    Accepts: plain integers, or simple expressions like 24*7 or 24 * 7 → hours.
    """
    if raw is None or not str(raw).strip():
        return True, 24
    s = str(raw).strip()
    low = s.lower()
    if low in ("false", "off", "no", "never", "disabled", "none"):
        return False, 24
    # e.g. 24*7, 24 * 7
    mul = re.fullmatch(r"^\s*(\d+)\s*\*\s*(\d+)\s*$", s)
    if mul:
        try:
            return True, int(mul.group(1)) * int(mul.group(2))
        except ValueError:
            pass
    try:
        return True, int(s)
    except ValueError:
        _LOG.warning(
            "Invalid STRATEGY_UPDATE_INTERVAL_HOURS=%r — using 24h. "
            "Use an integer, e.g. 24, 168, or false to disable metrics + strategy updates.",
            raw,
        )
        return True, 24


def _parse_metrics_fetch_max(raw: str | None, analyze_last_n: int) -> int:
    """
    Max tweets to refresh per metrics run (X API calls). Empty/unset → max(analyze_last_n, 30).
    Set to 0 or 'all' / 'unlimited' for no cap (every row in post_history).
    """
    if raw is None or not str(raw).strip():
        return max(analyze_last_n, 30)
    s = str(raw).strip().lower()
    if s in ("0", "all", "unlimited", "none"):
        return 0
    return max(1, int(s))


def _parse_use_trends_mode_cycle(raw: str | None) -> list[str]:
    """
    Parse USE_TRENDS into a cycle of word-source modes.

    Tokens (case-insensitive):
      trends, true, 1, yes, on  → pick word from X trending topics
      pool                       → AI word pick + ephemeral random theme (German→English bank)
      false, 0, no, off, strategy → AI word pick + strategy next_topic / style

    Examples:
      "false"                         → ["strategy"]
      "true"                          → ["trends"]
      "trends,trends,pool,pool,trends" → five-step cycle
      "true,false,false,false"        → trends then three strategy steps (backward compatible)
    """
    if raw is None or not str(raw).strip():
        return ["strategy"]
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return ["strategy"]
    out: list[str] = []
    for p in parts:
        if p in ("trends", "true", "1", "yes", "on"):
            out.append("trends")
        elif p == "pool":
            out.append("pool")
        elif p in ("false", "0", "no", "off", "strategy"):
            out.append("strategy")
        else:
            _LOG.warning("Invalid USE_TRENDS token %r — treating as strategy.", p)
            out.append("strategy")
    return out


def _parse_on_off_env(key: str, default: bool = False) -> bool:
    """
    Parse a boolean env var as true/false, yes/no, on/off, or 1/0.
    Unknown values fall back to *default*.
    """
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    return default


def _parse_ktv_font_size(raw: str | None) -> int:
    """
    KTV karaoke subtitle font size in px at a 720p output frame (standard HD reference).
    At other resolutions the size is scaled proportionally (e.g. ~53px at 480p Wan when
    set to 80). Clamped to a safe range (12–200).
    """
    if raw is None or not str(raw).strip():
        v = 80
    else:
        try:
            v = int(str(raw).strip())
        except ValueError:
            _LOG.warning("Invalid KTV_FONT_SIZE=%r — using 80.", raw)
            v = 80
    return max(12, min(200, v))


# Load public configuration first, then secret keys.
# Values in .env take precedence over settings.env (override=True on .env).
load_dotenv("settings.env")
load_dotenv(override=True)   # loads .env (keys only, gitignored)

# ── Language pair ─────────────────────────────────────────────────────────────
# Only SOURCE_LANGUAGE and TARGET_LANGUAGE are set by the user in settings.env.
# All derived fields (codes, flags, colors, trends country) are auto-resolved
# once via AI and cached in data/language_config.json — call resolve_language_config()
# at startup to populate them.
SOURCE_LANGUAGE: str      = os.getenv("SOURCE_LANGUAGE", "German")
TARGET_LANGUAGE: str      = os.getenv("TARGET_LANGUAGE", "English")

# Derived fields — populated by resolve_language_config() at startup.
SOURCE_LANGUAGE_CODE: str = "de"
TARGET_LANGUAGE_CODE: str = "en"
SOURCE_FLAG: str          = "🇩🇪"
TARGET_FLAG: str          = "🇺🇸"
TRENDS_COUNTRY: str       = "germany"
SOURCE_FLAG_COLORS: str   = "000000,DD0000,FFCE00"
TARGET_FLAG_COLORS: str   = "B22234,FFFFFF,3C3B6E"


def resolve_language_config() -> None:
    """
    Auto-derive language-pair config (codes, flags, colors, trends country) from
    SOURCE_LANGUAGE / TARGET_LANGUAGE and update the global config variables.

    Results are cached in data/language_config.json and only re-generated when
    the language pair changes.  Call this once at bot startup.
    """
    global SOURCE_LANGUAGE_CODE, TARGET_LANGUAGE_CODE
    global SOURCE_FLAG, TARGET_FLAG
    global TRENDS_COUNTRY
    global SOURCE_FLAG_COLORS, TARGET_FLAG_COLORS

    try:
        from services.language_config import resolve as _resolve
        derived = _resolve(SOURCE_LANGUAGE, TARGET_LANGUAGE)
    except Exception as exc:
        logger.warning(
            "Language config resolution failed (%s) — using built-in defaults "
            "(%s / %s).  Update settings.env or fix the error and restart.",
            exc, SOURCE_LANGUAGE_CODE, TARGET_LANGUAGE_CODE,
        )
        return

    SOURCE_LANGUAGE_CODE = derived.get("source_language_code", SOURCE_LANGUAGE_CODE)
    TARGET_LANGUAGE_CODE = derived.get("target_language_code", TARGET_LANGUAGE_CODE)
    SOURCE_FLAG          = derived.get("source_flag",          SOURCE_FLAG)
    TARGET_FLAG          = derived.get("target_flag",          TARGET_FLAG)
    TRENDS_COUNTRY       = derived.get("trends_country",       TRENDS_COUNTRY)
    SOURCE_FLAG_COLORS   = derived.get("source_flag_colors",   SOURCE_FLAG_COLORS)
    TARGET_FLAG_COLORS   = derived.get("target_flag_colors",   TARGET_FLAG_COLORS)


def _parse_flag_colors(env_val: str, default: list) -> list:
    """Parse a comma-separated hex color string into a list of (R, G, B) tuples."""
    try:
        parts = [p.strip() for p in env_val.split(",") if p.strip()]
        result = [tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) for c in parts]
        if len(result) >= 3:
            return result
    except Exception:
        pass
    return default


# ── AI provider ──────────────────────────────────────────────────────────────
AI_PROVIDER: str = os.getenv("AI_PROVIDER", "grok").lower().strip()

# ── Twitter / X ──────────────────────────────────────────────────────────────
X_BEARER_TOKEN: str = os.getenv("X_BEARER_TOKEN", "")
TWITTER_CONSUMER_KEY: str = os.getenv("TWITTER_CONSUMER_KEY", "")
TWITTER_CONSUMER_SECRET: str = os.getenv("TWITTER_CONSUMER_SECRET", "")
TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_TOKEN_SECRET: str = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")

# ── ElevenLabs ───────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

# ── Midjourney via TTAPI ─────────────────────────────────────────────────────
TT_API_KEY: str = os.getenv("TT_API_KEY", "")

# ── Image style ──────────────────────────────────────────────────────────────
# Single value  → every tweet uses that style.
# Comma-separated → cycles deterministically across tweets.
#   "photographic"              → always photographic
#   "disney"                    → always disney
#   "photographic,disney,disney"→ tweet 0: photographic, 1: disney, 2: disney, 3: photographic, …
_IMAGE_STYLE_RAW: str = os.getenv("IMAGE_STYLE", "photographic")
IMAGE_STYLE_CYCLE: list[str] = [s.lower().strip() for s in _IMAGE_STYLE_RAW.split(",") if s.strip()]
if not IMAGE_STYLE_CYCLE:
    IMAGE_STYLE_CYCLE = ["photographic"]

# Convenience alias for single-style setups (first element of the cycle).
IMAGE_STYLE: str = IMAGE_STYLE_CYCLE[0]


def resolve_image_style(cycle: int) -> str:
    """Return the effective image style for the given tweet cycle index."""
    return IMAGE_STYLE_CYCLE[cycle % len(IMAGE_STYLE_CYCLE)]


# ── Tweet style ───────────────────────────────────────────────────────────────
# Single value  → every tweet uses that style.
# Comma-separated → cycles deterministically across tweets.
# Valid values: "funny", "normal"
#   "funny"              → always funny tone
#   "normal"             → always neutral/warm tone
#   "funny,normal,normal"→ tweet 0: funny, 1: normal, 2: normal, 3: funny, …
_TWEET_STYLE_RAW: str = os.getenv("TWEET_STYLE", "funny")
TWEET_STYLE_CYCLE: list[str] = [s.lower().strip() for s in _TWEET_STYLE_RAW.split(",") if s.strip()]
if not TWEET_STYLE_CYCLE:
    TWEET_STYLE_CYCLE = ["funny"]

# Convenience alias for single-style setups (first element of the cycle).
TWEET_STYLE: str = TWEET_STYLE_CYCLE[0]


def resolve_tweet_style(cycle: int) -> str:
    """Return the effective tweet style for the given tweet cycle index."""
    return TWEET_STYLE_CYCLE[cycle % len(TWEET_STYLE_CYCLE)]


def resolve_word_source_mode(cycle: int) -> str:
    """Return 'trends' | 'pool' | 'strategy' for this cycle index."""
    modes = USE_TRENDS_MODE_CYCLE
    return modes[cycle % len(modes)]


def resolve_use_trends(cycle: int) -> bool:
    """True iff this cycle uses X trending topics for word pick."""
    return resolve_word_source_mode(cycle) == "trends"


def reload_settings() -> None:
    """
    Re-read settings.env (and .env) so any changes made between cycles take
    effect on the next cycle without restarting the bot.

    Only the 'live' behavioural settings are updated — static things like file
    paths, API keys, and the KTV font *file* (KTV_FONT path) are left unchanged.
    KTV_FONT_SIZE reloads here so subtitle size can be tuned between cycles.
    """
    load_dotenv("settings.env", override=True)
    load_dotenv(override=True)  # .env (API keys) always wins

    global AI_PROVIDER
    global USE_TRENDS, USE_TRENDS_CYCLE, USE_TRENDS_MODE_CYCLE, TREND_CANDIDATE_LIMIT, CEFR_ROTATION, METRICS_FETCH_PER_CYCLE
    global IMAGE_STYLE_CYCLE, IMAGE_STYLE
    global TWEET_STYLE_CYCLE, TWEET_STYLE
    global IMAGE_PROVIDER, GENERATED_IMAGE_COUNT, INDIVIDUAL_IMAGE_PROMPTS, Z_IMAGE_TURBO_STEPS, Z_IMAGE_TURBO_WIDTH, Z_IMAGE_TURBO_HEIGHT, Z_IMAGE_PROMPT_SUFFIX
    global Z_IMAGE_BASE_MODEL_ID, Z_IMAGE_BASE_STEPS, Z_IMAGE_BASE_GUIDANCE_SCALE, Z_IMAGE_BASE_WIDTH, Z_IMAGE_BASE_HEIGHT, Z_IMAGE_BASE_NEGATIVE_PROMPT
    global ENABLE_INSTRUCTIR_ENHANCE, INSTRUCTIR_DIR, INSTRUCTIR_PROMPT
    global VIDEO_INTERPOLATION, RIFE_DIR, RIFE_PYTHON, VIDEO_UPLOAD_FPS
    global MAX_TWEET_LENGTH, MAX_EXAMPLE_WORDS, POST_INTERVAL_SECONDS, VIDEO_STYLE, ANALYZE_LAST_N
    global FLAG_OVERLAY
    global ENABLE_VIDEO, ENABLE_GROK_VIDEO, VIDEO_FREQUENCY, GROK_VIDEO_FREQUENCY, ENABLE_KEN_BURNS, ENABLE_BACKGROUND_MUSIC, WAN_VIDEO_DIR, WAN_VIDEO_STEPS, WAN_VIDEO_FRAMES, WAN_VIDEO_HISTORY_FILE, KTV_FONT_SIZE, VIDEO_FPS
    global ENABLE_REALESRGAN, REALESRGAN_DIR, REALESRGAN_MODEL, REALESRGAN_OUTSCALE, REALESRGAN_TILE
    global ENABLE_SELF_IMPROVEMENT, IMPROVEMENT_INTERVAL_CYCLES, IMPROVEMENT_SCORE_THRESHOLD
    global MAX_CONSECUTIVE_FAILURES
    global STRATEGY_METRICS_UPDATES_ENABLED, STRATEGY_UPDATE_INTERVAL_HOURS
    global METRICS_FETCH_MAX_TWEETS
    global COMFYUI_ARGS
    global TWEET_MODEL, TWEET_PICKER_MODEL, STRATEGY_MODEL
    global TREND_FILTER_MODEL, WORD_PICK_MODEL, SIMILARITY_MODEL, VOICE_PICKER_MODEL

    AI_PROVIDER                    = os.getenv("AI_PROVIDER", "grok").lower().strip()
    USE_TRENDS_MODE_CYCLE          = _parse_use_trends_mode_cycle(os.getenv("USE_TRENDS"))
    USE_TRENDS_CYCLE               = [m == "trends" for m in USE_TRENDS_MODE_CYCLE]
    USE_TRENDS                     = USE_TRENDS_CYCLE[0]
    TREND_CANDIDATE_LIMIT          = int(os.getenv("TREND_CANDIDATE_LIMIT", "5"))
    CEFR_ROTATION                  = _parse_on_off_env("CEFR_ROTATION", default=False)
    METRICS_FETCH_PER_CYCLE        = max(0, int(os.getenv("METRICS_FETCH_PER_CYCLE", "0")))
    _raw                           = os.getenv("IMAGE_STYLE", "photographic")
    IMAGE_STYLE_CYCLE              = [s.lower().strip() for s in _raw.split(",") if s.strip()] or ["photographic"]
    IMAGE_STYLE                    = IMAGE_STYLE_CYCLE[0]
    _raw_tweet                     = os.getenv("TWEET_STYLE", "funny")
    TWEET_STYLE_CYCLE              = [s.lower().strip() for s in _raw_tweet.split(",") if s.strip()] or ["funny"]
    TWEET_STYLE                    = TWEET_STYLE_CYCLE[0]
    IMAGE_PROVIDER                 = os.getenv("IMAGE_PROVIDER", "midjourney").lower().strip()
    GENERATED_IMAGE_COUNT          = int(os.getenv("GENERATED_IMAGE_COUNT", os.getenv("GROK_IMAGE_COUNT", "1")))
    INDIVIDUAL_IMAGE_PROMPTS       = os.getenv("INDIVIDUAL_IMAGE_PROMPTS", "false").lower().strip() == "true"
    Z_IMAGE_TURBO_STEPS            = int(os.getenv("Z_IMAGE_TURBO_STEPS",  "8"))
    Z_IMAGE_TURBO_WIDTH            = int(os.getenv("Z_IMAGE_TURBO_WIDTH",  "832"))
    Z_IMAGE_TURBO_HEIGHT           = int(os.getenv("Z_IMAGE_TURBO_HEIGHT", "480"))
    Z_IMAGE_PROMPT_SUFFIX          = os.getenv("Z_IMAGE_PROMPT_SUFFIX", "").strip()
    Z_IMAGE_BASE_MODEL_ID          = os.getenv("Z_IMAGE_BASE_MODEL_ID", "Tongyi-MAI/Z-Image").strip()
    Z_IMAGE_BASE_STEPS             = int(os.getenv("Z_IMAGE_BASE_STEPS", "30"))
    Z_IMAGE_BASE_GUIDANCE_SCALE    = float(os.getenv("Z_IMAGE_BASE_GUIDANCE_SCALE", "5.0"))
    Z_IMAGE_BASE_WIDTH             = int(os.getenv("Z_IMAGE_BASE_WIDTH",  "832"))
    Z_IMAGE_BASE_HEIGHT            = int(os.getenv("Z_IMAGE_BASE_HEIGHT", "480"))
    Z_IMAGE_BASE_NEGATIVE_PROMPT   = os.getenv("Z_IMAGE_BASE_NEGATIVE_PROMPT", "").strip()
    ENABLE_INSTRUCTIR_ENHANCE      = _parse_on_off_env("ENABLE_INSTRUCTIR_ENHANCE", default=False)
    INSTRUCTIR_DIR                 = os.getenv("INSTRUCTIR_DIR", "").strip()
    _ir_prompt_raw                 = (os.getenv("INSTRUCTIR_PROMPT") or "").strip()
    INSTRUCTIR_PROMPT              = _ir_prompt_raw or _DEFAULT_INSTRUCTIR_PROMPT
    VIDEO_INTERPOLATION            = os.getenv("VIDEO_INTERPOLATION", "false").lower().strip() == "true"
    RIFE_DIR                       = os.getenv("RIFE_DIR", os.path.join(os.path.expanduser("~"), "Programming", "Practical-RIFE"))
    RIFE_PYTHON                    = os.getenv("RIFE_PYTHON", os.path.join(os.path.expanduser("~"), "Programming", "Practical-RIFE", "venv", "bin", "python"))
    VIDEO_UPLOAD_FPS               = int(os.getenv("VIDEO_UPLOAD_FPS", "32"))
    MAX_TWEET_LENGTH               = int(os.getenv("MAX_TWEET_LENGTH", "280"))
    MAX_EXAMPLE_WORDS              = int(os.getenv("MAX_EXAMPLE_WORDS", "13"))
    POST_INTERVAL_SECONDS          = int(os.getenv("POST_INTERVAL_SECONDS", "18000"))
    VIDEO_STYLE                    = os.getenv("VIDEO_STYLE", "ktv").lower().strip()
    ANALYZE_LAST_N                 = int(os.getenv("ANALYZE_LAST_N", "10"))
    METRICS_FETCH_MAX_TWEETS       = _parse_metrics_fetch_max(
        os.getenv("METRICS_FETCH_MAX_TWEETS"), ANALYZE_LAST_N
    )
    FLAG_OVERLAY                   = os.getenv("FLAG_OVERLAY", "true").lower().strip() == "true"
    ENABLE_VIDEO                   = os.getenv("ENABLE_VIDEO", "off").lower().strip()
    ENABLE_GROK_VIDEO              = ENABLE_VIDEO == "grok"
    ENABLE_KEN_BURNS               = os.getenv("ENABLE_KEN_BURNS", "false").lower().strip() == "true"
    ENABLE_BACKGROUND_MUSIC        = _parse_on_off_env("ENABLE_BACKGROUND_MUSIC", default=False)
    VIDEO_FREQUENCY                = int(os.getenv("VIDEO_FREQUENCY", os.getenv("GROK_VIDEO_FREQUENCY", "1")))
    GROK_VIDEO_FREQUENCY           = VIDEO_FREQUENCY
    WAN_VIDEO_DIR                  = os.getenv("WAN_VIDEO_DIR", str(os.path.join(os.path.expanduser("~"), "Programming", "Wan2GP")))
    WAN_VIDEO_STEPS                = int(os.getenv("WAN_VIDEO_STEPS", "10"))
    WAN_VIDEO_FRAMES               = int(os.getenv("WAN_VIDEO_FRAMES", "81"))
    WAN_VIDEO_HISTORY_FILE         = os.getenv("WAN_VIDEO_HISTORY_FILE", "data/wan_video_history.jsonl")
    VIDEO_FPS                      = int(os.getenv("WAN_VIDEO_FPS", "16"))
    ENABLE_REALESRGAN              = _parse_on_off_env("ENABLE_REALESRGAN", default=False)
    REALESRGAN_DIR                 = os.getenv("REALESRGAN_DIR", os.path.join(os.path.expanduser("~"), "Programming", "Real-ESRGAN"))
    REALESRGAN_MODEL               = os.getenv("REALESRGAN_MODEL", "RealESRGAN_x4plus")
    REALESRGAN_OUTSCALE            = float(os.getenv("REALESRGAN_OUTSCALE", "1.5"))
    REALESRGAN_TILE                = int(os.getenv("REALESRGAN_TILE", "256"))
    KTV_FONT_SIZE                  = _parse_ktv_font_size(os.getenv("KTV_FONT_SIZE"))
    ENABLE_SELF_IMPROVEMENT        = os.getenv("ENABLE_SELF_IMPROVEMENT", "false").lower().strip() == "true"
    IMPROVEMENT_INTERVAL_CYCLES    = int(os.getenv("IMPROVEMENT_INTERVAL_CYCLES", "5"))
    IMPROVEMENT_SCORE_THRESHOLD    = float(os.getenv("IMPROVEMENT_SCORE_THRESHOLD", "9999"))
    MAX_CONSECUTIVE_FAILURES       = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "5"))
    STRATEGY_METRICS_UPDATES_ENABLED, STRATEGY_UPDATE_INTERVAL_HOURS = _parse_strategy_update_interval(
        os.getenv("STRATEGY_UPDATE_INTERVAL_HOURS", "24")
    )
    TWEET_MODEL                    = os.getenv("TWEET_MODEL", "flagship").lower().strip()
    TWEET_PICKER_MODEL             = os.getenv("TWEET_PICKER_MODEL", "flagship").lower().strip()
    STRATEGY_MODEL                 = os.getenv("STRATEGY_MODEL", "reasoning").lower().strip()
    TREND_FILTER_MODEL             = os.getenv("TREND_FILTER_MODEL", "non-reasoning").lower().strip()
    WORD_PICK_MODEL                = os.getenv("WORD_PICK_MODEL", "non-reasoning").lower().strip()
    SIMILARITY_MODEL               = os.getenv("SIMILARITY_MODEL", "non-reasoning").lower().strip()
    VOICE_PICKER_MODEL             = os.getenv("VOICE_PICKER_MODEL", "non-reasoning").lower().strip()
    COMFYUI_ARGS                   = os.getenv("COMFYUI_ARGS", "--normalvram --fp16-vae").strip()

# ── Image generation provider ────────────────────────────────────────────────
# "midjourney"    = Midjourney via TTAPI (default, requires TT_API_KEY)
# "grok"          = xAI Grok Imagine API  (requires XAI_API_KEY)
# "z-image-turbo" = Z-Image-Turbo FP8 AIO via local ComfyUI (requires COMFYUI_URL/COMFYUI_DIR)
# "z-image-base"  = Z-Image base model via diffusers (local GPU, higher quality, no ComfyUI)
IMAGE_PROVIDER: str = os.getenv("IMAGE_PROVIDER", "midjourney").lower().strip()

# Number of images to generate per cycle.
# Grok: requested in a single API call. z-image-turbo: sequential runs with different seeds.
# All candidates are ranked by ImageReward and the best one is picked.
GENERATED_IMAGE_COUNT: int = int(os.getenv("GENERATED_IMAGE_COUNT", os.getenv("GROK_IMAGE_COUNT", "1")))

# When True, the LLM is called once per image to generate a unique prompt variation for each.
# Each image is scored against its own prompt; the best-scoring (prompt, image) pair wins.
# When False (default), one prompt is generated and reused for all images.
INDIVIDUAL_IMAGE_PROMPTS: bool = os.getenv("INDIVIDUAL_IMAGE_PROMPTS", "false").lower().strip() == "true"

# Denoising steps for Z-Image-Turbo (IMAGE_PROVIDER=z-image-turbo).
# Model card recommendation: 8–9. CFG/sampler/scheduler are locked in the service.
Z_IMAGE_TURBO_STEPS: int  = int(os.getenv("Z_IMAGE_TURBO_STEPS",  "8"))
# Output resolution (defaults match Wan2.1 I2V; override e.g. 3840×2160 for 4K image-only).
Z_IMAGE_TURBO_WIDTH: int  = int(os.getenv("Z_IMAGE_TURBO_WIDTH",  "832"))
Z_IMAGE_TURBO_HEIGHT: int = int(os.getenv("Z_IMAGE_TURBO_HEIGHT", "480"))

# Quality / anatomy suffix appended to every Z-Image prompt after the LLM output.
# Applies to both z-image-turbo and z-image-base. Leave empty to disable.
Z_IMAGE_PROMPT_SUFFIX: str = os.getenv("Z_IMAGE_PROMPT_SUFFIX", "").strip()

# ── Z-Image Base (diffusers, IMAGE_PROVIDER=z-image-base) ────────────────────
# HuggingFace model ID.  Change to a fine-tune if desired.
Z_IMAGE_BASE_MODEL_ID: str  = os.getenv("Z_IMAGE_BASE_MODEL_ID", "Tongyi-MAI/Z-Image").strip()
# Denoising steps (20–50 recommended; 30 is the sweet spot for quality/speed).
Z_IMAGE_BASE_STEPS: int     = int(os.getenv("Z_IMAGE_BASE_STEPS", "30"))
# CFG scale (classifier-free guidance).  3.5–7.0 works well; 5.0 is a safe default.
Z_IMAGE_BASE_GUIDANCE_SCALE: float = float(os.getenv("Z_IMAGE_BASE_GUIDANCE_SCALE", "5.0"))
# Output resolution — keep at 832×480 to match WAN2.1 480p I2V input exactly.
Z_IMAGE_BASE_WIDTH: int     = int(os.getenv("Z_IMAGE_BASE_WIDTH",  "832"))
Z_IMAGE_BASE_HEIGHT: int    = int(os.getenv("Z_IMAGE_BASE_HEIGHT", "480"))
# Optional negative prompt (base model supports full CFG unlike turbo).
Z_IMAGE_BASE_NEGATIVE_PROMPT: str = os.getenv("Z_IMAGE_BASE_NEGATIVE_PROMPT", "").strip()

# InstructIR post-pass (IMAGE_PROVIDER=z-image-turbo only): improve each candidate PNG
# before ImageReward. Requires a clone of https://github.com/mv-lab/InstructIR and
# PyYAML + InstructIR deps (see README). Increases peak GPU RAM in the bot process.
_DEFAULT_INSTRUCTIR_PROMPT = (
    "enhance overall appeal, boost contrast and saturation, "
    "improve sharpness and vibrance, make it look more professional and vibrant"
)
ENABLE_INSTRUCTIR_ENHANCE: bool = _parse_on_off_env("ENABLE_INSTRUCTIR_ENHANCE", default=False)
INSTRUCTIR_DIR: str = os.getenv("INSTRUCTIR_DIR", "").strip()
_INSTRUCTIR_PROMPT_RAW: str | None = os.getenv("INSTRUCTIR_PROMPT")
INSTRUCTIR_PROMPT: str = (
    _INSTRUCTIR_PROMPT_RAW.strip() if _INSTRUCTIR_PROMPT_RAW and _INSTRUCTIR_PROMPT_RAW.strip() else _DEFAULT_INSTRUCTIR_PROMPT
)

# When True, interpolate the generated video to VIDEO_UPLOAD_FPS with Practical-RIFE
# before uploading. RIFE_DIR must point to a fully set-up Practical-RIFE clone.
VIDEO_INTERPOLATION: bool = os.getenv("VIDEO_INTERPOLATION", "false").lower().strip() == "true"
RIFE_DIR: str             = os.getenv("RIFE_DIR", os.path.join(os.path.expanduser("~"), "Programming", "Practical-RIFE"))
# Python interpreter that runs inference_video.py — must have PyTorch with GPU support.
RIFE_PYTHON: str          = os.getenv("RIFE_PYTHON", os.path.join(os.path.expanduser("~"), "Programming", "Practical-RIFE", "venv", "bin", "python"))
# 32 = exact 2x from 16fps (cleanest). 30 = standard broadcast.
VIDEO_UPLOAD_FPS: int     = int(os.getenv("VIDEO_UPLOAD_FPS", "32"))

# ComfyUI server URL and installation directory (used by IMAGE_PROVIDER=z-image-turbo
# and the ComfyUI-based video / test scripts).
COMFYUI_URL: str          = os.getenv("COMFYUI_URL",   "http://127.0.0.1:8188").rstrip("/")
COMFYUI_DIR: str          = os.getenv("COMFYUI_DIR",   os.path.join(os.path.expanduser("~"), "ComfyUI"))
# CLI flags forwarded to ComfyUI main.py on auto-start (IMAGE_PROVIDER=z-image-turbo).
COMFYUI_ARGS: str         = os.getenv("COMFYUI_ARGS",  "--normalvram --fp16-vae")
# Seconds to wait for ComfyUI to become ready after auto-start.
COMFYUI_START_TIMEOUT: int = int(os.getenv("COMFYUI_START_TIMEOUT", "120"))

# ── Tweet constraints ────────────────────────────────────────────────────────
# Maximum character length of a posted tweet. X's hard limit is 280 for free
# accounts; Premium accounts support up to 25 000. Adjust if your account has
# an extended limit.
MAX_TWEET_LENGTH: int = int(os.getenv("MAX_TWEET_LENGTH", "280"))

# Maximum number of words allowed in the source-language example sentence.
MAX_EXAMPLE_WORDS: int = int(os.getenv("MAX_EXAMPLE_WORDS", "13"))

# ── Bot behaviour ─────────────────────────────────────────────────────────────
POST_INTERVAL_SECONDS: int = int(os.getenv("POST_INTERVAL_SECONDS", "18000"))
HISTORY_FILE: str = os.getenv("HISTORY_FILE", "data/post_history.json")
LOG_FILE: str = os.getenv("LOG_FILE", "data/bot.log")
VIDEO_STYLE: str = os.getenv("VIDEO_STYLE", "ktv").lower().strip()
ANALYZE_LAST_N: int = int(os.getenv("ANALYZE_LAST_N", "10"))
# Cap X API get_tweet calls per refresh: default max(ANALYZE_LAST_N, 30); 0 = unlimited.
METRICS_FETCH_MAX_TWEETS: int = _parse_metrics_fetch_max(
    os.getenv("METRICS_FETCH_MAX_TWEETS"), ANALYZE_LAST_N
)

# Number of most-recent tweets to refresh metrics for at the start of every
# cycle, independently of any strategy-update gate.  Deleted tweets are pruned
# from history.  Set to 0 to disable.
METRICS_FETCH_PER_CYCLE: int = max(0, int(os.getenv("METRICS_FETCH_PER_CYCLE", "0")))

# Word selection: trends (X topics), pool (AI + random theme bank), or strategy (AI + strategy topic).
# Comma-separated cycle (same index as TWEET_STYLE / IMAGE_STYLE).
#   "true,false,false,false" → same as trends,strategy,strategy,strategy
#   "trends,trends,pool,pool,trends" → mixed cycle
_USE_TRENDS_RAW: str = os.getenv("USE_TRENDS", "false")
USE_TRENDS_MODE_CYCLE: list[str] = _parse_use_trends_mode_cycle(_USE_TRENDS_RAW)
USE_TRENDS_CYCLE: list[bool] = [m == "trends" for m in USE_TRENDS_MODE_CYCLE]
USE_TRENDS: bool = USE_TRENDS_CYCLE[0]

# How many of the AI's top-ranked trend word candidates to try before falling back to
# pure AI word selection. Once the top N candidates are all already used, the bot gives
# up on trends and lets the AI pick freely instead.
TREND_CANDIDATE_LIMIT: int = int(os.getenv("TREND_CANDIDATE_LIMIT", "5"))

# When True the bot cycles through CEFR levels (A1→A2→B1→B2→C1→C2→A1→…),
# reading the last level used from post history and advancing by one each cycle.
CEFR_ROTATION: bool = _parse_on_off_env("CEFR_ROTATION", default=False)

# ── Self-Improvement Engine ────────────────────────────────────────────────────
# When True, the improvement engine runs automatically after every
# IMPROVEMENT_INTERVAL_CYCLES cycles during the wait period.
ENABLE_SELF_IMPROVEMENT: bool = os.getenv("ENABLE_SELF_IMPROVEMENT", "false").lower().strip() == "true"

# Run improvement every N cycles (default: every 5 cycles).
IMPROVEMENT_INTERVAL_CYCLES: int = int(os.getenv("IMPROVEMENT_INTERVAL_CYCLES", "5"))

# Only run improvement if the average engagement score of recent posts is below
# this threshold. Set high (e.g. 9999) to always run. Use --force to override.
IMPROVEMENT_SCORE_THRESHOLD: float = float(os.getenv("IMPROVEMENT_SCORE_THRESHOLD", "9999"))

# Stop the bot after this many consecutive failed cycles (any non-fatal error in a row).
# Prevents the bot from burning upstream API credits indefinitely when a provider is down.
# Fatal billing/auth errors (HTTP 401/402/403) always stop the bot immediately regardless.
MAX_CONSECUTIVE_FAILURES: int = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "5"))


# Which video engine to use for animating the generated image.
#   "off"    → no video animation; static KTV or Ken Burns only (default)
#   "grok"   → Grok Imagine API (requires XAI_API_KEY)
#   "WAN2.1" → local Wan2.1 model via Wan2GP (requires WAN_VIDEO_DIR)
ENABLE_VIDEO: str = os.getenv("ENABLE_VIDEO", "off").lower().strip()

# Backward-compat alias used by older code paths.
ENABLE_GROK_VIDEO: bool = ENABLE_VIDEO == "grok"

# When True, applies a slow PIL AFFINE Ken Burns zoom+pan to the still image
# on the static video path (i.e. when ENABLE_VIDEO=off or when the video gate
# skips this cycle).
ENABLE_KEN_BURNS: bool = os.getenv("ENABLE_KEN_BURNS", "false").lower().strip() == "true"

# When True, mix TTS voice with BACKGROUND_MUSIC_PATH in create_video.
# Accepts true/false or on/off. Default: off (voice-only on the video track).
ENABLE_BACKGROUND_MUSIC: bool = _parse_on_off_env("ENABLE_BACKGROUND_MUSIC", default=False)

# How often to generate a video: every Nth tweet.
#   1 = every tweet  (default)
#   2 = every 2nd tweet  …etc.
# Applies to both "grok" and "wan" engines.
VIDEO_FREQUENCY: int = int(os.getenv("VIDEO_FREQUENCY", os.getenv("GROK_VIDEO_FREQUENCY", "1")))

# Backward-compat alias.
GROK_VIDEO_FREQUENCY: int = VIDEO_FREQUENCY

# Filesystem path to the Wan2GP installation directory.
# Only used when ENABLE_VIDEO=wan.
WAN_VIDEO_DIR: str = os.getenv("WAN_VIDEO_DIR", str(os.path.join(os.path.expanduser("~"), "Programming", "Wan2GP")))

# Number of denoising steps for Wan video generation.
# Lower = faster but lower quality. Higher = slower but better quality.
WAN_VIDEO_STEPS: int = int(os.getenv("WAN_VIDEO_STEPS", "10"))

# Number of video frames (only used when ENABLE_VIDEO=WAN2.1).
# Must be 4k+1 (e.g. 49, 65, 81, 97, 121). At 16 fps: 81 ≈ 5s. At 24 fps: 121 ≈ 5s.
WAN_VIDEO_FRAMES: int = int(os.getenv("WAN_VIDEO_FRAMES", "81"))

# Frames per second for all video output — both Wan2.1 generation and MoviePy composition.
# 16 = Wan2.1 native fps (no resampling). 24 = smoother (needs more frames for same duration).
VIDEO_FPS: int = int(os.getenv("WAN_VIDEO_FPS", "16"))

# Append-only JSONL file: generation params + video reward scores per Wan run.
WAN_VIDEO_HISTORY_FILE: str = os.getenv("WAN_VIDEO_HISTORY_FILE", "data/wan_video_history.jsonl")

# ── Real-ESRGAN video upscaling (ENABLE_VIDEO=WAN2.1 only) ───────────────────
# When True, the WAN2.1 480p video is upscaled to 720p with Real-ESRGAN
# before the KTV/badge overlay is composited.  Requires a Real-ESRGAN clone
# and model weights (see services/realesrgan_upscale.py for setup instructions).
ENABLE_REALESRGAN: bool  = _parse_on_off_env("ENABLE_REALESRGAN", default=False)
# Path to the Real-ESRGAN repository root (must contain inference_realesrgan_video.py).
REALESRGAN_DIR: str      = os.getenv("REALESRGAN_DIR", os.path.join(os.path.expanduser("~"), "Programming", "Real-ESRGAN"))
# Model name (weights/<model>.pth must exist).  RealESRGAN_x4plus is the best
# all-round model; realesr-general-x4v3 is slightly faster with similar quality.
REALESRGAN_MODEL: str    = os.getenv("REALESRGAN_MODEL", "RealESRGAN_x4plus")
# Scale factor: 1.5 → 480p × 1.5 = 720p (the exact target resolution).
REALESRGAN_OUTSCALE: float = float(os.getenv("REALESRGAN_OUTSCALE", "1.5"))
# Tile size for tiled inference — keeps VRAM under 4 GB even on longer clips.
REALESRGAN_TILE: int     = int(os.getenv("REALESRGAN_TILE", "256"))

# KTV overlay subtitle font size in px at 720p (standard HD reference).
# E.g. KTV_FONT_SIZE=80 → ~80px on 720p Grok video, ~53px on 480p Wan video.
# Bar height, text box, and stroke all scale proportionally. Default: 80.
KTV_FONT_SIZE: int = _parse_ktv_font_size(os.getenv("KTV_FONT_SIZE"))

# When True, a source→target flag badge is added to the top-right corner of
# each generated image, reinforcing the language-learning branding.
FLAG_OVERLAY: bool = os.getenv("FLAG_OVERLAY", "true").lower().strip() == "true"

# Model used for strategy analysis ("reasoning" = grok-4-1-fast, "non-reasoning" = grok-4-1-fast-non-reasoning).
# Only applies when AI_PROVIDER=grok. Scaleway always uses llama-3.3-70b.
STRATEGY_MODEL: str = os.getenv("STRATEGY_MODEL", "reasoning").lower().strip()

# Model used for tweet generation:
#   "flagship"      = grok-4          (best language quality, ~$0.003/tweet)
#   "reasoning"     = grok-4-1-fast   (reasoning variant, same price as fast)
#   "non-reasoning" = grok-4-1-fast-non-reasoning  (default fast model)
# Only applies when AI_PROVIDER=grok.
TWEET_MODEL: str = os.getenv("TWEET_MODEL", "flagship").lower().strip()

# Model used to pick the best tweet from the generated candidates:
#   "flagship"      = grok-4  (default — highest judgement quality)
#   "reasoning"     = grok-4-1-fast
#   "non-reasoning" = grok-4-1-fast-non-reasoning
# Only applies when AI_PROVIDER=grok.
TWEET_PICKER_MODEL: str = os.getenv("TWEET_PICKER_MODEL", "flagship").lower().strip()

# Model used to filter trend keywords and rank them by German learning value:
#   "flagship"      = grok-4
#   "reasoning"     = grok-4-1-fast
#   "non-reasoning" = grok-4-1-fast-non-reasoning  (default — fast and cheap)
# Only applies when AI_PROVIDER=grok.
TREND_FILTER_MODEL: str = os.getenv("TREND_FILTER_MODEL", "non-reasoning").lower().strip()

# Model used for free-form word selection (when trends are off this cycle or trends yield nothing):
#   "flagship"      = grok-4
#   "reasoning"     = grok-4-1-fast
#   "non-reasoning" = grok-4-1-fast-non-reasoning  (default)
# Only applies when AI_PROVIDER=grok.
WORD_PICK_MODEL: str = os.getenv("WORD_PICK_MODEL", "non-reasoning").lower().strip()

# Model used for the semantic duplicate / similarity check:
#   "flagship"      = grok-4
#   "reasoning"     = grok-4-1-fast
#   "non-reasoning" = grok-4-1-fast-non-reasoning  (default)
# Only applies when AI_PROVIDER=grok.
SIMILARITY_MODEL: str = os.getenv("SIMILARITY_MODEL", "non-reasoning").lower().strip()

# Model used to pick the best TTS voice for the tweet:
#   "flagship"      = grok-4
#   "reasoning"     = grok-4-1-fast
#   "non-reasoning" = grok-4-1-fast-non-reasoning  (default — fast and cheap)
# Only applies when AI_PROVIDER=grok.
VOICE_PICKER_MODEL: str = os.getenv("VOICE_PICKER_MODEL", "non-reasoning").lower().strip()

# How many hours must pass before metrics are refreshed and strategy is re-analysed.
# Set STRATEGY_UPDATE_INTERVAL_HOURS=false (or off/never/disabled) to never refresh
# X metrics and never re-run strategy analysis.
STRATEGY_METRICS_UPDATES_ENABLED, STRATEGY_UPDATE_INTERVAL_HOURS = _parse_strategy_update_interval(
    os.getenv("STRATEGY_UPDATE_INTERVAL_HOURS", "24")
)

# ── Folder layout ─────────────────────────────────────────────────────────────

def _resolve_ktv_font() -> str:
    """Return the path to the first available bold TTF font on this OS."""
    if platform.system() == "Darwin":
        candidates = [
            "/Library/Fonts/Lato-Bold.ttf",                          # brew install --cask font-lato
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",          # Ubuntu default
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No KTV font found. Tried: {candidates}\n"
        "On macOS: brew install --cask font-lato\n"
        "On Ubuntu: sudo apt install fonts-lato"
    )

KTV_FONT = _resolve_ktv_font()

STRATEGY_FILE: str = "data/strategy.json"
STRATEGY_HISTORY_FILE: str = "data/strategy_history.json"

IMAGES_DIR = "Images"
VOICES_DIR = "Voices"
VOICES_MUSIC_DIR = "Voices with Background Music"
VIDEOS_DIR = "Videos"
BACKGROUND_MUSIC_PATH = "Background Music/music.mp3"
CHECKPOINT_DB = "data/checkpoints.sqlite"

for _d in (IMAGES_DIR, VOICES_DIR, VOICES_MUSIC_DIR, VIDEOS_DIR, "data"):
    os.makedirs(_d, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

class _ConsoleFormatter(logging.Formatter):
    """
    Clean, coloured console output.
    - INFO:    HH:MM:SS  message   (plain)
    - WARNING: HH:MM:SS  ⚠  message (yellow, truncated exception)
    - ERROR:   HH:MM:SS  ✖  message (red)
    Logger name is omitted — stage banners in ui.py carry context instead.
    """
    _R  = "\033[0m"
    _Y  = "\033[93m"
    _RE = "\033[91m"
    _G  = "\033[90m"   # gray for dim info lines

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()

        # Truncate very long messages (full detail stays in the file log)
        if len(msg) > 220:
            msg = msg[:220] + " …"

        if record.levelno >= logging.ERROR:
            return f"{self._RE}{ts}  ✖  {msg}{self._R}"
        if record.levelno >= logging.WARNING:
            return f"{self._Y}{ts}  ⚠  {msg}{self._R}"
        return f"{ts}     {msg}"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("lang_bot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console: clean & coloured ──────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ConsoleFormatter())

    # ── File: full detail ──────────────────────────────────────────────────
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_logging()
