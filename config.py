import os
import platform
import logging
from dotenv import load_dotenv

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


def reload_settings() -> None:
    """
    Re-read settings.env (and .env) so any changes made between cycles take
    effect on the next cycle without restarting the bot.

    Only the 'live' behavioural settings are updated — static things like file
    paths, API keys, and the KTV font are left unchanged.
    """
    load_dotenv("settings.env", override=True)
    load_dotenv(override=True)  # .env (API keys) always wins

    global AI_PROVIDER
    global USE_TRENDS, TREND_CANDIDATE_LIMIT
    global IMAGE_STYLE_CYCLE, IMAGE_STYLE
    global TWEET_STYLE_CYCLE, TWEET_STYLE
    global IMAGE_PROVIDER, GROK_IMAGE_COUNT
    global MAX_TWEET_LENGTH, MAX_EXAMPLE_WORDS, POST_INTERVAL_SECONDS, VIDEO_STYLE, ANALYZE_LAST_N
    global FLAG_OVERLAY
    global ENABLE_VIDEO, ENABLE_GROK_VIDEO, VIDEO_FREQUENCY, GROK_VIDEO_FREQUENCY, ENABLE_KEN_BURNS, WAN_VIDEO_DIR
    global ENABLE_SELF_IMPROVEMENT, IMPROVEMENT_INTERVAL_CYCLES, IMPROVEMENT_SCORE_THRESHOLD
    global STRATEGY_UPDATE_INTERVAL_HOURS
    global TWEET_MODEL, TWEET_PICKER_MODEL, STRATEGY_MODEL
    global TREND_FILTER_MODEL, WORD_PICK_MODEL, SIMILARITY_MODEL, VOICE_PICKER_MODEL

    AI_PROVIDER                    = os.getenv("AI_PROVIDER", "grok").lower().strip()
    USE_TRENDS                     = os.getenv("USE_TRENDS", "false").lower().strip() == "true"
    TREND_CANDIDATE_LIMIT          = int(os.getenv("TREND_CANDIDATE_LIMIT", "5"))
    _raw                           = os.getenv("IMAGE_STYLE", "photographic")
    IMAGE_STYLE_CYCLE              = [s.lower().strip() for s in _raw.split(",") if s.strip()] or ["photographic"]
    IMAGE_STYLE                    = IMAGE_STYLE_CYCLE[0]
    _raw_tweet                     = os.getenv("TWEET_STYLE", "funny")
    TWEET_STYLE_CYCLE              = [s.lower().strip() for s in _raw_tweet.split(",") if s.strip()] or ["funny"]
    TWEET_STYLE                    = TWEET_STYLE_CYCLE[0]
    IMAGE_PROVIDER                 = os.getenv("IMAGE_PROVIDER", "midjourney").lower().strip()
    GROK_IMAGE_COUNT               = int(os.getenv("GROK_IMAGE_COUNT", "1"))
    MAX_TWEET_LENGTH               = int(os.getenv("MAX_TWEET_LENGTH", "280"))
    MAX_EXAMPLE_WORDS              = int(os.getenv("MAX_EXAMPLE_WORDS", "13"))
    POST_INTERVAL_SECONDS          = int(os.getenv("POST_INTERVAL_SECONDS", "18000"))
    VIDEO_STYLE                    = os.getenv("VIDEO_STYLE", "ktv").lower().strip()
    ANALYZE_LAST_N                 = int(os.getenv("ANALYZE_LAST_N", "10"))
    FLAG_OVERLAY                   = os.getenv("FLAG_OVERLAY", "true").lower().strip() == "true"
    ENABLE_VIDEO                   = os.getenv("ENABLE_VIDEO", "off").lower().strip()
    ENABLE_GROK_VIDEO              = ENABLE_VIDEO == "grok"
    ENABLE_KEN_BURNS               = os.getenv("ENABLE_KEN_BURNS", "false").lower().strip() == "true"
    VIDEO_FREQUENCY                = int(os.getenv("VIDEO_FREQUENCY", os.getenv("GROK_VIDEO_FREQUENCY", "1")))
    GROK_VIDEO_FREQUENCY           = VIDEO_FREQUENCY
    WAN_VIDEO_DIR                  = os.getenv("WAN_VIDEO_DIR", str(os.path.join(os.path.expanduser("~"), "Programming", "Wan2GP")))
    ENABLE_SELF_IMPROVEMENT        = os.getenv("ENABLE_SELF_IMPROVEMENT", "false").lower().strip() == "true"
    IMPROVEMENT_INTERVAL_CYCLES    = int(os.getenv("IMPROVEMENT_INTERVAL_CYCLES", "5"))
    IMPROVEMENT_SCORE_THRESHOLD    = float(os.getenv("IMPROVEMENT_SCORE_THRESHOLD", "9999"))
    STRATEGY_UPDATE_INTERVAL_HOURS = int(os.getenv("STRATEGY_UPDATE_INTERVAL_HOURS", "24"))
    TWEET_MODEL                    = os.getenv("TWEET_MODEL", "flagship").lower().strip()
    TWEET_PICKER_MODEL             = os.getenv("TWEET_PICKER_MODEL", "flagship").lower().strip()
    STRATEGY_MODEL                 = os.getenv("STRATEGY_MODEL", "reasoning").lower().strip()
    TREND_FILTER_MODEL             = os.getenv("TREND_FILTER_MODEL", "non-reasoning").lower().strip()
    WORD_PICK_MODEL                = os.getenv("WORD_PICK_MODEL", "non-reasoning").lower().strip()
    SIMILARITY_MODEL               = os.getenv("SIMILARITY_MODEL", "non-reasoning").lower().strip()
    VOICE_PICKER_MODEL             = os.getenv("VOICE_PICKER_MODEL", "non-reasoning").lower().strip()

# ── Image generation provider ────────────────────────────────────────────────
# "midjourney" = Midjourney via TTAPI (default, requires TT_API_KEY)
# "grok"       = xAI Grok Imagine API  (requires XAI_API_KEY)
IMAGE_PROVIDER: str = os.getenv("IMAGE_PROVIDER", "midjourney").lower().strip()

# Number of images to request per cycle when IMAGE_PROVIDER=grok.
# If Grok Imagine ignores this and always returns a fixed count, all returned
# images are still ranked and the best one is selected automatically.
GROK_IMAGE_COUNT: int = int(os.getenv("GROK_IMAGE_COUNT", "1"))

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
# When True, word selection is based on real-time trending topics (TRENDS_COUNTRY).
# When False (default), the AI picks the word freely.
USE_TRENDS: bool = os.getenv("USE_TRENDS", "false").lower().strip() == "true"

# How many of the AI's top-ranked trend word candidates to try before falling back to
# pure AI word selection. Once the top N candidates are all already used, the bot gives
# up on trends and lets the AI pick freely instead.
TREND_CANDIDATE_LIMIT: int = int(os.getenv("TREND_CANDIDATE_LIMIT", "5"))

# ── Self-Improvement Engine ────────────────────────────────────────────────────
# When True, the improvement engine runs automatically after every
# IMPROVEMENT_INTERVAL_CYCLES cycles during the wait period.
ENABLE_SELF_IMPROVEMENT: bool = os.getenv("ENABLE_SELF_IMPROVEMENT", "false").lower().strip() == "true"

# Run improvement every N cycles (default: every 5 cycles).
IMPROVEMENT_INTERVAL_CYCLES: int = int(os.getenv("IMPROVEMENT_INTERVAL_CYCLES", "5"))

# Only run improvement if the average engagement score of recent posts is below
# this threshold. Set high (e.g. 9999) to always run. Use --force to override.
IMPROVEMENT_SCORE_THRESHOLD: float = float(os.getenv("IMPROVEMENT_SCORE_THRESHOLD", "9999"))


# Which video engine to use for animating the generated image.
#   "off"  → no video animation; static KTV or Ken Burns only (default)
#   "grok" → Grok Imagine API (requires XAI_API_KEY)
#   "wan"  → local Wan2.1 model via Wan2GP (requires WAN_VIDEO_DIR)
ENABLE_VIDEO: str = os.getenv("ENABLE_VIDEO", "off").lower().strip()

# Backward-compat alias used by older code paths.
ENABLE_GROK_VIDEO: bool = ENABLE_VIDEO == "grok"

# When True, applies a slow PIL AFFINE Ken Burns zoom+pan to the still image
# on the static video path (i.e. when ENABLE_VIDEO=off or when the video gate
# skips this cycle).
ENABLE_KEN_BURNS: bool = os.getenv("ENABLE_KEN_BURNS", "false").lower().strip() == "true"

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

# Model used for free-form word selection (when USE_TRENDS=false or trends yield nothing):
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
# Both steps are skipped together when the interval has not elapsed (default: 24h).
STRATEGY_UPDATE_INTERVAL_HOURS: int = int(os.getenv("STRATEGY_UPDATE_INTERVAL_HOURS", "24"))

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
