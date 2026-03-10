import os
import platform
import logging
from dotenv import load_dotenv

# Load public configuration first, then secret keys.
# Values in .env take precedence over settings.env (override=True on .env).
load_dotenv("settings.env")
load_dotenv(override=True)   # loads .env (keys only, gitignored)

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

# ── DeepL ────────────────────────────────────────────────────────────────────
DEEPL_AUTH_KEY: str = os.getenv("DEEPL_AUTH_KEY", "")

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

# ── Bot behaviour ─────────────────────────────────────────────────────────────
POST_INTERVAL_SECONDS: int = int(os.getenv("POST_INTERVAL_SECONDS", "18000"))
HISTORY_FILE: str = os.getenv("HISTORY_FILE", "data/post_history.json")
LOG_FILE: str = os.getenv("LOG_FILE", "data/bot.log")
VIDEO_STYLE: str = os.getenv("VIDEO_STYLE", "ktv").lower().strip()
ANALYZE_LAST_N: int = int(os.getenv("ANALYZE_LAST_N", "10"))
# When True, word selection is based on real-time German trending topics.
# When False (default), the AI picks the word freely. Currently: picking the word freely.
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

# When True, the example sentence takes a funny, ironic angle to increase engagement.
# Set to "false" to use warm, neutral sentences instead.
FUNNY_MODE: bool = os.getenv("FUNNY_MODE", "false").lower().strip() == "true"

# When True, Grok Imagine animates the selected image into a short video used as the
# animated base for the KTV overlay.  Requires XAI_API_KEY.
ENABLE_GROK_VIDEO: bool = os.getenv("ENABLE_GROK_VIDEO", "false").lower().strip() == "true"

# How often to generate a Grok video: every Nth tweet.
#   1 = every tweet  (default)
#   2 = every 2nd tweet
#   3 = every 3rd tweet  …etc.
# Only used when ENABLE_GROK_VIDEO=true.
GROK_VIDEO_FREQUENCY: int = int(os.getenv("GROK_VIDEO_FREQUENCY", "1"))

# When True, a US-to-German flag overlay is added to the top-right corner of
# each generated image, reinforcing the German-for-English-speakers branding.
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
    logger = logging.getLogger("german_bot")
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
