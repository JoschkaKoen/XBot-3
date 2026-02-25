import os
import logging
from dotenv import load_dotenv

load_dotenv()

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

# ── Bot behaviour ─────────────────────────────────────────────────────────────
POST_INTERVAL_SECONDS: int = int(os.getenv("POST_INTERVAL_SECONDS", "18000"))
HISTORY_FILE: str = os.getenv("HISTORY_FILE", "data/post_history.json")
LOG_FILE: str = os.getenv("LOG_FILE", "data/bot.log")
VIDEO_STYLE: str = os.getenv("VIDEO_STYLE", "ktv").lower().strip()
ANALYZE_LAST_N: int = int(os.getenv("ANALYZE_LAST_N", "10"))
# When True, word selection is based on real-time German trending topics.
# When False (default), the AI picks the word freely. Currently: picking the word freely.
USE_TRENDS: bool = os.getenv("USE_TRENDS", "false").lower().strip() == "true"

# When True, the example sentence takes a funny, ironic angle to increase engagement.
# Set to "false" to use warm, neutral sentences instead.
FUNNY_MODE: bool = os.getenv("FUNNY_MODE", "false").lower().strip() == "true"

# Model used for strategy analysis ("reasoning" = grok-4-1-fast, "non-reasoning" = grok-4-1-fast-non-reasoning).
# Only applies when AI_PROVIDER=grok. Scaleway always uses llama-3.3-70b.
STRATEGY_MODEL: str = os.getenv("STRATEGY_MODEL", "reasoning").lower().strip()

# Model used for German example sentence generation:
#   "flagship"      = grok-4          (best language quality, ~$0.003/sentence)
#   "reasoning"     = grok-4-1-fast   (reasoning variant, same price as fast)
#   "non-reasoning" = grok-4-1-fast-non-reasoning  (default fast model)
# Only applies when AI_PROVIDER=grok.
SENTENCE_MODEL: str = os.getenv("SENTENCE_MODEL", "flagship").lower().strip()

# ── Folder layout ─────────────────────────────────────────────────────────────
KTV_FONT = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"

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
