"""
config_logging — console formatter and root-logger wiring for the xbot tree.

Imported by config.setup_logging(). Kept here so the formatter can be tweaked
without scrolling past hundreds of lines of constants in config.py.
"""

import logging


class ConsoleFormatter(logging.Formatter):
    """
    Clean, coloured console output.
      INFO:    HH:MM:SS  message   (plain)
      WARNING: HH:MM:SS  ⚠  message (yellow, truncated exception)
      ERROR:   HH:MM:SS  ✖  message (red)

    Logger name is omitted — stage banners in utils/ui.py carry context instead.
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


def build_root_logger(log_file: str) -> logging.Logger:
    """
    Configure the root xbot logger (idempotent) with a coloured console handler
    at INFO and a full-detail file handler at DEBUG. Returns the configured
    logger so callers can keep a reference.
    """
    logger = logging.getLogger("xbot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ConsoleFormatter())

    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
