"""File-based logging with rotation for EgoDaO."""
import logging
import os
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(os.getcwd()) / "logs"
_KEEP_LOGS = 10


def _ensure_log_dir():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_old_logs():
    """Keep only the most recent N log files."""
    files = sorted(_LOG_DIR.glob("egodao_*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[_KEEP_LOGS:]:
        try:
            f.unlink()
        except OSError:
            pass


def setup_logging(name: str = "egodao") -> logging.Logger:
    _ensure_log_dir()
    _cleanup_old_logs()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOG_DIR / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler — everything
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    # Console handler — WARNING and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    logger.info(f"Logging started → {log_path}")
    return logger
