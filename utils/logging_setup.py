from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for easy analysis."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def configure_agent_logging(
    *,
    log_path: str = "data/logs/agent.log",
    level: int = logging.INFO,
    max_bytes: int = 2_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure JSON logging with rotation for agent events."""
    logger = logging.getLogger("agent")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_agent_event(
    *,
    event: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    logger = logging.getLogger("agent")
    payload = {"event": event}
    if data:
        payload.update(data)
    logger.info(message, extra={"extra": payload})
