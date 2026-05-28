"""Centralized logger setup.

Loguru replaces the stdlib logging usage scattered across routers. Output
goes two places:

- stdout: pretty + colorized in dev, JSON in prod (toggled by `OURKIN_ENV`)
- file:   `logs/api.jsonl` always, JSON-serialized, rotated daily, 14-day
          retention. Promtail tails this for Loki ingestion.

Use as:

    from app.log import logger
    logger.bind(event="redate", path=p, by=user).info("media redated")

The convention (enforced socially, not by code):
- `event` — short verb describing the action ("redate", "face.assigned")
- primary id of the subject (`path`, `person_id`, `cluster_id`, …)
- `by` — user email (CF Access header value) on writes

Every log line that records a state change must carry enough context that
a future Grafana query can answer who/what/where without follow-up.
"""

import os
import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(os.environ.get("OURKIN_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Clear the default sink (which is stderr) and rebuild from scratch so we
# control format + destinations.
logger.remove()

_ENV = os.environ.get("OURKIN_ENV", "dev")
_LEVEL = os.environ.get("OURKIN_LOG_LEVEL", "DEBUG" if _ENV == "dev" else "INFO")

# Every record carries `env` automatically — Grafana queries can filter
# `{env="prod"}` without touching individual log call sites.
logger.configure(extra={"env": _ENV})

if _ENV == "dev":
    # Human-readable, colored, single-line per record.
    logger.add(
        sys.stdout,
        level=_LEVEL,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> "
            "| <level>{level: <7}</level> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level> {extra}"
        ),
    )
else:
    # JSON to stdout — Docker captures, Promtail / Dozzle can read.
    logger.add(sys.stdout, level=_LEVEL, serialize=True)

# Persistent JSONL file. Always JSON-serialized so Promtail's JSON pipeline
# stage works uniformly. Daily rotation; 14-day retention.
logger.add(
    LOG_DIR / "api.jsonl",
    level=_LEVEL,
    serialize=True,
    rotation="00:00",
    retention="14 days",
    enqueue=True,  # async-safe writes from the FastAPI event loop
)
