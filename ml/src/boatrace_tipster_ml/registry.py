"""Model registry: production model selection and dev model counter.

Two pieces of state, separate concerns, separate files:

- ``models/active.json``: ``{"model": "<name>"}`` — the production model the
  runner / predict / backtests resolve to. Switch by editing this file.
- ``models/.run-counter``: a single integer — the next dev-model prefix to
  hand out. ``next_prefix()`` reads it, advances it, returns the prefix
  string. The integer→prefix mapping is fixed-2-letter base-26
  (0=aa, 1=ab, ..., 25=az, 26=ba, ..., 675=zz).
"""

from __future__ import annotations

import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ACTIVE_PATH = MODELS_DIR / "active.json"
COUNTER_PATH = MODELS_DIR / ".run-counter"


def get_active_model_name() -> str:
    """Return the active production model name (e.g., 'p2_v2')."""
    if not ACTIVE_PATH.exists():
        raise RuntimeError(f"Missing active model config: {ACTIVE_PATH}")
    data = json.loads(ACTIVE_PATH.read_text())
    name = data.get("model")
    if not name:
        raise RuntimeError(f"active.json missing 'model' field: {ACTIVE_PATH}")
    return name


def get_active_model_dir() -> str:
    """Return the active production model directory as 'models/<name>'."""
    return f"models/{get_active_model_name()}"


def _to_prefix(n: int) -> str:
    """Encode an integer as a fixed-2-letter base-26 prefix.

    0 -> aa, 1 -> ab, ..., 25 -> az, 26 -> ba, ..., 675 -> zz.
    """
    if n < 0 or n >= 26 * 26:
        raise ValueError(f"Run counter out of range: {n}")
    return chr(ord("a") + n // 26) + chr(ord("a") + n % 26)


def _read_counter() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return int(COUNTER_PATH.read_text().strip())


def peek_prefix() -> str:
    """Return the next prefix without advancing the counter."""
    return _to_prefix(_read_counter())


def next_prefix() -> str:
    """Return the next prefix and advance the counter atomically."""
    count = _read_counter()
    prefix = _to_prefix(count)
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    COUNTER_PATH.write_text(f"{count + 1}\n")
    return prefix
