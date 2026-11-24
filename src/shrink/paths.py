"""Where shrink keeps its files.

* :func:`cache_dir`: downloaded data (tokenizer vocabularies, models). Not the
  system temp directory: the OS may clear it, and then the next run would need
  the network again. ``SHRINK_CACHE_DIR`` overrides it.
* :func:`state_dir`: small records shrink writes itself (the ``shrink gain``
  log). ``SHRINK_STATE_DIR`` overrides it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cache_dir() -> Path:
    override = os.environ.get("SHRINK_CACHE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "shrink" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "shrink"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "shrink"


def state_dir() -> Path:
    override = os.environ.get("SHRINK_STATE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "shrink" / "state"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "shrink"
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "shrink"
