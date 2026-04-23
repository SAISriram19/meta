"""Minimal .env loader — no extra dep.

If `.env` exists at the repo root, load any KEY=value lines into os.environ
without clobbering values already set externally. Called once on import of
the `env` package.

This means users can just `cp .env.example .env` and start running scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str | None = None, override: bool = False) -> int:
    """Load KEY=value lines from `.env`. Returns number of vars set."""
    if path is None:
        # Walk up from this file to find repo root (where .env would sit).
        here = Path(__file__).resolve()
        for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
            candidate = parent / ".env"
            if candidate.exists():
                path = candidate
                break
        else:
            return 0
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if not k:
                continue
            if override or k not in os.environ:
                if v:  # don't set empty strings — they suppress provider fallbacks
                    os.environ[k] = v
                    n += 1
    return n


# Auto-load on first import. Silent if .env absent.
_loaded = load_dotenv()
