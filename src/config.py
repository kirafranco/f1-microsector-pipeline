"""Project-wide paths.

Everything resolves from the repository root so that the module behaves the
same whether it is imported from a notebook, a test, or a container.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_ROOT: Path = PROJECT_ROOT / "data"

#: Mandatory FastF1 cache location (project CLAUDE.md, no exceptions).
FASTF1_CACHE_DIR: Path = DATA_ROOT / "cache" / "fastf1"

#: Immutable raw snapshots: data/raw/fastf1/<snapshot-date>/<session-slug>/
FASTF1_RAW_ROOT: Path = DATA_ROOT / "raw" / "fastf1"

INTERIM_ROOT: Path = DATA_ROOT / "interim"
PROCESSED_ROOT: Path = DATA_ROOT / "processed"


def session_slug(season: int, event: str, session: str) -> str:
    """Filesystem-safe identifier for one session."""
    safe_event = "".join(c if c.isalnum() else "-" for c in event).strip("-")
    safe_session = "".join(c if c.isalnum() else "-" for c in session).strip("-")
    return f"{season}_{safe_event}_{safe_session}"
