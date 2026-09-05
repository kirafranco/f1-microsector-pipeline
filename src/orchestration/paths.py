"""Where one session's artefacts live, decided in one place.

Every stage already defaults its output to a root derived from its input's
directory name, which works but leaves the naming convention implicit: the
Suzuka artefacts on disk are all `2024_Japan_Q_projection`, yet nothing in
`src/` ever wrote that suffix. It was passed by hand on the command line
during F008 and F003 and never became code. A DAG cannot pass things by hand,
so it becomes code here.

The alignment method is part of the name because two methods can be run over
the same snapshot and their outputs must not overwrite each other -- D5 chose
projection, and `data/interim/aligned/` still holds an `_anchors` root from
the comparison that settled it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.align.session import DEFAULT_METHOD
from src.config import FASTF1_RAW_ROOT, INTERIM_ROOT, PROCESSED_ROOT, session_slug


class RunPathError(ValueError):
    """The run is not identified well enough to name its artefacts."""


@dataclass(frozen=True)
class SessionRun:
    """One session through the pipeline, and every path it touches.

    `event` is the event name as the schedule spells it ("Japanese Grand
    Prix"), not the country: three 2024 rounds are in the United States and two
    in Italy, so a country would collide. FastF1 accepts either.
    """

    season: int
    event: str
    session: str
    method: str = DEFAULT_METHOD

    def __post_init__(self) -> None:
        if not str(self.event).strip():
            raise RunPathError("event is empty; the schedule's event_name is expected")
        if not str(self.session).strip():
            raise RunPathError("session is empty; a code such as 'Q' or 'R' is expected")
        if int(self.season) < 1950:
            raise RunPathError(f"season {self.season} is before Formula 1 existed")

    @property
    def snapshot_slug(self) -> str:
        """Names the raw snapshot. Owned by F002, reproduced by asking it."""
        return session_slug(self.season, self.event, self.session)

    @property
    def slug(self) -> str:
        """Names every derived artefact: the snapshot slug plus the method."""
        return f"{self.snapshot_slug}_{self.method}"

    def snapshot_root(self, snapshot_date: date | str) -> Path:
        """The immutable raw snapshot, which is dated and never overwritten."""
        stamp = snapshot_date.isoformat() if isinstance(snapshot_date, date) else str(snapshot_date)
        return FASTF1_RAW_ROOT / stamp / self.snapshot_slug

    @property
    def aligned_root(self) -> Path:
        return INTERIM_ROOT / "aligned" / self.slug

    @property
    def grid_root(self) -> Path:
        return INTERIM_ROOT / "grid" / self.slug

    @property
    def microsector_root(self) -> Path:
        return INTERIM_ROOT / "microsectors" / self.slug

    @property
    def processed_root(self) -> Path:
        return PROCESSED_ROOT / self.slug

    @property
    def label(self) -> str:
        return f"{self.season} {self.event} {self.session}"

    def to_dict(self) -> dict[str, str | int]:
        return {"season": int(self.season), "event": str(self.event),
                "session": str(self.session), "method": str(self.method)}

    @classmethod
    def from_params(cls, params: dict) -> "SessionRun":
        """Build from a DAG run's params, failing loudly on a missing one.

        Airflow hands params through as whatever JSON they were triggered with,
        so this is where a typo becomes an error message instead of a path.
        """
        missing = [key for key in ("season", "event", "session") if not params.get(key)]
        if missing:
            raise RunPathError(f"dag run params missing {missing}")
        return cls(season=int(params["season"]), event=str(params["event"]),
                   session=str(params["session"]),
                   method=str(params.get("method") or DEFAULT_METHOD))


def latest_snapshot_root(run: SessionRun, root: Path | None = None) -> Path | None:
    """The most recent dated snapshot for this session, if one exists.

    Snapshots are immutable and dated, so a re-run on a later day would ingest
    again rather than reuse. Downstream stages want whatever is already there.
    """
    root = root or FASTF1_RAW_ROOT
    if not root.exists():
        return None
    candidates = sorted(
        (day / run.snapshot_slug for day in root.iterdir() if day.is_dir()),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate
    return None
