"""Cache-first client for the Jolpica-F1 API.

The API documents 4 requests per second and 500 per hour for unauthenticated
use, returns no rate-limit headers, and asks consumers to cache first. So this
client caches every response by request URL, paces live calls itself, pages on
the reported total, and retries transient failures with the same backoff F002
uses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

import requests

from src.config import DATA_ROOT
from src.ingest.retry import PermanentIngestError, retry_with_backoff

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"

#: Cache root (global 4.7): every external fetch goes through here.
JOLPICA_CACHE_DIR: Path = DATA_ROOT / "cache" / "jolpica"

#: Immutable snapshots of what the API returned, by date.
JOLPICA_RAW_ROOT: Path = DATA_ROOT / "raw" / "jolpica"

#: Documented burst limit is 4 requests/second; a quarter of the budget is
#: plenty for a few dozen calls and leaves room for anyone sharing the IP.
DEFAULT_MIN_INTERVAL_S = 0.25

#: The API's own default is 30 rows, silently. Always ask for a full page.
PAGE_SIZE = 100

#: A hard stop, so a wrong total can never spin forever.
MAX_PAGES = 50

USER_AGENT = "f1-microsector-pipeline/0.1 (personal research project; https://github.com/kirafranco/f1-microsector-pipeline)"


class JolpicaError(RuntimeError):
    """The API could not be used as expected."""


class JolpicaEmptyError(JolpicaError, PermanentIngestError):
    """The API answered successfully with no rows.

    This is not a transport failure and retrying will not change it: the
    request asked for something that does not exist. It is raised rather than
    returned so a typo in a season never looks like a clean fetch of nothing.
    """


def cache_key(path: str, params: Mapping[str, Any]) -> str:
    """Stable key for a request, independent of parameter order."""
    ordered = urlencode(sorted((k, str(v)) for k, v in params.items()))
    return hashlib.sha256(f"{path}?{ordered}".encode("utf-8")).hexdigest()[:32]


@dataclass
class JolpicaClient:
    """Reads Jolpica-F1, from the cache whenever possible."""

    cache_dir: Path = JOLPICA_CACHE_DIR
    base_url: str = BASE_URL
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S
    user_agent: str = USER_AGENT
    timeout_s: float = 30.0
    session: Any = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _last_request_at: float | None = field(default=None, init=False, repr=False)
    requests_made: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": self.user_agent})

    # --- cache ---------------------------------------------------------------

    def _cache_path(self, path: str, params: Mapping[str, Any]) -> Path:
        return self.cache_dir / f"{cache_key(path, params)}.json"

    def _read_cache(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["payload"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("jolpica_cache_unreadable path=%s error=%s", path.name, exc)
            return None

    def _write_cache(self, path: Path, url: str, payload: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        document = {"url": url, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "payload": payload}
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(document), encoding="utf-8")
        temporary.replace(path)

    # --- network -------------------------------------------------------------

    def _pace(self) -> None:
        """Keep a minimum gap between live requests; the API has no headers to react to."""
        if self._last_request_at is None:
            return
        waited = self.clock() - self._last_request_at
        if waited < self.min_interval_s:
            self.sleep(self.min_interval_s - waited)

    def _fetch(self, url: str, params: Mapping[str, Any]) -> dict:
        def once() -> dict:
            self._pace()
            response = self.session.get(url, params=dict(params), headers={"User-Agent": self.user_agent},
                                        timeout=self.timeout_s)
            # Counted before the status is judged: a 429 is a request the API
            # already charged us for, and the pacing budget must reflect it.
            self._last_request_at = self.clock()
            self.requests_made += 1
            response.raise_for_status()
            return response.json()

        return retry_with_backoff(once, description=f"jolpica {url}", sleep=self.sleep)

    # --- public --------------------------------------------------------------

    def get(self, path: str, *, refresh: bool = False, **params: Any) -> dict:
        """One payload for ``path``, from the cache unless ``refresh``."""
        path = path.strip("/")
        cache_path = self._cache_path(path, params)
        if not refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                self.cache_hits += 1
                logger.debug("jolpica_cache_hit path=%s params=%s", path, dict(params))
                return cached

        url = f"{self.base_url}/{path}.json"
        payload = self._fetch(url, params)
        if "MRData" not in payload:
            raise JolpicaError(f"{url}: response has no MRData envelope")
        self._write_cache(cache_path, url, payload)
        logger.info("jolpica_fetched path=%s params=%s total=%s", path, dict(params),
                    payload["MRData"].get("total"))
        return payload

    def get_all(self, path: str, table_key: str, list_key: str, *, nested_key: str | None = None,
                refresh: bool = False, **params: Any) -> list[dict]:
        """Every row for ``path``, following the API's own paging.

        The default page is 30 rows and nothing in the body says so, so this
        asks for a full page and continues until the reported total is covered.

        ``total`` does not always count the returned list. For a flat table
        (drivers, races) it does. For qualifying and results the list is Races
        and ``total`` counts the *entries inside them*, so paging a whole
        season would return the same race repeatedly with different slices of
        its results. ``nested_key`` names that inner list: pages are then
        merged by round and progress is measured in entries.

        An empty total raises: absence is not a successful fetch.
        """
        flat: list[dict] = []
        merged: dict[tuple, dict] = {}
        offset = 0
        total: int | None = None
        fetched = 0

        for _ in range(MAX_PAGES):
            payload = self.get(path, refresh=refresh, limit=PAGE_SIZE, offset=offset, **params)
            data = payload["MRData"]
            if total is None:
                total = int(data.get("total", 0))
                if total == 0:
                    raise JolpicaEmptyError(f"{path} returned no rows (params={dict(params)})")
            table = data.get(table_key)
            if table is None:
                raise JolpicaError(f"{path}: response has no {table_key}")
            page = table.get(list_key)
            if page is None:
                raise JolpicaError(f"{path}: {table_key} has no {list_key}")

            if nested_key is None:
                flat.extend(page)
                fetched = len(flat)
            else:
                for item in page:
                    key = (item.get("season"), item.get("round"))
                    known = merged.get(key)
                    if known is None:
                        merged[key] = dict(item)
                        merged[key][nested_key] = list(item.get(nested_key) or [])
                    else:
                        known[nested_key].extend(item.get(nested_key) or [])
                fetched = sum(len(item.get(nested_key) or []) for item in merged.values())

            offset += PAGE_SIZE
            if fetched >= total or not page:
                break
        else:
            raise JolpicaError(f"{path}: more than {MAX_PAGES} pages; refusing to continue")

        rows = list(merged.values()) if nested_key is not None else flat
        if total is not None and fetched != total:
            logger.warning("jolpica_row_count path=%s expected=%d got=%d", path, total, fetched)
        return rows

    # --- convenience: the five payloads this project needs -------------------

    def races(self, season: int, **kwargs: Any) -> list[dict]:
        return self.get_all(f"{season}/races", "RaceTable", "Races", **kwargs)

    def drivers(self, season: int, **kwargs: Any) -> list[dict]:
        return self.get_all(f"{season}/drivers", "DriverTable", "Drivers", **kwargs)

    def constructors(self, season: int, **kwargs: Any) -> list[dict]:
        return self.get_all(f"{season}/constructors", "ConstructorTable", "Constructors", **kwargs)

    def qualifying(self, season: int, round_number: int, **kwargs: Any) -> list[dict]:
        return self.get_all(f"{season}/{round_number}/qualifying", "RaceTable", "Races",
                            nested_key="QualifyingResults", **kwargs)

    def results(self, season: int, round_number: int, **kwargs: Any) -> list[dict]:
        return self.get_all(f"{season}/{round_number}/results", "RaceTable", "Races",
                            nested_key="Results", **kwargs)
