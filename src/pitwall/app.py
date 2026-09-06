"""The pit-wall web service (F017).

Routes are thin: they parse the request, ask `queries` for rows, hand them to
`figure`, and return. Everything worth testing lives in those two modules, so
almost all of this feature is checkable without a container.

`plotly.min.js` is served from the installed package. No CDN: the global source
policy allows no unvetted executable fetched at runtime, and a pit wall has to
work with no internet.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from plotly.offline import get_plotlyjs
from psycopg_pool import ConnectionPool

from src.pitwall import figure as figure_module
from src.pitwall import queries

logger = logging.getLogger(__name__)

#: Served by this service at this path. Immutable: the file is pinned by the
#: plotly version in requirements-pitwall.txt, so a browser may keep it.
SCRIPT_PATH = "/static/plotly.min.js"
_PLOTLY_JS = get_plotlyjs()

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise HTTPException(status_code=503, detail="the warehouse connection is not open")
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """One pool for the process, opened before the first request.

    A fresh connection costs 22 ms to open and 82 ms for the overlay's first
    execution; on a reused connection the same queries take 11 ms, because
    psycopg prepares a statement after its fifth execution and the plan is then
    cached. Keeping one warm is most of the request budget.
    """
    global _pool
    _pool = queries.build_pool()
    logger.info("pitwall_pool_open min=%d max=%d", queries.POOL_MIN_SIZE, queries.POOL_MAX_SIZE)
    try:
        yield
    finally:
        _pool.close()
        _pool = None


app = FastAPI(title="Pit wall", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, bool]:
    """What the container healthcheck probes. Deliberately touches no table."""
    return {"ok": True}


@app.get(SCRIPT_PATH)
def plotly_js() -> Response:
    return Response(
        _PLOTLY_JS,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/sessions")
def api_sessions(pool: ConnectionPool = Depends(get_pool)) -> list[dict[str, Any]]:
    return [{"session_id": sid, "label": label} for sid, label in queries.sessions(pool)]


@app.get("/api/drivers")
def api_drivers(session_id: int = Query(..., ge=1),
                pool: ConnectionPool = Depends(get_pool)) -> list[str]:
    return queries.drivers(pool, session_id)


def _overlay(pool: ConnectionPool, session_id: int, driver_a: str, driver_b: str,
             lap_a: int | None, lap_b: int | None) -> queries.Overlay:
    try:
        return queries.overlay(pool, session_id, driver_a, driver_b, lap_a, lap_b)
    except queries.PitWallError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@app.get("/api/overlay.json")
def api_overlay(session_id: int = Query(..., ge=1), driver_a: str = Query(..., min_length=1),
                driver_b: str = Query(..., min_length=1),
                lap_a: int | None = None, lap_b: int | None = None,
                pool: ConnectionPool = Depends(get_pool)) -> JSONResponse:
    """The figure as JSON, and the rows behind it.

    The rows are here so a second consumer's numbers can be compared with
    Grafana's without scraping a chart -- which is how F017's cross-consumer
    equivalence criterion is checked.
    """
    overlay = _overlay(pool, session_id, driver_a, driver_b, lap_a, lap_b)
    import json

    return JSONResponse({
        "session_id": overlay.session_id,
        "lap_a": {"lap_id": overlay.lap_a.lap_id, "label": overlay.lap_a.label},
        "lap_b": {"lap_id": overlay.lap_b.lap_id, "label": overlay.lap_b.label},
        "columns": overlay.columns,
        "rows": [list(row) for row in overlay.rows],
        "figure": json.loads(figure_module.build(overlay).to_json()),
    })


@app.get("/pitwall", response_class=HTMLResponse)
def pitwall(session_id: int = Query(..., ge=1), driver_a: str = Query(..., min_length=1),
            driver_b: str = Query(..., min_length=1),
            lap_a: int | None = None, lap_b: int | None = None,
            pool: ConnectionPool = Depends(get_pool)) -> HTMLResponse:
    overlay = _overlay(pool, session_id, driver_a, driver_b, lap_a, lap_b)
    html = figure_module.to_html(overlay, SCRIPT_PATH)
    return HTMLResponse(html, headers={"X-Rows": str(len(overlay.rows))})


@app.get("/", response_class=HTMLResponse)
def index(pool: ConnectionPool = Depends(get_pool)) -> HTMLResponse:
    """A picker, so the service is usable without hand-writing a query string."""
    options = "\n".join(
        f'<option value="{sid}">{label}</option>' for sid, label in queries.sessions(pool)
    ) or '<option value="">no session loaded</option>'
    return HTMLResponse(_INDEX.replace("__SESSIONS__", options))


_INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Pit wall</title>
<style>
 body{background:#111217;color:#e8e8ea;font:15px/1.5 system-ui,sans-serif;margin:0;padding:3rem 1.5rem}
 form{max-width:44rem;margin:0 auto}
 h1{font-size:1.35rem;margin:0 0 .35rem}
 p{color:#9aa0aa;margin:0 0 2rem}
 label{display:block;margin:1.1rem 0 .3rem;color:#9aa0aa;font-size:.85rem}
 select,button{font:inherit;padding:.55rem .7rem;border-radius:6px;border:1px solid #2c2f36;background:#191b21;color:#e8e8ea;width:100%}
 button{margin-top:1.8rem;background:#2b6cb0;border-color:#2b6cb0;cursor:pointer}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
</style></head><body>
<form action="/pitwall" method="get">
 <h1>Pit wall</h1>
 <p>Two laps on one distance axis, under one crosshair.</p>
 <label for="session_id">Session</label>
 <select id="session_id" name="session_id" onchange="loadDrivers()">__SESSIONS__</select>
 <div class="pair">
  <div><label for="driver_a">Driver A</label><select id="driver_a" name="driver_a"></select></div>
  <div><label for="driver_b">Driver B</label><select id="driver_b" name="driver_b"></select></div>
 </div>
 <button type="submit">Compare fastest laps</button>
</form>
<script>
async function loadDrivers(){
  const id = document.getElementById('session_id').value;
  if(!id) return;
  const codes = await (await fetch('/api/drivers?session_id=' + id)).json();
  for(const [select, pick] of [['driver_a',0],['driver_b',1]]){
    const el = document.getElementById(select);
    el.innerHTML = codes.map(c => '<option value="'+c+'">'+c+'</option>').join('');
    el.selectedIndex = Math.min(pick, codes.length - 1);
  }
}
loadDrivers();
</script>
</body></html>"""
