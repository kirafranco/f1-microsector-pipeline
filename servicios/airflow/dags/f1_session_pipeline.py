"""One session, end to end (F006).

Wiring only. Every decision this DAG appears to make lives in
`src.orchestration`, which is importable and tested without Airflow; what is
here is the shape of the graph, the retry policy, and the sensor.

Triggered rather than scheduled: `f1_calendar_dispatch` creates the runs, and a
person can trigger one by hand from the UI with the same three parameters.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, task
from airflow.sdk.bases.sensor import PokeReturnValue

from src.orchestration import stages
from src.orchestration.availability import (
    SENSOR_TIMEOUT_S,
    backoff_interval_s,
    probe,
)
from src.orchestration.paths import SessionRun

logger = logging.getLogger(__name__)

#: One session's pandas work at a time. The container is capped at 4 GB (D9)
#: and a session is ~600k grid rows; two concurrent runs would race for it.
MAX_ACTIVE_RUNS = 1


@dag(
    dag_id="f1_session_pipeline",
    description="Ingest, align, resample, segment, measure, validate and load one session",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=["f1", "pipeline"],
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2),
                  "retry_exponential_backoff": True, "max_retry_delay": timedelta(minutes=20)},
    params={
        "season": Param(2024, type="integer", description="Championship season"),
        "event": Param("Japanese Grand Prix", type="string",
                       description="Event name as the Jolpica schedule spells it"),
        "session": Param("Q", type="string", enum=["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"],
                         description="Session code"),
        "snapshot_date": Param(None, type=["null", "string"],
                               description="Reuse a dated snapshot instead of today's"),
    },
)
def f1_session_pipeline():
    @task.sensor(poke_interval=60, timeout=SENSOR_TIMEOUT_S, mode="reschedule")
    def wait_for_data(**context) -> PokeReturnValue:
        """Poll until the timing backend has published the session (D4).

        `mode="reschedule"` frees the worker slot between pokes, which matters
        when the wait is hours and the executor is local. The interval widens
        with each attempt rather than hammering a backend that has already said
        no.
        """
        run = SessionRun.from_params(context["params"])
        attempt = int(context["ti"].try_number)
        result = probe(run.season, run.event, run.session)

        if result.ready:
            return PokeReturnValue(is_done=True, xcom_value=result.to_dict())
        if not result.verdict.should_wait:
            raise RuntimeError(f"{run.label}: {result.detail}")

        wait_s = backoff_interval_s(attempt)
        context["ti"].task.poke_interval = wait_s
        logger.info("sensor_waiting %s verdict=%s next_poke_s=%d detail=%s",
                    run.label, result.verdict.value, wait_s, result.detail)
        return PokeReturnValue(is_done=False)

    @task()
    def ingest(**context) -> dict:
        params = context["params"]
        return stages.ingest(SessionRun.from_params(params), params.get("snapshot_date"))

    @task()
    def align(ingested: dict, **context) -> dict:
        return stages.align(SessionRun.from_params(context["params"]), ingested["snapshot_date"])

    @task()
    def grid(aligned: dict, **context) -> dict:
        return stages.grid(SessionRun.from_params(context["params"]))

    @task()
    def segment(gridded: dict, ingested: dict, **context) -> dict:
        return stages.segment(SessionRun.from_params(context["params"]), ingested["snapshot_date"])

    @task()
    def metrics(segmented: dict, ingested: dict, **context) -> dict:
        return stages.metrics(SessionRun.from_params(context["params"]), ingested["snapshot_date"])

    @task()
    def validate(measured: dict, ingested: dict, **context) -> dict:
        return stages.validate(SessionRun.from_params(context["params"]), ingested["snapshot_date"])

    @task(retries=0)
    def quality(validated: dict, ingested: dict, **context) -> dict:
        """The gate. Failing here means nothing is written to the warehouse.

        No retries: a contract failure is a deterministic property of the data
        that was just written, so a second attempt reaches the same verdict.
        Retrying cost twelve minutes per failing session during the F015
        backfill and changed nothing.
        """
        return stages.quality(SessionRun.from_params(context["params"]), ingested["snapshot_date"])

    @task(retries=1)
    def load(checked: dict, ingested: dict, **context) -> dict:
        return stages.load(SessionRun.from_params(context["params"]), ingested["snapshot_date"])

    @task()
    def summarise(loaded: dict, **context) -> dict:
        """Refresh the per-circuit validation table over everything loaded."""
        return stages.summarise(SessionRun.from_params(context["params"]))

    available = wait_for_data()
    ingested = ingest()
    aligned = align(ingested)
    gridded = grid(aligned)
    segmented = segment(gridded, ingested)
    measured = metrics(segmented, ingested)
    validated = validate(measured, ingested)
    checked = quality(validated, ingested)
    loaded = load(checked, ingested)
    summarised = summarise(loaded)

    available >> ingested
    return summarised


f1_session_pipeline()
