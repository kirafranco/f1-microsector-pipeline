"""Which sessions still need running, hourly (F006).

There is no push event at the end of a session (D4), so the calendar is the
trigger: take the season schedule, work out which qualifying and race sessions
should have published by now, subtract the ones the warehouse already holds,
and start a pipeline run for what is left.

Per-session fault tolerance is a property of asking the question this way. A
run that failed is simply still missing from `dim_session` an hour later, so it
is offered again without anything having to remember that it failed.
"""

from __future__ import annotations

import logging

import pendulum
from airflow.sdk import Param, dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

from src.orchestration import calendar, warehouse
from src.orchestration.calendar import DEFAULT_CODES
from src.reference.session import load_reference
from src.warehouse.connection import Settings, connect

logger = logging.getLogger(__name__)

PIPELINE_DAG_ID = "f1_session_pipeline"


@dag(
    dag_id="f1_calendar_dispatch",
    description="Trigger a pipeline run for every session that is due and not yet loaded",
    schedule="@hourly",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    # Deliberately paused on arrival, against the container-wide default: this
    # is the only DAG here that starts work on its own, and an unattended
    # laptop should not begin ingesting a season because the stack came up.
    # Unpause it when a backfill is wanted.
    is_paused_upon_creation=True,
    tags=["f1", "calendar"],
    params={
        "season": Param(2024, type="integer"),
        "codes": Param(list(DEFAULT_CODES), type="array",
                       description="Session codes to keep up to date (D10: Q and R)"),
        "max_triggers": Param(2, type="integer", minimum=0,
                              description="Runs to start per dispatch; a season backfill is 48 sessions"),
        "dry_run": Param(False, type="boolean",
                         description="List what would be triggered and trigger nothing"),
    },
)
def f1_calendar_dispatch():
    @task()
    def outstanding(**context) -> list[dict]:
        """Sessions that are due, minus the ones already in the warehouse."""
        params = context["params"]
        season = int(params["season"])
        reference = load_reference(season)

        with connect(Settings.from_env()) as connection:
            loaded = warehouse.loaded_sessions(connection)

        due = calendar.sessions_due(
            reference["dim_session_schedule"], reference["dim_event"],
            now=pendulum.now("UTC"), codes=tuple(params["codes"]),
        )
        missing = calendar.not_yet_loaded(due, loaded)

        limit = int(params["max_triggers"])
        selected = missing.head(limit) if limit >= 0 else missing
        # Exactly the kwargs TriggerDagRunOperator takes: expand_kwargs passes
        # each dict straight into the operator, so an extra key is a failure.
        payload = [
            {"trigger_run_id": calendar.run_id(row), "conf": calendar.run_params(row)}
            for _, row in selected.iterrows()
        ]
        logger.info("dispatch_selected season=%d due=%d loaded=%d outstanding=%d selected=%d dry_run=%s",
                    season, len(due), len(loaded), len(missing), len(payload), params["dry_run"])
        for (_, row), item in zip(selected.iterrows(), payload):
            logger.info("dispatch_candidate run_id=%s conf=%s starts=%s",
                        item["trigger_run_id"], item["conf"],
                        row["session_start_utc"].isoformat())
        return [] if params["dry_run"] else payload

    TriggerDagRunOperator.partial(
        task_id="trigger_pipeline",
        trigger_dag_id=PIPELINE_DAG_ID,
        # Do not block a scheduler slot for the hours a session takes, and do
        # not reset a run that is already going: a duplicate trigger for a live
        # session must be refused, which is what the stable run id is for.
        wait_for_completion=False,
        reset_dag_run=False,
        skip_when_already_exists=True,
    ).expand_kwargs(outstanding())


f1_calendar_dispatch()
