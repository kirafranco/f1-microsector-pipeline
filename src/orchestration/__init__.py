"""Everything the DAGs decide, kept out of the DAGs (F006).

Airflow is a scheduler. The choices it makes here -- which sessions are due,
whether a session's data has been published yet, how long to wait before
asking again, where a run's artefacts live -- are not scheduling, and none of
them needs a scheduler to be tested. They live in this package, with the DAG
files in servicios/airflow/dags/ reduced to wiring.

That split is what makes the orchestration testable in the conda env, where
Airflow is not installed and has no business being: its dependency tree is
about a hundred and fifty packages, and importing two DAG files locally is not
worth that. DAG integrity is asserted inside the container, which is where the
DAGs actually run.

The one piece of real cleverness here is `availability`. Decision D4 recorded
that there is no push event at the end of a session -- data appears on the
backend some tens of minutes later -- so orchestration polls. What the decision
could not have known is that FastF1 makes "not published yet" look like a
permanent failure: it swallows the API's own SessionNotAvailableError into a
warning, and the missing laps then surface as DataNotLoadedError, which this
project's retry helper correctly classes as not worth retrying. Probing for
availability, and translating those exceptions back into "ask again later", is
the whole reason a sensor exists rather than a bare retry count.
"""
