"""Postgres star schema and loader (F005).

Two fact grains so the dashboard can draw an overlay from the grid table and
aggregate by compound or stint from the micro-sector table without scanning
millions of rows, plus the dimensions that make either filterable.

Three properties are structural rather than checked:

* **Nothing exists for one consumer** (decision D1). No view, no function, no
  denormalised column. Grafana, Superset and a FastAPI service read the same
  tables, and the column names are the parquet contracts' names.
* **A reload cannot duplicate a row.** Each fact is partitioned by event and
  then by session, so loading a session again truncates its own leaf and
  copies into it. There is no DELETE to get wrong.
* **A session is all there or not there.** The F011 gate runs before the first
  insert and the whole load is one transaction.

Measured on 2026-09-04 against the F001 stack: COPY at ~52,000 rows/s on the
Windows bind mount, 119 B/row all-in, 1.41 GiB projected for the whole 2024
qualifying-and-race scope, overlay queries in 5 ms.
"""
