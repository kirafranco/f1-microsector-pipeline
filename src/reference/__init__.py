"""Reference and dimension data from Jolpica-F1 (F012).

Season schedules with per-session start times, event and circuit metadata, and
driver and constructor identity: everything F005's dimensions need that
telemetry does not carry, and the calendar F006 will schedule from.

Two facts, both measured against the live API on 2026-09-04, shape this
package:

* **A driver's constructor comes from that round's entry, never from a team
  name.** FastF1's three-letter codes join to Jolpica's perfectly (20 of 20 at
  Suzuka), but the team strings agree only 6 times in 10 -- "Kick Sauber" is
  "Sauber" here, "Red Bull Racing" is "Red Bull". Matching on those strings
  would mislabel four constructors in ten, silently.
* **The API answers absence with success.** An unknown season returns HTTP 200
  and a total of zero rather than a 404, and a request without an explicit
  limit returns only the first 30 rows. So this client pages on the reported
  total and raises on an empty result instead of reporting a clean fetch of
  nothing.

Every response is cached under ``data/cache/jolpica`` before anything else
happens, which is what the API's own documentation asks of consumers and what
makes a re-run free and offline.
"""
