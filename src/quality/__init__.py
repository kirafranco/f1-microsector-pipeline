"""Data quality gate (F011).

Declarative contracts for every artefact the pipeline produces, and an engine
that applies them: completeness on critical columns, uniqueness on business
keys, referential integrity between layers, physical ranges, and the
structural invariants F003, F004 and F009 established.

Two principles, both from the profile this feature was specified against:

* **A null needs a reason, not a relaxed rule.** Lap times are absent only on
  laps FastF1 marks inaccurate; brake columns only where the corner has no
  braking. Those cases are ``unless`` predicates on a strict rule, so a null
  appearing anywhere else is still an error.
* **Every rule runs before anything is judged.** A failure never hides the
  next finding, and a batch is never abandoned half-checked -- the gate is a
  decision taken on the finished report.

Ranges are physical envelopes, not statistics: they catch unit mistakes and
corrupted rows, not slow drivers. Distribution drift belongs to F015.
"""
