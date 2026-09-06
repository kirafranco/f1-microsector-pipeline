"""The pit-wall service: a second read-only consumer of the star schema (F017).

Additive, never a migration (D1). Grafana keeps every panel it has; this exists
for the one interaction Grafana cannot draw -- two laps stacked on one distance
axis under a single crosshair -- and, in doing so, proves D1's constraint on
F005: a schema with no dashboard-specific column, view or naming serves a
second, different consumer with no change at all.
"""
