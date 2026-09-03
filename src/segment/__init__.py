"""Corner detection and micro-sector segmentation (F009).

Partitions the shared distance axis of one session into corner phases --
braking, entry, apex, exit -- and straights, plus fixed 100 m bins (D8), so
that micro-sector *i* is the same stretch of track on every lap.

The unit of segmentation is the *speed trough* of the session-median speed
trace, not the numbered corner: chained corners (Suzuka's esses, Degner, the
Casio chicane) share one deceleration, and corners that cost less than the
prominence threshold are straights with a heading change. Circuit-info corners
label events; they do not define them.

Boundaries are session medians at grid resolution. Per D6 every one of them is
known to roughly +/-20 m, and a driver who brakes later than the field will
have part of their braking inside the shared ``straight`` sector -- per-lap
metrics (F004) are computed inside event windows, never by reading a grid
point's phase label.
"""
