"""Delta-t curves, micro-sector times and corner metrics (F004).

Every time curve is re-zeroed at grid 0 -- the point where the lap reaches the
F008 reference line -- rather than at the first telemetry sample: FastF1 opens
the telemetry window up to 0.36 s before the line, and that offset would
otherwise leak into every delta. The reference lap is a parameter (D7):
session fastest by default, a nominated lap, or each driver's own best.

Micro-sector times telescope over the F009 partition, so per-sector deltas sum
to the delta-t curve exactly. Sectors one or two bins long are timed across
source gaps of 7-80 m and are individually noisy; ``length_m`` travels with
every row so consumers can filter. Braking points carry ``brake_gap_m`` -- the
source spacing at that point -- as their D6 uncertainty.
"""
