"""Spatial resampling onto a uniform distance grid (F003).

Takes F008's track-referenced laps and puts each one on the same 10 m grid, so
that grid index *i* is the same physical place for every driver and every lap.

Known limitation, inherent to the source and documented rather than fixed: car
data arrives at roughly 4 Hz, median 7.2 m between samples but with a wide tail
(p95 ~20 m, occasional dropouts of 80 m). About one 10 m bin in five contains no
source sample, so roughly one grid point in five is pure interpolation. The
``source_gap_m`` column carries that information per point; downstream consumers
that imply point-wise precision must read it.
"""
