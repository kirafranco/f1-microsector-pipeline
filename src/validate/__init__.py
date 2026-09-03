"""Ground-truth validation against FastF1's own timing data (F010).

The official timing line is not at grid 0: on Suzuka 2024 Q it sits 30 m
before the reference line's start and 20 m after its end, and every telemetry
window opens ~0.18 s after the start crossing and closes ~0.18 s before the end
crossing. This package locates the line on the aligned axis from the laps'
own timing, reconstructs lap and sector times from the grid between those
positions, and compares them with the official values -- the end-to-end test
of ingestion, alignment, resampling and the delta maths.

The residual it leaves (lap std ~0.09 s) is the source's floor: the interior
timing noise plus ~4 m of timing-versus-telemetry registration at each line
crossing. Everything downstream that compares a grid time with an official time
inherits it.
"""
