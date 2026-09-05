"""Turning the processed artefacts into a defended result (F016).

The pipeline's output is a pile of parquet. This package answers one question
with it -- where did the gap between two laps go? -- and, just as importantly,
says when the answer is "the data cannot tell you".

Measured on Suzuka 2024 Qualifying, the pole gap of 0.066 s between VER and PER
does not decompose: no corner phase and no corner event separates from zero by
more than 1.6 times its own within-driver spread. That is not a failure of the
pipeline, it is the resolution of 4 Hz public telemetry cut into corner phases,
and this package is built to state it rather than hide it behind a bar chart.

What does survive is repetition. Comparing every lap of one driver against
every lap of the other turns a single noisy measurement into a sign test: PER
is slower through T5 on all sixteen VER-PER pairings, with a speed deficit in
the raw traces to corroborate it. Consistency across pairings is the claim this
package is designed to support; a single lap-pair delta is not.

Nothing here computes a new metric. Every number comes from F004, F009 and
F010 outputs, and every figure is rendered from those same frames by code, so
the document in docs/findings/ can be rebuilt from data/ at any time.
"""
