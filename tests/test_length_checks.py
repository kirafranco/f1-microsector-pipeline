"""Criterion 2, split: scale error vs racing-line geometry."""

from __future__ import annotations

import pytest

from src.align.session import LengthChecks

OFFICIAL = 5807.0


def _checks(aligned: float, driven: float = 5753.9) -> LengthChecks:
    return LengthChecks(aligned_m=aligned, driven_m=driven, official_m=OFFICIAL)


class TestScaleCheck:
    def test_the_measured_suzuka_case_passes(self) -> None:
        """Reference-line arc 5722.2 m against speed-integrated 5753.9 m."""
        checks = _checks(5722.2)
        assert checks.scale_error_pct == pytest.approx(0.551, abs=0.01)
        assert checks.scale_ok

    def test_a_wrong_metre_is_caught(self) -> None:
        """A 5% scale error is exactly what this half exists to find."""
        assert not _checks(5753.9 * 1.05).scale_ok

    def test_is_symmetric_because_scale_error_has_no_preferred_sign(self) -> None:
        assert _checks(5753.9 * 1.02).scale_error_pct == pytest.approx(
            _checks(5753.9 * 0.98).scale_error_pct, abs=1e-9
        )

    def test_racing_line_geometry_does_not_trip_it(self) -> None:
        """Both sides measure the driven path, so the centreline gap is absent."""
        assert _checks(5745.3).scale_ok


class TestOfficialBand:
    def test_shorter_than_official_is_expected(self) -> None:
        checks = _checks(5722.2)
        assert checks.official_error_pct == pytest.approx(-1.461, abs=0.01)
        assert checks.official_ok

    def test_a_corner_dense_circuit_still_fits(self) -> None:
        assert _checks(OFFICIAL * 0.975, driven=OFFICIAL * 0.975).official_ok

    def test_an_implausible_shortfall_is_rejected(self) -> None:
        """Missing a whole section of track should not pass as 'racing line'."""
        assert not _checks(OFFICIAL * 0.95, driven=OFFICIAL * 0.95).official_ok

    def test_longer_than_official_is_rejected(self) -> None:
        """The asymmetry that matters: a racing line cannot exceed the
        centreline, so an overshoot is a defect -- a scale error or a wrap
        double-counting a section. A symmetric band would accept it."""
        overshoot = _checks(OFFICIAL * 1.015, driven=OFFICIAL * 1.015)
        assert overshoot.official_error_pct > 0
        assert not overshoot.official_ok

    def test_small_overshoot_within_noise_is_tolerated(self) -> None:
        assert _checks(OFFICIAL * 1.001, driven=OFFICIAL * 1.001).official_ok


class TestCombined:
    def test_ok_requires_both_halves(self) -> None:
        assert _checks(5722.2).ok
        # Correct scale, implausible length: a wrap double-counting a section.
        assert not _checks(OFFICIAL * 1.02, driven=OFFICIAL * 1.02).ok
        # Plausible length, wrong scale: the axis drifted from the driven path.
        assert not _checks(5722.2, driven=5100.0).ok

    def test_official_error_is_signed_but_scale_error_is_not(self) -> None:
        assert _checks(5722.2).official_error_pct < 0
        assert _checks(5722.2).scale_error_pct > 0
