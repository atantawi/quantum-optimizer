"""ζ calibration: level (eq 22) and slope (docs/slope-calibrated-zeta/)."""

import math

import pytest

from qopt.zeta import (
    ZETA_LEVEL,
    ZETA_MODES,
    ZETA_SHAPE_TOL,
    ZETA_SLOPE,
    resolve_zeta_mode,
)


def test_the_two_modes_are_distinct_strings():
    assert ZETA_LEVEL == "level"
    assert ZETA_SLOPE == "slope"
    assert ZETA_MODES == (ZETA_LEVEL, ZETA_SLOPE)


def test_none_resolves_to_the_level_default():
    # The default must be the incumbent: an unspecified mode is eq 22, always.
    assert resolve_zeta_mode(None) == ZETA_LEVEL


def test_each_mode_resolves_to_itself():
    for mode in ZETA_MODES:
        assert resolve_zeta_mode(mode) == mode


def test_an_unknown_mode_is_rejected_and_the_message_names_the_alternatives():
    with pytest.raises(ValueError) as excinfo:
        resolve_zeta_mode("sloped")
    message = str(excinfo.value)
    assert "sloped" in message
    assert ZETA_LEVEL in message and ZETA_SLOPE in message


def test_a_non_string_mode_is_rejected():
    # A caller reaching for a bool or a number is confusing this with a flag.
    for bad in (True, 1, 1.0, [], object()):
        with pytest.raises(ValueError):
            resolve_zeta_mode(bad)


def test_the_shape_tolerance_default_is_a_fraction_not_a_percentage():
    # 0.25 means 25%. A value > 1 would mean the cross-check never fires.
    assert ZETA_SHAPE_TOL == 0.25
    assert math.isfinite(ZETA_SHAPE_TOL) and 0.0 < ZETA_SHAPE_TOL < 1.0
