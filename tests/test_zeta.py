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


from qopt.station import ForkJoinStation, GG1Station, Station


def _every_station_kind(**kwargs):
    """One instance of each concrete station type, all constructor kwargs forwarded."""
    return [
        GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0, name="gg1", **kwargs),
        GG1Station.mm1(0.6, 1.5, c=2.0, name="mm1", **kwargs),
        GG1Station.md1(0.6, 1.5, c=2.0, name="md1", **kwargs),
        ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, name="fj", **kwargs),
        ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                        name="fj-tuned", **kwargs),
    ]


def test_every_station_kind_defaults_to_level():
    for st in _every_station_kind():
        assert st.zeta_mode == ZETA_LEVEL, st.name


def test_every_station_kind_accepts_slope():
    for st in _every_station_kind(zeta_mode=ZETA_SLOPE):
        assert st.zeta_mode == ZETA_SLOPE, st.name


def test_an_unknown_mode_is_rejected_at_construction_by_every_kind():
    for ctor in (
        lambda **kw: GG1Station(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=1.0, **kw),
        lambda **kw: GG1Station.mm1(0.6, 1.5, c=2.0, **kw),
        lambda **kw: GG1Station.md1(0.6, 1.5, c=2.0, **kw),
        lambda **kw: ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, **kw),
    ):
        with pytest.raises(ValueError, match="zeta_mode"):
            ctor(zeta_mode="slopes")


def test_zeta_mode_is_read_only():
    # Read-only for the reason ForkJoinStation.policy is: nothing should change a
    # calibration mid-run, when some iterations have already been priced the other way.
    st = GG1Station.mm1(0.6, 1.5, c=2.0)
    with pytest.raises(AttributeError):
        st.zeta_mode = ZETA_SLOPE


def test_zeta_mode_is_keyword_only():
    # Positional would collide with `name` and with every subclass's own signature.
    with pytest.raises(TypeError):
        GG1Station(0.6, 1.5, 1.0, ZETA_SLOPE)  # type: ignore[misc]


def test_the_mode_survives_a_forkjoin_retune_and_reset():
    # retune/reset_policy rewrite mu, r and r_star. The calibration is not policy state
    # and must not be touched by either.
    st = ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                         zeta_mode=ZETA_SLOPE)
    st.retune(3.0)
    assert st.zeta_mode == ZETA_SLOPE
    st.reset_policy()
    assert st.zeta_mode == ZETA_SLOPE
