"""ζ calibration: level (eq 22) and slope (docs/slope-calibrated-zeta/)."""

import math

import pytest

from qopt.exceptions import InstabilityError
from qopt.station import ForkJoinStation, GG1Station
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


def test_phi_is_identically_one_for_mm1():
    # THE invariant that explains why the two calibrations were never distinguished:
    # M/M/1 is the station type most of the suite uses, and there eq 22 is already
    # slope-correct. Any closed form that breaks this is wrong.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    for S in (0.61, 0.8, 1.0, 2.5, 10.0, 1e4):
        # Tolerance is set by cancellation in S*mu - gamma (x) at small x/S, not by
        # derivative accuracy: for x=0.01, relative error is ~1e-14 in x, ~1e-12 in T,
        # ~7e-8 in the finite-difference numerator (verified independent of S).
        assert st.phi(S) == pytest.approx(1.0, abs=1e-7), S


def test_phi_is_one_minus_rho_for_a_zero_cov_station():
    # cov_a = cov_s = 0 gives E[T] = 1/m with no queueing term, so spare capacity buys
    # almost nothing and phi collapses toward 0 -- exactly 1 - rho.
    st = GG1Station(0.6, 1.0, c=1.0, cov_a=0.0, cov_s=0.0)
    for S in (0.7, 1.0, 2.0, 12.0):
        rho = st.gamma / (S * st.mu)
        assert st.phi(S) == pytest.approx(1.0 - rho, abs=1e-8), S


def test_dt_ds_is_negative_for_every_station_kind():
    # More capacity cannot lengthen the sojourn time. A positive derivative would make
    # phi negative and eq 21's sqrt(zeta) complex.
    for st in _every_station_kind():
        assert st.dT_dS(3.0) < 0.0, st.name


def test_phi_is_strictly_positive_for_every_station_kind():
    for st in _every_station_kind():
        assert st.phi(3.0) > 0.0, st.name


def test_the_finite_difference_default_serves_a_subclass_with_no_closed_form():
    # The base implementation is deliberately concrete, not abstract, so slope
    # calibration works for a user station whose derivative qopt has never seen.
    from qopt.station import Station

    class QuadraticStation(Station):
        """E[T] = 1/x**2 -- not a queue qopt ships, which is the point."""

        def sojourn_time(self, S):
            m = S * self.mu
            self._check_stable(m)
            return 1.0 / (m - self.gamma) ** 2

        def sim_node(self, S, job_class):
            raise NotImplementedError

        @property
        def alloc_cost(self):
            return 1.0

        @property
        def default_zeta(self):
            return 1.0

    st = QuadraticStation(gamma=0.5, mu=1.0)
    # T = x**-2 so dT/dx = -2 x**-3 and phi = -dT/dS * x/(mu T) = 2, at every S.
    for S in (0.6, 1.0, 4.0):
        assert st.phi(S) == pytest.approx(2.0, rel=1e-6), S


def test_dt_ds_refuses_an_unstable_capacity():
    # Review Focus 2. At S == gamma/mu the step h is 0; below it h is NEGATIVE, so
    # S - h is the MORE stable side and only S + h raises. An abs(h), or evaluating
    # S - h first and returning early, would hand back a derivative for a station that
    # has no sojourn time at all.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    boundary = st.gamma / st.mu           # 0.6
    for S in (boundary, boundary * 0.5, boundary - 1e-12):
        with pytest.raises(InstabilityError):
            st.dT_dS(S)
        with pytest.raises(InstabilityError):
            st.phi(S)


def test_the_finite_difference_step_stays_inside_the_stability_region():
    # Scaling h to SPARE CAPACITY rather than to S is what guarantees S - h > gamma/mu.
    # A fixed step would fall off the boundary for a station run close to it.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    assert st.dT_dS(0.6 + 1e-9) < 0.0          # a hair above the boundary, still fine


def test_phi_on_an_unbound_gamma_names_the_station():
    # Review Focus 3. A station built bare for a Network has no gamma until bind_gamma.
    # The canonical ValueError must survive -- not AttributeError, and certainly not a
    # number computed from a None.
    st = GG1Station(mu=1.0, c=1.0, cov_a=1.0, cov_s=1.0, name="unbound")
    with pytest.raises(ValueError, match="unbound"):
        st.phi(2.0)
    with pytest.raises(ValueError, match="unbound"):
        st.dT_dS(2.0)
