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


def _central_difference(st, S, frac=1e-7):
    """The base-class difference, computed independently of whatever dT_dS now does."""
    h = frac * (S - st.gamma / st.mu)
    return (st.sojourn_time(S + h) - st.sojourn_time(S - h)) / (2.0 * h)


def test_gg1_closed_form_matches_the_central_difference():
    # Over loads AND coefficients of variation: the closed form has a k*gamma*(2m-gamma)
    # term whose sign and grouping a single (cov, load) pair cannot pin.
    for cov_a in (0.0, 0.5, 1.0, 2.0, 5.0):
        for cov_s in (0.0, 1.0, 3.0):
            st = GG1Station(0.6, 1.5, c=1.0, cov_a=cov_a, cov_s=cov_s)
            for S in (0.45, 0.6, 1.0, 3.0, 20.0):
                assert st.dT_dS(S) == pytest.approx(
                    _central_difference(st, S), rel=1e-6
                ), (cov_a, cov_s, S)


def test_gg1_phi_is_exactly_one_for_mm1_under_the_closed_form():
    # Now abs=1e-12 rather than the difference-limited 1e-9 of the fd default: the
    # closed form should make this invariant hold to machine precision.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    for S in (0.61, 1.0, 2.5, 100.0, 1e6):
        assert st.phi(S) == pytest.approx(1.0, abs=1e-12), S


def test_gg1_phi_is_exactly_one_minus_rho_for_cov_zero_under_the_closed_form():
    st = GG1Station(0.6, 1.0, c=1.0, cov_a=0.0, cov_s=0.0)
    for S in (0.7, 1.0, 2.0, 50.0):
        rho = st.gamma / (S * st.mu)
        assert st.phi(S) == pytest.approx(1.0 - rho, rel=1e-12), S


def test_gg1_phi_matches_the_documented_sensitivity_table():
    # Spec section 8.2, and this test is that table's source of record. gamma = 0.6,
    # cov_s = 1, phi read at three loads for a true and a mis-specified cov_a.
    #
    # It is here because under slope calibration `cov_a` stops being decorative on the
    # simulated path -- it is never sent to qsim and never measured back, so it enters
    # the allocation only through phi. Understating it drives phi toward 1 and degrades
    # to level calibration; OVERSTATING it can land worse than the incumbent.
    #
    # NOTE: three of the seven rows below were corrected from the task brief's literal
    # text (1.240030 -> 1.240326 at rho=0.67/cov_a=3; 1.039697 -> 1.039583 at
    # rho=0.95/cov_a=3; 1.165419 -> 1.165411 at rho=0.67/cov_a=2). The brief's own
    # scale-free formula phi = ((1-rho)^2 + k*rho*(2-rho)) / ((1-rho) + k*rho), and the
    # already-locked docs/slope-calibrated-zeta/findings.md formula it specializes,
    # both reproduce every OTHER row and spot-check in the brief exactly (including the
    # k=25/rho=0.25 -> 23/14 check and findings.md's own probe-verified cov=2/cov=5
    # table), and the unmodified fd default -- already passing, unchanged by this task
    # -- agrees with the corrected values to ~1e-9, not the brief's originals. The
    # brief's own test_a_wrong_cov_a_moves_... below independently pins phi at this
    # exact (rho=0.67, k=5) point to ~1.2403 (abs=0.005), confirming 1.240326 over
    # 1.240030. See task-4-report.md for the full derivation.
    rows = [
        # rho,  cov_a, expected phi
        (0.30, 1.0, 1.000000),
        (0.30, 3.0, 1.381818),
        (0.67, 1.0, 1.000000),
        (0.67, 3.0, 1.240326),
        (0.95, 1.0, 1.000000),
        (0.95, 3.0, 1.039583),
        (0.67, 2.0, 1.165411),
    ]
    gamma = 0.6
    for rho, cov_a, expected in rows:
        st = GG1Station(gamma, 1.0, c=1.0, cov_a=cov_a, cov_s=1.0)
        S = gamma / rho / st.mu          # m = gamma/rho
        assert st.phi(S) == pytest.approx(expected, rel=1e-6), (rho, cov_a)


def test_a_wrong_cov_a_moves_expected_sojourn_time_far_more_than_it_moves_phi():
    # Why the measured-vs-analytic E[T] cross-check (Task 9) is the right detector: the
    # symptom is roughly ten times larger than the defect it indicates.
    S = 0.6 / 0.67
    truth = GG1Station(0.6, 1.0, c=1.0, cov_a=1.0, cov_s=1.0)
    wrong = GG1Station(0.6, 1.0, c=1.0, cov_a=3.0, cov_s=1.0)
    phi_error = wrong.phi(S) / truth.phi(S) - 1.0
    t_error = wrong.sojourn_time(S) / truth.sojourn_time(S) - 1.0
    assert phi_error == pytest.approx(0.240, abs=0.005)
    assert t_error == pytest.approx(2.68, abs=0.02)
    assert t_error > 10 * phi_error


def test_forkjoin_closed_form_matches_the_central_difference():
    # Across r AND r_star, because r_star rewrites which server binds: r_star < 1 swaps
    # the anchor, and the closed form reads the EFFECTIVE mu and r, not the constructed
    # ones. r_star = 1.0 is included deliberately -- see the next test.
    for r in (1.0, 2.0, 4.0, 20.0):
        for r_star in (0.05, 1.0, 2.0, 3.0, 50.0):
            st = ForkJoinStation(0.45, 1.0, r=r, c1=4.0, c2=1.0, r_star=r_star)
            base = st.gamma / st.mu
            for mult in (1.01, 1.5, 4.0, 100.0):
                S = base * mult
                assert st.dT_dS(S) == pytest.approx(
                    _central_difference(st, S), rel=1e-5
                ), (r, r_star, mult)


def test_forkjoin_closed_form_is_right_where_the_policy_helper_is_wrong():
    # forkjoin_policy._dt_dm1 takes the "m1 is the non-bottleneck" branch at m1 == m2 and
    # drops the alpha*t_bot term entirely. That is harmless inside _min_on_spend_line,
    # whose kink is measure-zero there, but it is wrong for pricing dT/d(spend) -- and
    # r_star = 1 is exactly where tight budgets and every beta1 == beta2 station sit.
    # Measured at 17.3% for spend/floor = 1.05 (findings.md section 5).
    #
    # This test does not reimplement the helper; it pins that the radial derivative
    # agrees with a difference of the SHIPPED sojourn_time at r_star = 1, which is the
    # property the helper lacks.
    st = ForkJoinStation(0.45, 1.0, r=1.0, c1=4.0, c2=1.0, r_star=1.0)
    base = st.gamma / st.mu
    for mult in (1.05, 1.5, 4.0):
        S = base * mult
        assert st.dT_dS(S) == pytest.approx(_central_difference(st, S), rel=1e-5), mult


def test_forkjoin_phi_stays_near_one_but_not_at_one():
    # The fork-join is the mildest deviation of the station types qopt ships -- within
    # about 1.3% of 1 -- which is why a network whose only non-M/M/1 stations are
    # fork-joins gains only 0.0002-0.039%. It is still NOT 1, so it still moves.
    st = ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0)
    base = st.gamma / st.mu
    values = [st.phi(base * m) for m in (1.01, 1.5, 4.0, 100.0)]
    assert all(0.95 < v < 1.05 for v in values), values
    assert any(abs(v - 1.0) > 1e-6 for v in values), values


def test_forkjoin_phi_is_positive_over_a_wide_grid():
    # Spec assumption 2, and eq 21 needs a strictly positive zeta: sqrt(w*zeta/...).
    for r in (1.0, 2.0, 20.0):
        for r_star in (0.05, 1.0, 3.0, 50.0):
            st = ForkJoinStation(0.45, 1.0, r=r, c1=4.0, c2=1.0, r_star=r_star)
            base = st.gamma / st.mu
            for mult in (1.000001, 1.001, 1.1, 2.0, 10.0, 1e4, 1e8):
                assert st.phi(base * mult) > 0.0, (r, r_star, mult)
