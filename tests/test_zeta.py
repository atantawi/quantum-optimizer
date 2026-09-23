"""ζ calibration: level (eq 22) and slope (docs/slope-calibrated-zeta/)."""

import math
import warnings

import pytest

from qopt.allocator import min_feasible_budget
from qopt.exceptions import InstabilityError
from qopt.optimizer import Optimizer, Result
from qopt.station import _FD_STEP, ForkJoinStation, GG1Station, Station
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


# Module scope, not nested in one test: EVERY test below that means to exercise the base
# class's finite difference has to build a station with no dT_dS override, and the two
# that once used GG1Station.mm1 for it silently reached its closed form instead.
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


def test_the_finite_difference_default_serves_a_subclass_with_no_closed_form():
    # The base implementation is deliberately concrete, not abstract, so slope
    # calibration works for a user station whose derivative qopt has never seen.
    st = QuadraticStation(gamma=0.5, mu=1.0)
    # T = x**-2 so dT/dx = -2 x**-3 and phi = -dT/dS * x/(mu T) = 2, at every S.
    for S in (0.6, 1.0, 4.0):
        assert st.phi(S) == pytest.approx(2.0, rel=1e-6), S


def test_dt_ds_refuses_an_unstable_capacity():
    # Review Focus 2, on QuadraticStation so that the BASE finite difference is what
    # runs. GG1Station.mm1 would route to its closed form instead -- which is a real but
    # SEPARATE property, kept in test_the_gg1_closed_form_dt_ds_refuses_an_unstable_capacity.
    #
    # What this pins is "no derivative for a station that has no sojourn time", and
    # nothing more. At and below the boundary every evaluation the difference makes is
    # unstable, so the subclass's own _check_stable raises whichever of the two calls
    # runs first: no input distinguishes the sign of h or their order by OUTCOME, though
    # the two orders do quote different S*mu in the message they raise. At
    # S == gamma/mu the step is 0, and sojourn_time(S) raises before the division by
    # 2*h can. The step property that IS pinnable is its scaling -- next test.
    st = QuadraticStation(gamma=0.5, mu=1.0)
    boundary = st.gamma / st.mu           # 0.5
    for S in (boundary, boundary * 0.5, boundary - 1e-12):
        with pytest.raises(InstabilityError):
            st.dT_dS(S)
        with pytest.raises(InstabilityError):
            st.phi(S)


def test_the_gg1_closed_form_dt_ds_refuses_an_unstable_capacity():
    # GG1Station.dT_dS never calls sojourn_time, so it needs a _check_stable of its own
    # or it would hand back a finite number for a capacity with no sojourn time at all.
    # This test exists because the mm1 version of the assertion above was pinning THIS,
    # not the base finite difference -- which is covered separately, on QuadraticStation.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    boundary = st.gamma / st.mu           # 0.6
    for S in (boundary, boundary * 0.5, boundary - 1e-12):
        with pytest.raises(InstabilityError):
            st.dT_dS(S)
        with pytest.raises(InstabilityError):
            st.phi(S)


def test_the_finite_difference_step_stays_inside_the_stability_region():
    # QuadraticStation, again because GG1Station.mm1 overrides dT_dS: the previous
    # version of this test asserted `mm1.dT_dS(0.6 + 1e-9) < 0.0`, which is true by
    # inspection of a closed form whose every term is negative and involves no step.
    #
    # The kill set, measured: scaling h to SPARE CAPACITY, `S - gamma/mu`, is what keeps
    # S - h inside the stability region. A hair above the boundary the spare capacity is
    # 1e-9, so h = _FD_STEP * S (1e-7 * 0.5) and a fixed h = _FD_STEP (1e-7) both put
    # S - h BELOW gamma/mu and raise InstabilityError from inside a derivative that is
    # perfectly well defined. Only the shipped scaling returns a number here.
    st = QuadraticStation(gamma=0.5, mu=1.0)
    boundary = st.gamma / st.mu                      # 0.5
    slope = st.dT_dS(boundary + 1e-9)                # a hair above it, still fine
    assert math.isfinite(slope) and slope < 0.0, slope


def test_the_finite_difference_endpoints_stay_representably_distinct():
    # Near the boundary the step is a fraction of a spare capacity that is itself tiny,
    # so h underflows relative to S: `S + h == S - h == S`, the difference of two
    # IDENTICAL sojourn times is 0, and phi comes back -0.0. `zeta_from` then rejects a
    # point that is perfectly stable -- T is finite and x > 0 -- with the "phi is
    # non-positive" ValueError, which names the wrong cause.
    #
    # The kill set: `h < ulp(S)` at every spare capacity below ~1.1e-9 for this station,
    # i.e. the whole of the region 1e-7 * spare < ulp(S). Restoring `/ (2.0 * h)` with
    # `S ± h` endpoints returns -0.0 at every S below.
    st = QuadraticStation(gamma=0.5, mu=1.0)
    boundary = st.gamma / st.mu                      # 0.5
    for spare in (1e-10, 3e-10, 1e-12, 1e-14, 1e-15):
        S = boundary + spare
        assert S * st.mu > st.gamma, spare           # genuinely stable, so phi must exist
        slope = st.dT_dS(S)
        assert math.isfinite(slope) and slope < 0.0, (spare, slope)
        # T = x**-2 makes phi exactly 2 at every S -- the boundary is no exception.
        #
        # The tolerance is DERIVED, not chosen. Once widening is in force the step is one
        # ulp of S, so the central difference's own second-order truncation error is
        # O((ulp(S)/spare)**2). Measured relative error at the last three spares is
        # 2.5e-08, 2.5e-04 and 2.5e-02, against a bound of 1.0e-06, 9.9e-04 and 9.9e-02
        # -- the 1e-6 floor decides the first of those, where the truncation term is only
        # 9.9e-08, and the derived term decides the other two. That is the accuracy the
        # number line affords at this S and no step choice beats it; the bound is still
        # far from vacuous, since the pre-fix -0.0 is a relative error of 1.0 and the
        # half-collapsed band's factor of 2 is 1.0 as well.
        resolvable = math.ulp(S) / spare
        assert st.phi(S) == pytest.approx(
            2.0, rel=max(1e-6, 8.0 * resolvable ** 2)
        ), (spare, st.phi(S))
        # And the calibration it feeds must go through rather than reject the point.
        assert st.zeta_from(st.sojourn_time(S), S) > 0.0, spare


def test_the_finite_difference_divides_by_the_spacing_it_actually_used():
    # The half-collapsed band, which is worse than the collapse above because it is
    # SILENT: `S + h != S - h`, so no guard fires, but rounding has moved the endpoints
    # to a spacing that is not the nominal `2 * h`. Dividing by `2 * h` then misreports
    # the slope by the ratio of the two -- measured up to a clean factor of 2, with no
    # error and no warning, straight into eq 21's allocation.
    #
    # This is why the fix is `(hi - lo)` and not merely "widen when the endpoints
    # collapse": widening alone leaves this band reporting phi = 3.97 where the station's
    # phi is exactly 2. Sterbenz makes `hi - lo` exact for endpoints this close, so the
    # divisor is the spacing that was actually differenced.
    st = QuadraticStation(gamma=0.5, mu=1.0)
    boundary = st.gamma / st.mu
    # Each of these has distinct endpoints whose true spacing is not 2*h; the last is
    # the factor-of-2 case, the first two are the ordinary ~0.08% skew far from it.
    for spare in (5.6e-10, 1e-9, 1.2e-9, 1e-8, 1e-7):
        S = boundary + spare
        h = _FD_STEP * (S - boundary)
        assert S + h != S - h, spare                 # not the collapsed case above
        assert st.phi(S) == pytest.approx(2.0, rel=1e-6), (spare, st.phi(S))


def test_the_finite_difference_keeps_its_lower_endpoint_inside_the_region():
    # One ulp above the boundary there is NO representable capacity between S and
    # gamma/mu, so a central difference is impossible and `nextafter(S, -inf)` is the
    # boundary itself. Widening blindly would raise InstabilityError from inside a
    # derivative whose station is stable; the difference has to go one-sided instead.
    #
    # Only the SIGN and finiteness are claimed here, deliberately. With a single ulp of
    # resolution on a T that varies as x**-2 the truncation error is O(1) -- phi measures
    # 0.75 against an exact 2 -- and no step choice can recover accuracy at this S. What
    # the sign buys is that phi stays positive, so slope calibration proceeds on a stable
    # station instead of rejecting it, which is the defect being fixed. A caller this
    # close to a boundary has a capacity problem, not a derivative problem.
    st = QuadraticStation(gamma=0.5, mu=1.0)
    boundary = st.gamma / st.mu
    S = math.nextafter(boundary, math.inf)
    assert math.nextafter(S, -math.inf) == boundary   # no room below: one-sided or bust
    slope = st.dT_dS(S)
    assert math.isfinite(slope) and slope < 0.0, slope
    assert st.phi(S) > 0.0, st.phi(S)


def test_the_finite_difference_reports_a_zero_it_cannot_resolve():
    # The residual corner widening cannot fix, pinned so it cannot quietly get worse.
    # E[T] depends on x = S*mu - gamma, and within an ulp or two of the boundary a
    # one-ulp change in S need not move `S*mu` at all: for gamma=0.6, mu=1.5 the
    # capacities either side of the boundary share a single `S*mu`, so both sojourn times
    # are bitwise equal and the difference is a true 0 at the finest available resolution.
    #
    # Claimed here: it returns 0 rather than a wrong nonzero number, and the calibration
    # REFUSES the point rather than allocating on it. That is the safe outcome, and it is
    # unreachable in practice -- GG1Station overrides dT_dS with a closed form, so the
    # base difference has to be called explicitly to see this at all.
    st = GG1Station(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=1.0, zeta_mode=ZETA_SLOPE)
    boundary = st.gamma / st.mu
    S = math.nextafter(boundary, math.inf)
    assert S * st.mu == math.nextafter(S, math.inf) * st.mu   # the collision itself
    assert Station.dT_dS(st, S) == 0.0
    # The closed form this station actually uses is unaffected, which is why no shipped
    # station reaches the corner.
    assert st.dT_dS(S) < 0.0 and math.isfinite(st.dT_dS(S))


def test_the_finite_difference_still_refuses_at_and_below_the_boundary():
    # The widening must not resurrect a derivative where there is no sojourn time. At
    # S == gamma/mu the spare capacity is 0, so every candidate endpoint is at or below
    # the boundary and `sojourn_time` has to raise -- the property the old `h == 0`
    # arithmetic gave for free and a nextafter fallback could silently take away.
    st = QuadraticStation(gamma=0.5, mu=1.0)
    boundary = st.gamma / st.mu
    for S in (boundary, math.nextafter(boundary, -math.inf), boundary * 0.5):
        with pytest.raises(InstabilityError):
            st.dT_dS(S)

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
    # the allocation only through phi. Every row below is at cov_s = 1, where
    # k = (cov_a**2 + 1)/2 and phi == 1 at cov_a == 1. In general it is k, not cov_a,
    # that controls phi -- phi is strictly increasing in k and equals 1 exactly at
    # k == 1 -- so the cov_a that reproduces level calibration is sqrt(2 - cov_s**2),
    # which is 1 only here at cov_s = 1. See README.md's zeta-calibration subsection.
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


def test_level_mode_is_bit_for_bit_the_shipped_expression():
    # `==`, not approx: the default path must not move by one ulp. Binding x to a local
    # does not change float semantics, but reordering the multiply would.
    for st in _every_station_kind():
        for T in (0.5, 2.5, 137.125):
            for S in (2.0, 7.5):
                assert st.zeta_from(T, S) == T * (S * st.mu - st.gamma), (st.name, T, S)


def test_slope_mode_is_phi_times_the_level_value():
    for st in _every_station_kind(zeta_mode=ZETA_SLOPE):
        for S in (2.0, 7.5):
            T = st.sojourn_time(S)
            level = T * (S * st.mu - st.gamma)
            assert st.zeta_from(T, S) == pytest.approx(st.phi(S) * level, rel=1e-15), st.name


def test_slope_zeta_is_x_squared_times_the_slope_in_x():
    # The identity that makes eq 21 exact: zeta/x has derivative -|dT/dx| in x, so
    # zeta = x**2 * |dT/dx|. Checked against dT_dS with the mu factor undone.
    st = GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0, zeta_mode=ZETA_SLOPE)
    for S in (0.5, 1.0, 4.0):
        x = S * st.mu - st.gamma
        dT_dx = st.dT_dS(S) / st.mu
        assert st.zeta_from(st.sojourn_time(S), S) == pytest.approx(
            x ** 2 * abs(dT_dx), rel=1e-12
        ), S


def test_mm1_slope_and_level_zeta_agree_exactly():
    # phi == 1 identically, so the two calibrations must coincide -- and since phi is
    # computed, not assumed, this also pins that the closed form returns exactly 1.
    level = GG1Station.mm1(0.6, 1.0, c=2.0)
    slope = GG1Station.mm1(0.6, 1.0, c=2.0, zeta_mode=ZETA_SLOPE)
    for S in (0.7, 1.0, 5.0):
        T = level.sojourn_time(S)
        assert slope.zeta_from(T, S) == pytest.approx(level.zeta_from(T, S), rel=1e-12)


def test_zeta_from_is_linear_in_T_in_both_modes():
    # The ONLY property Optimizer._noise_floor relies on: it propagates a CI half-width
    # through this same hook, so zeta_from(h, S) must be the correctly scaled
    # perturbation. If slope mode were not linear in T, the noise floor would need
    # special-casing and this change would not be cheap.
    for mode in ZETA_MODES:
        for st in _every_station_kind(zeta_mode=mode):
            for S in (2.0, 9.0):
                base = st.zeta_from(1.0, S)
                assert st.zeta_from(2.0, S) == pytest.approx(2.0 * base, rel=1e-14)
                assert st.zeta_from(0.25, S) == pytest.approx(0.25 * base, rel=1e-14)


def test_the_station_zeta_method_follows_the_mode():
    st_l = GG1Station.md1(0.6, 1.0, c=1.0)
    st_s = GG1Station.md1(0.6, 1.0, c=1.0, zeta_mode=ZETA_SLOPE)
    assert st_s.zeta(2.0) != st_l.zeta(2.0)
    assert st_s.zeta(2.0) == pytest.approx(st_s.phi(2.0) * st_l.zeta(2.0), rel=1e-12)


def test_a_non_positive_phi_is_refused_and_the_message_names_the_station():
    # A station whose E[T] does not decrease in capacity has no slope to calibrate to.
    # allocate would reject the zeta anyway; this fails earlier with a message that says
    # which station and at what capacity, because the cause is a modelling error.
    from qopt.station import Station

    class FlatStation(Station):
        def sojourn_time(self, S):
            self._check_stable(S * self.mu)
            return 1.0                      # constant: dT/dS == 0, so phi == 0

        def sim_node(self, S, job_class):
            raise NotImplementedError

        @property
        def alloc_cost(self):
            return 1.0

        @property
        def default_zeta(self):
            return 1.0

    st = FlatStation(gamma=0.5, mu=1.0, name="flat", zeta_mode=ZETA_SLOPE)
    with pytest.raises(ValueError, match="flat"):
        st.zeta_from(1.0, 2.0)
    # Level mode on the same station is untouched: the guard is slope-only.
    ok = FlatStation(gamma=0.5, mu=1.0, name="flat", zeta_mode=ZETA_LEVEL)
    assert ok.zeta_from(1.0, 2.0) == 1.5      # T*x = 1.0 * (2.0*1.0 - 0.5)


def test_an_infinite_phi_is_refused_and_the_message_names_the_station():
    # The isfinite half of the guard, pinned on its own: a station whose derivative is
    # unbounded has no finite slope to calibrate to, even though phi > 0.0 holds here --
    # +inf passes that comparison, so it is only the isfinite half that catches it.
    from qopt.station import Station

    class UnboundedSlopeStation(Station):
        def sojourn_time(self, S):
            self._check_stable(S * self.mu)
            return 1.0

        def dT_dS(self, S):
            self._check_stable(S * self.mu)
            return -math.inf            # phi = -dT_dS * x / (mu * T) = +inf

        def sim_node(self, S, job_class):
            raise NotImplementedError

        @property
        def alloc_cost(self):
            return 1.0

        @property
        def default_zeta(self):
            return 1.0

    st = UnboundedSlopeStation(gamma=0.5, mu=1.0, name="unbounded", zeta_mode=ZETA_SLOPE)
    with pytest.raises(ValueError, match="unbounded"):
        st.zeta_from(1.0, 2.0)


def test_a_tiny_phi_still_produces_a_usable_zeta():
    # Review Focus 4. A cov = 0 station has phi = 1 - rho exactly, so at extreme load
    # zeta_slope is ~1e-6 of zeta_level. ZETA_FLOOR is local to noise_floor and clamps
    # nothing in allocate, which needs only finite and > 0 -- so this must pass straight
    # through rather than being clamped or refused.
    from qopt.allocator import allocate

    st = GG1Station(0.6, 1.0, c=1.0, cov_a=0.0, cov_s=0.0, zeta_mode=ZETA_SLOPE)
    S = st.gamma / st.mu * (1.0 + 1e-6)
    z = st.zeta_from(st.sojourn_time(S), S)
    assert 0.0 < z < 1e-9
    assert math.isfinite(z)
    partner = GG1Station.mm1(0.6, 1.0, c=1.0)
    caps = allocate([st, partner], 10.0, [z, partner.zeta(2.0)])
    assert all(math.isfinite(c) and c > 0 for c in caps)


def _mixed_pair(mode_a, mode_b):
    return [
        GG1Station.md1(0.6, 1.5, c=2.0, name="md1", zeta_mode=mode_a),
        GG1Station.mm1(1.2, 3.0, c=0.5, name="mm1", zeta_mode=mode_b),
    ]


def test_result_reports_the_mode_of_each_station():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_LEVEL)
    res = Optimizer(stations, 4.0 * min_feasible_budget(stations)).run()
    assert res.zeta_mode == [ZETA_SLOPE, ZETA_LEVEL]


def test_phi_is_literally_one_for_a_level_station():
    # Not a computed phi: level calibration never consults the derivative, so the
    # default path must neither start paying for one nor acquire a new way to fail.
    # A level station whose phi would RAISE still reports 1.0.
    class BrokenDerivative(GG1Station):
        def dT_dS(self, S):
            raise AssertionError("a level-mode run must never evaluate the derivative")

    stations = [
        BrokenDerivative(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=0.0, name="level"),
        GG1Station.mm1(1.2, 3.0, c=0.5, name="mm1"),
    ]
    res = Optimizer(stations, 4.0 * min_feasible_budget(stations)).run()
    assert res.zeta_phi == [1.0, 1.0]


def test_phi_is_reported_for_a_slope_station_and_recovers_the_eq22_value():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_LEVEL)
    res = Optimizer(stations, 4.0 * min_feasible_budget(stations)).run()
    # An M/D/1 station's phi is strictly below 1.
    assert 0.0 < res.zeta_phi[0] < 1.0
    assert res.zeta_phi[1] == 1.0
    # zeta is the value that actually drove the allocation, so dividing out phi gives
    # back eq 22's level calibration.
    level = res.sojourn_times[0] * (res.capacities[0] * stations[0].mu - stations[0].gamma)
    assert res.zeta[0] / res.zeta_phi[0] == pytest.approx(level, rel=1e-12)
    assert res.zeta[0] == pytest.approx(res.zeta_phi[0] * level, rel=1e-12)


def test_the_new_result_fields_are_all_defaulted():
    # tests/test_optimizer_loop.py and the example tests construct Result directly.
    res = Result(
        capacities=[1.0], sojourn_times=[2.0], zeta=[1.0], objective=2.0,
        iterations=1, converged=True, residual=0.0,
    )
    assert res.zeta_phi == []
    assert res.zeta_mode == []
    assert res.zeta_shape_flags == []


def test_the_reported_zeta_closes_the_eq_21_round_trip_on_the_analytic_path():
    # Re-running allocate on the reported zeta must reproduce the reported capacities.
    #
    # ANALYTIC path only, and that is the whole scope of this test. `Result.zeta` is
    # recomputed from the REPORTED sojourn times; on the analytic path the final
    # evaluation is deterministic at the converged S, so those are the same numbers the
    # last allocate saw and the round trip closes. On a stochastic analyzer they come
    # from the fresh-seeded final evaluation -- a different sample path from the CRN
    # iterate that set the capacities -- and the round trip does NOT close. Measured on
    # this same pair, with an analyzer that returns the analytic E[T] unperturbed to the
    # loop and scales station i by factors[i] on the fresh_seed call only: factors
    # [1.2, 0.8] gives +4.06% / -11.20%, [1.1, 0.9] gives +2.06% / -5.68%, [1.05, 0.95]
    # gives +1.04% / -2.86%. The gap scales with the perturbation and has no fixed size,
    # so quote the recipe with any figure. A UNIFORM factor closes to ~4e-13 instead
    # ([1.1, 1.1]), because eq 21 is a share rule -- so no test that scales all stations
    # alike can see the gap at all. README's `Result.zeta` sentence states which E[T] the
    # reported zeta belongs to; do not read this test as promising more.
    from qopt.allocator import allocate

    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    res = Optimizer(stations, C).run()
    again = allocate(stations, C, res.zeta)
    for a, b in zip(again, res.capacities):
        assert a == pytest.approx(b, rel=1e-8)


# Station specs and objectives from docs/slope-calibrated-zeta/probe-output.txt section 3.
# The WEIGHTS are part of the data: eq 21 combines w and zeta under one square root, so a
# dropped weight is indistinguishable from a phi error in the objective alone.

def _net_fj_mm1(mode):
    return [
        ForkJoinStation(0.45, 1.0, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                        name="FJ-A", zeta_mode=mode),
        ForkJoinStation(0.80, 2.0, 2.0, r=2.0, c1=1.0, c2=3.0, r_star="tuned",
                        name="FJ-B", zeta_mode=mode),
        GG1Station(0.60, 1.5, 1.0, c=2.0, cov_a=1.0, cov_s=1.0, name="SS-1",
                   zeta_mode=mode),
        GG1Station(1.20, 3.0, 1.5, c=0.5, cov_a=1.0, cov_s=1.0, name="SS-2",
                   zeta_mode=mode),
    ]


def _net_mixed_cov(mode):
    return [
        ForkJoinStation(0.45, 1.0, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                        name="FJ-A", zeta_mode=mode),
        GG1Station(0.60, 1.5, 1.0, c=2.0, cov_a=1.0, cov_s=0.0, name="M/D/1",
                   zeta_mode=mode),
        GG1Station(1.20, 3.0, 1.5, c=0.5, cov_a=2.0, cov_s=2.0, name="cov2",
                   zeta_mode=mode),
        GG1Station(0.90, 2.0, 1.0, c=1.0, cov_a=1.0, cov_s=1.0, name="M/M/1",
                   zeta_mode=mode),
    ]


_MULTS = (1.01, 1.05, 1.2, 1.5, 2, 5, 20)

_FJ_MM1_LEVEL = (845.653557502, 169.433928883, 42.584355698, 17.147817304,
                 8.624738064, 2.176046247, 0.459984473)
_FJ_MM1_SLOPE = (845.326602738, 169.374048331, 42.572541019, 17.144312750,
                 8.623541689, 2.175978022, 0.459983404)
_MIXED_LEVEL = (672.781044226, 132.897962166, 32.456697879, 12.821589655,
                6.396010744, 1.600929447, 0.336149757)
_MIXED_SLOPE = (672.409070110, 132.736738624, 32.343754976, 12.755896143,
                6.364185455, 1.598169104, 0.336090106)


def _objectives(build, mode):
    out = []
    for mult in _MULTS:
        # A FRESH network per row: a tuned fork-join is mutated by retune, and
        # min_spend must be read before any run has moved the ray.
        stations = build(mode)
        C = mult * min_feasible_budget(stations)
        out.append(Optimizer(stations, C).run().objective)
    return out


@pytest.mark.parametrize("build,expected", [
    (_net_fj_mm1, _FJ_MM1_LEVEL),
    (_net_mixed_cov, _MIXED_LEVEL),
])
def test_level_objectives_reproduce_the_reference_probe_run(build, expected):
    # Review Focus 1. Today's code path already produces these, so this test fails if
    # the station specs were ported wrongly -- a dropped weight, a swapped cov, the
    # wrong r_star -- BEFORE slope mode is exercised at all. Do not loosen the
    # tolerance to make it pass; fix the specs.
    got = _objectives(build, ZETA_LEVEL)
    for mult, g, e in zip(_MULTS, got, expected):
        assert g == pytest.approx(e, rel=1e-9), mult


@pytest.mark.parametrize("build,expected", [
    (_net_fj_mm1, _FJ_MM1_SLOPE),
    (_net_mixed_cov, _MIXED_SLOPE),
])
def test_slope_objectives_reproduce_the_reference_probe_run(build, expected):
    # rel AND abs: the reference table is printed to nine decimal places, so its own
    # absolute precision is only ~5e-10. At the smallest magnitude row (mult=20, ~0.336)
    # that caps the achievable relative precision at ~1.5e-9 -- measured at 1.28e-9 on
    # _MIXED_SLOPE there, which a rel-only tolerance of 1e-9 would reject even though the
    # port is correct. `rel` pins the large rows to nine significant figures; `abs` pins
    # the small rows at the reference's own printed precision.
    got = _objectives(build, ZETA_SLOPE)
    for mult, g, e in zip(_MULTS, got, expected):
        assert g == pytest.approx(e, rel=1e-9, abs=1e-9), mult


@pytest.mark.parametrize("build", [_net_fj_mm1, _net_mixed_cov])
def test_slope_never_loses_to_level_on_the_reference_networks(build):
    # From LIVE runs, not from the reference tuples above: the objective is a weighted
    # sum of sojourn times, so lower is better, and slope calibration is exact, so it
    # cannot be beaten by the level approximation. Asserting this over the frozen
    # reference numbers instead would be arithmetic on constants -- a test that passes
    # whatever the code does.
    level = _objectives(build, ZETA_LEVEL)
    slope = _objectives(build, ZETA_SLOPE)
    for mult, l, s in zip(_MULTS, level, slope):
        assert s <= l, mult


def test_the_mixed_network_gain_reaches_the_documented_half_percent():
    # findings.md section 4: up to 0.515%, at C/floor = 1.5 on the mixed-cov network.
    # The gain tracks |phi - 1|, so it is the network with an M/D/1 and a cov = 2
    # station that shows it, not the fork-join pair. Measured live for the same reason
    # as the test above.
    i = _MULTS.index(1.5)
    mixed_l = _objectives(_net_mixed_cov, ZETA_LEVEL)
    mixed_s = _objectives(_net_mixed_cov, ZETA_SLOPE)
    gain = (mixed_l[i] - mixed_s[i]) / mixed_l[i]
    assert gain == pytest.approx(0.00515, rel=0.02)
    # And the fork-join-only network gains far less, for the same reason.
    fj_l = _objectives(_net_fj_mm1, ZETA_LEVEL)
    fj_s = _objectives(_net_fj_mm1, ZETA_SLOPE)
    fj_gain = (fj_l[i] - fj_s[i]) / fj_l[i]
    assert fj_gain < 0.0005


def test_slope_calibration_needs_the_retune_to_run_last():
    # The radial derivative equals the true marginal only ON the optimal ray, so
    # zeta_from must see a station whose ray is already optimal for the spend it holds.
    # The Optimizer guarantees that by calling retune LAST in each iteration.
    #
    # This test pins the CONSEQUENCE rather than the ordering: every tuned fork-join
    # must sit on the ray that is locally optimal for the spend it actually holds.
    # Deliberately stopped SHORT of convergence, which is what gives the test its
    # teeth. Retuning last makes the property hold bit-for-bit at EVERY iterate, so
    # nothing is lost by stopping early; retuning before eq 21 leaves the ray one
    # allocation stale, and the gap is then the size of the remaining step -- 2.8e-4
    # relative at max_iter=3, against the 1e-9 asserted below. At convergence that
    # same gap collapses to 8.5e-12 and this assertion would no longer see it, which
    # is why the iterate matters here and the tolerance does not.
    from qopt.forkjoin_policy import optimal_ray

    stations = _net_mixed_cov(ZETA_SLOPE)
    C = 1.5 * min_feasible_budget(stations)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = Optimizer(stations, C, max_iter=3).run()
    fj = stations[0]
    spend = res.capacities[0] * fj.alloc_cost
    assert fj.r_star == pytest.approx(
        optimal_ray(fj.gamma, fj.mu_base, fj.r_base, fj.c1, fj.c2, spend), rel=1e-9
    )


def test_slope_mode_on_an_all_mm1_network_matches_level_mode_to_machine_precision():
    # Review Focus 5, and the cheapest witness that phi is right: phi is 1 on M/M/1 --
    # algebraically exactly, and to within one ulp in floating point, since the
    # cancellation in its closed form does not round exactly. So every capacity must
    # agree to machine precision -- and what that catches is a NON-UNIFORM error in phi:
    # eq 21 is a share rule, so a stray factor applied to every station alike cancels out
    # of the capacities by construction and is invisible here at any size.
    #
    # Measured sensitivity, so nobody reads more into rel=1e-12 than it holds: a
    # station-dependent phi deviation of at most e moves the capacities by 0.2*e to
    # 0.5*e (0.23 with one station off, 0.52 with one up and one down by e) -- the share
    # rule ATTENUATES a non-uniform error rather than amplifying it. So this assertion
    # first fails around e = 1e-11 and a non-uniform error of 1e-12 or below slips
    # through. That is the intended reach: under any plausible defect in the slope arm
    # phi is wrong here by a factor, not by an ulp.
    def net(mode):
        return [
            GG1Station.mm1(0.6, 1.5, 1.0, c=2.0, name="a", zeta_mode=mode),
            GG1Station.mm1(1.2, 3.0, 1.5, c=0.5, name="b", zeta_mode=mode),
            GG1Station.mm1(0.9, 2.0, 0.2, c=1.0, name="c", zeta_mode=mode),
        ]

    level = net(ZETA_LEVEL)
    slope = net(ZETA_SLOPE)
    C = 3.0 * min_feasible_budget(level)
    r_l = Optimizer(level, C).run()
    r_s = Optimizer(slope, C).run()
    for a, b in zip(r_s.capacities, r_l.capacities):
        assert a == pytest.approx(b, rel=1e-12)
    assert r_s.objective == pytest.approx(r_l.objective, rel=1e-12)
    assert r_s.iterations == r_l.iterations


def test_a_slope_run_converges_on_a_network_with_a_very_small_phi():
    # Review Focus 4, end to end. A cov = 0 station's phi falls toward 1 - rho, so its
    # converged zeta is orders of magnitude below its level value while its neighbours'
    # are not. default_zeta is unchanged at 1.0, so the run starts far from the answer
    # and must still converge rather than stall at max_iter.
    #
    # Measured, not predicted: at this budget the allocation starves the cov=0 station
    # until it sits at rho = 1 - 1e-10, where phi = 1 - rho is itself ~8.7e-11 and the
    # resulting zeta is ~7.6e-21 -- "a few percent" is off by nine orders of magnitude.
    # All four assertions below hold at that measured point.
    stations = [
        GG1Station(0.6, 1.0, 1.0, c=1.0, cov_a=0.0, cov_s=0.0, name="dd1",
                   zeta_mode=ZETA_SLOPE),
        GG1Station.mm1(1.2, 3.0, 1.0, c=0.5, name="mm1", zeta_mode=ZETA_SLOPE),
    ]
    res = Optimizer(stations, 1.05 * min_feasible_budget(stations)).run()
    assert res.converged, res.stop_reason
    assert all(z > 0.0 for z in res.zeta)
    assert 0.0 < res.zeta_phi[0] < 0.5


from qopt.analyzer import Analyzer, Evaluation


class _ScaledAnalyzer(Analyzer):
    """Reports each station's analytic E[T] scaled by a factor: a stand-in for a
    simulator whose measurement disagrees with the station's model."""

    is_stochastic = False

    def __init__(self, factor):
        self.factor = factor

    def evaluate(self, stations, S, *, fresh_seed=False):
        return Evaluation(
            sojourn_times=[st.sojourn_time(Si) * self.factor
                           for st, Si in zip(stations, S)],
        )


def test_the_cross_check_is_silent_when_measurement_matches_the_model():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning becomes a failure
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(1.10)).run()
    assert res.zeta_shape_flags == []           # 10% is inside the 25% default


def test_the_cross_check_fires_above_the_tolerance_and_names_the_station():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_LEVEL)
    C = 4.0 * min_feasible_budget(stations)
    with pytest.warns(RuntimeWarning, match="md1"):
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0)).run()
    assert len(res.zeta_shape_flags) == 1       # only the SLOPE station is checked
    assert "md1" in res.zeta_shape_flags[0]
    assert "cov" in res.zeta_shape_flags[0]     # the message points at the likely cause


def test_the_shape_tolerance_is_a_relative_disagreement_not_an_absolute_time():
    # The check compares `abs(T / T_model - 1.0)`, a FRACTION. Swapping that for
    # `abs(T - T_model)`, a TIME, leaves every other cross-check test green: their
    # stations have E[T] of order 0.3-1.0, where 0.25 happens to fall the same way
    # under either reading.
    #
    # This is _mixed_pair with gamma AND mu scaled by 100. Scaling mu alone would not
    # do it -- the budget is the same multiple of a floor that scales with mu, so S
    # absorbs the change and S*mu, rho and E[T] all come out unmoved. Scaling both
    # leaves every rho and every capacity ANALYTICALLY unmoved while dividing E[T] by
    # 100 -- measured, they agree to 1-2 ulp rather than bitwise. Three separate roundings
    # contribute, so do not attribute it to one: gamma/mu itself moves (0.6/1.5 is
    # 0.39999999999999997 where 60.0/150.0 is exactly 0.4), which shifts the floor and
    # so the budget and slack BEFORE eq 21 runs; phi moves an ulp at the same S; and
    # eq 21's share weights divide by a sqrt(mu) 100x larger. Nothing here rests on any
    # of that: the test needs only the flag and the small absolute difference below. So
    # _ScaledAnalyzer(4.0) injects the same 300% disagreement a relative tolerance must
    # still fire on, while the absolute difference shrinks to ~0.016.
    stations = [
        GG1Station.md1(60.0, 150.0, c=2.0, name="md1", zeta_mode=ZETA_SLOPE),
        GG1Station.mm1(120.0, 300.0, c=0.5, name="mm1", zeta_mode=ZETA_LEVEL),
    ]
    C = 4.0 * min_feasible_budget(stations)
    with pytest.warns(RuntimeWarning, match="md1"):
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0)).run()
    assert len(res.zeta_shape_flags) == 1
    # Anti-vacuity, and the whole point: under an ABSOLUTE reading the slope station
    # would not have been flagged at all. Without this the test would pass under the
    # very mutation it exists to kill, so do NOT delete it.
    st, Si, T = stations[0], res.capacities[0], res.sojourn_times[0]
    assert abs(T - st.sojourn_time(Si)) < ZETA_SHAPE_TOL, st.name


def test_the_cross_check_warns_once_per_station_not_once_per_iteration():
    # The condition it detects is a constructor argument: it cannot heal between
    # iterations, so repeating the warning for every one of them is noise.
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0)).run()
    shape = [w for w in caught if "analytic" in str(w.message)]
    assert len(shape) == 2                      # two slope stations, one warning each
    assert len(res.zeta_shape_flags) == 2
    # Guards against the test being vacuous: if the loop ran no more iterations than it
    # emitted warnings, it proves nothing about per-iteration repetition. If this fires,
    # lower `damping` to force more iterations -- do NOT delete the assertion.
    assert res.iterations > len(shape), res.iterations


def test_the_cross_check_is_vacuous_on_the_analytic_path():
    # AnalyticAnalyzer returns st.sojourn_time(Si), and the check recomputes exactly
    # that at the same S, so the ratio is 1.0 and this can never fire.
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 1.01 * min_feasible_budget(stations)    # a tight budget, where E[T] is largest
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Optimizer(stations, C).run()
    assert res.zeta_shape_flags == []


def test_a_shape_flag_does_not_make_a_strict_run_raise():
    # A cov_a mismatch is a MODEL-SPECIFICATION signal, not a simulation-quality one.
    # Routing it through `degraded` would make strict=True abort a run whose simulation
    # was fine, so it gets its own field.
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    with pytest.warns(RuntimeWarning):
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0), strict=True).run()
    assert res.zeta_shape_flags
    assert res.degraded == []


def test_the_cross_check_can_be_retuned_or_disabled():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    # A tighter tolerance fires on a disagreement the default tolerates.
    with pytest.warns(RuntimeWarning):
        Optimizer(stations, C, analyzer=_ScaledAnalyzer(1.10), zeta_shape_tol=0.05).run()
    # None disables it entirely.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Optimizer(_mixed_pair(ZETA_SLOPE, ZETA_SLOPE), C,
                        analyzer=_ScaledAnalyzer(4.0), zeta_shape_tol=None).run()
    assert res.zeta_shape_flags == []


def test_a_level_only_run_never_evaluates_the_cross_check():
    # The default path must not start computing an analytic sojourn time it did not
    # need before.
    class BrokenAnalytic(GG1Station):
        calls = 0

        def sojourn_time(self, S):
            type(self).calls += 1
            return super().sojourn_time(S)

    stations = [BrokenAnalytic(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=0.0, name="lvl")]
    C = 4.0 * min_feasible_budget(stations)
    res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0)).run()
    assert res.zeta_shape_flags == []
    # Only the analyzer's own calls, one per iteration plus the final evaluation.
    assert BrokenAnalytic.calls == res.iterations + 1


def test_an_invalid_shape_tolerance_is_rejected():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    for bad in (-0.1, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="zeta_shape_tol"):
            Optimizer(stations, C, zeta_shape_tol=bad)


def test_the_finite_difference_survives_a_zero_width_step():
    """`h` can be exactly 0.0 on a station that is genuinely stable, which used to CRASH.

    The step scales to spare capacity, `h = _FD_STEP * (S - gamma/mu)`, and the stability
    test is `S * mu > gamma`. Those are different expressions, so they disagree at the
    boundary: for gamma=0.1, mu=0.39 the capacity `S = gamma/mu` satisfies `S * mu > gamma`
    -- x = 1.4e-17 -- while `S - gamma/mu` is exactly 0.0. The station is stable, E[T] is
    finite, and the step width is zero.

    Pre-fix that divided by `2.0 * h == 0.0` and raised ZeroDivisionError: a bare arithmetic
    crash from inside a derivative, naming no station and suggesting no cause. Widening turns
    it into the one-sided difference the number line can actually support. This is not an
    exotic hand-picked pair -- sweeping gamma in 0.1 steps and mu in 0.13 steps over a 59x59
    grid, 148 (gamma, mu) pairs put `gamma/mu` on a stable-but-zero-width capacity, and every
    one of them crashed before and returns a finite negative slope now.

    The kill set: restoring `/ (2.0 * h)` with `S ± h` endpoints raises ZeroDivisionError
    here. Restoring only the divisor, keeping widened endpoints, also raises it -- which is
    the point, since `hi - lo` is what makes the widened endpoints usable.
    """
    st = QuadraticStation(gamma=0.1, mu=0.39)
    S = st.gamma / st.mu

    assert S * st.mu > st.gamma                  # stable, so a derivative is owed
    assert S - st.gamma / st.mu == 0.0           # yet the scaled step has zero width
    assert _FD_STEP * (S - st.gamma / st.mu) == 0.0

    slope = st.dT_dS(S)
    assert math.isfinite(slope) and slope < 0.0, slope

    # phi is NOT checked against 2 here. One-sided over a single ulp at x = 1.4e-17 is the
    # coarsest resolution in the whole domain, and phi measures far from 2; the contract at
    # this capacity is that it returns a usable sign at all instead of crashing.
    assert st.zeta_from(st.sojourn_time(S), S) > 0.0

    # Every one of the 148 grid pairs, not just this one.
    survived = 0
    for gi in range(1, 60):
        for mi in range(1, 60):
            gamma, mu = gi * 0.1, mi * 0.13
            b = gamma / mu
            if not b * mu > gamma:
                continue
            other = QuadraticStation(gamma=gamma, mu=mu)
            d = other.dT_dS(b)
            assert math.isfinite(d) and d < 0.0, (gamma, mu, d)
            survived += 1
    assert survived == 148, survived
