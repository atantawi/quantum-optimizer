"""Measured arrival cov_a: station arithmetic and optimizer plumbing.

Spec: docs/superpowers/specs/2026-09-25-measured-arrival-cov-design.md
"""

import math

import pytest

from qopt.station import ForkJoinStation, GG1Station
from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE


def _gg1(cov_a, mode=ZETA_SLOPE, cov_s=0.5):
    return GG1Station(0.6, 1.5, c=2.0, cov_a=cov_a, cov_s=cov_s, name="g", zeta_mode=mode)


S_GRID = [0.5, 0.8, 1.2, 3.0]          # gamma/mu = 0.4, so every S here is stable


# --- station arithmetic -------------------------------------------------------

@pytest.mark.parametrize("S", S_GRID)
@pytest.mark.parametrize("c", [0.0, 0.3, 1.0, 2.5])
def test_a_measured_cov_a_prices_exactly_like_a_station_built_with_it(S, c):
    # Exact, not approx: the override must run the SAME expression on the measured value,
    # so a station built with cov_a=c is its bit-for-bit reference.
    built, measured = _gg1(c), _gg1(1.7)
    assert measured.sojourn_time(S, cov_a=c) == built.sojourn_time(S)
    assert measured.dT_dS(S, cov_a=c) == built.dT_dS(S)
    assert measured.phi(S, cov_a=c) == built.phi(S)
    assert measured.zeta_from(0.9, S, cov_a=c) == built.zeta_from(0.9, S)


def test_the_override_moves_slope_zeta():
    # Anti-vacuity for the test above: if the override were ignored, 1.7 and 0.2 would
    # still agree there only by coincidence -- they must not.
    st = _gg1(1.7)
    for S in S_GRID:
        assert st.zeta_from(0.9, S, cov_a=0.2) != st.zeta_from(0.9, S)


def test_the_override_does_not_persist_on_the_station():
    st = _gg1(1.7)
    before = [st.phi(S) for S in S_GRID]
    for S in S_GRID:
        st.phi(S, cov_a=0.2)
        st.zeta_from(0.9, S, cov_a=0.2)
        st.sojourn_time(S, cov_a=0.2)
    assert st.cov_a == 1.7
    assert [st.phi(S) for S in S_GRID] == before


def test_the_level_arm_ignores_the_override():
    st = _gg1(1.7, mode=ZETA_LEVEL)
    for S in S_GRID:
        x = S * st.mu - st.gamma
        assert st.zeta_from(0.9, S, cov_a=0.2) == 0.9 * x


def test_a_measured_zero_cov_takes_the_deterministic_branch():
    # cov_s = 0 and a measured cov_a = 0 give k == 0 from the OVERRIDE, although the
    # constructor k is 0.5. phi must then be exactly the k == 0 value, 1 - rho.
    st = _gg1(1.0, cov_s=0.0)
    for S in S_GRID:
        rho = st.gamma / (S * st.mu)
        assert st.phi(S, cov_a=0.0) == pytest.approx(1.0 - rho, rel=1e-12)
        assert st.sojourn_time(S, cov_a=0.0) == 1.0 / (S * st.mu)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, -math.inf])
def test_an_invalid_override_is_rejected_like_the_constructor_argument(bad):
    st = _gg1(1.7)
    with pytest.raises(ValueError, match="cov_a must be a finite number >= 0"):
        st.phi(1.0, cov_a=bad)
    with pytest.raises(ValueError, match="cov_a must be a finite number >= 0"):
        st.sojourn_time(1.0, cov_a=bad)


def test_a_fork_join_station_ignores_the_override():
    fj = ForkJoinStation(0.5, 1.0, r=2.0, c1=1.0, c2=1.0, name="fj", zeta_mode=ZETA_SLOPE)
    for S in [0.8, 1.5, 3.0]:
        assert fj.phi(S, cov_a=0.3) == fj.phi(S)
        assert fj.zeta_from(0.9, S, cov_a=0.3) == fj.zeta_from(0.9, S)


def test_only_a_slope_station_that_reads_cov_a_uses_a_measurement():
    assert GG1Station.reads_arrival_cov is True
    assert ForkJoinStation.reads_arrival_cov is False
    assert _gg1(1.0).uses_measured_cov_a is True
    assert _gg1(1.0, mode=ZETA_LEVEL).uses_measured_cov_a is False
    fj = ForkJoinStation(0.5, 1.0, r=2.0, c1=1.0, c2=1.0, zeta_mode=ZETA_SLOPE)
    assert fj.uses_measured_cov_a is False


def test_zeta_from_does_not_pass_the_keyword_when_there_is_no_measurement():
    # A user subclass written before the keyword existed must keep working on every
    # path that never measures.
    class Legacy(GG1Station):
        def phi(self, S):
            return super().phi(S)

    st = Legacy(0.6, 1.5, c=2.0, cov_a=1.7, cov_s=0.5, zeta_mode=ZETA_SLOPE)
    assert st.zeta_from(0.9, 1.0) == _gg1(1.7).zeta_from(0.9, 1.0)


# --- optimizer plumbing ---------------------------------------------------------

import warnings

from qopt.allocator import min_feasible_budget, noise_floor
from qopt.analyzer import Analyzer, Evaluation
from qopt.optimizer import Optimizer

# zeta_shape_tol=None keeps these plumbing tests independent of Task 5's check, which
# (before Task 5) would compare against the constructor model and warn.
NAIVE = dict(warm_start=False, damping=1.0, noise_kappa=0.0, max_iter=1000,
             zeta_shape_tol=None)


def _pair(md1_cov_a=1.0):
    # md1 is cov_s = 0 with constructor cov_a = 1 (k = 0.5); a measured 0.3 gives k = 0.045,
    # far enough that phi -- and so the fixed point -- moves.
    return [
        GG1Station(0.6, 1.5, c=2.0, cov_a=md1_cov_a, cov_s=0.0, name="md1",
                   zeta_mode=ZETA_SLOPE),
        GG1Station.mm1(1.2, 3.0, c=0.5, name="mm1", zeta_mode=ZETA_SLOPE),
    ]


C = 2.0 * min_feasible_budget(_pair())


class CovFake(Analyzer):
    """A stochastic-declared analyzer whose E[T] is the station's model AT the measured
    cov_a, and which reports that cov_a: a simulator whose measurement is exactly right.

    `loop` is returned by every non-final evaluation (or `script[i]` for the i-th, the last
    entry repeating), `final` by the fresh-seeded one, so a test can tell which evaluation
    a Result field came from. `factor` scales E[T], a stand-in for a model-shape error.
    """

    is_stochastic = True

    def __init__(self, loop=None, final="same", *, script=None, half_width=None,
                 factor=1.0):
        self.loop, self.script = loop, script
        self.final = loop if final == "same" else final
        self.half_width, self.factor = half_width, factor
        self.calls, self.seen = 0, []

    def evaluate(self, stations, S, *, fresh_seed=False):
        if fresh_seed:
            cov = self.final
        elif self.script is not None:
            cov = self.script[min(self.calls, len(self.script) - 1)]
        else:
            cov = self.loop
        if not fresh_seed:
            self.calls += 1
            self.seen.append(list(S))
        covs = [None] * len(stations) if cov is None else cov
        T = [(st.sojourn_time(Si) if c is None else st.sojourn_time(Si, cov_a=c))
             * self.factor for st, Si, c in zip(stations, S, covs)]
        ci = None if self.half_width is None else [
            (t - self.half_width, t + self.half_width) for t in T]
        return Evaluation(sojourn_times=T, ci=ci,
                          arrival_cov=None if cov is None else list(cov))


def test_a_measured_run_is_the_analytic_run_of_the_measured_model():
    # Bitwise: the override path runs the same arithmetic as a station built with the
    # measured value (Task 1), so the whole loop must be identical. Fails if the LOOP zeta
    # (point 1) or the FINAL zeta/phi (point 4) ignores the measurement.
    measured = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None]), **NAIVE).run()
    reference = Optimizer(_pair(md1_cov_a=0.3), C,
                          analyzer=CovFake(loop=None), **NAIVE).run()
    assert measured.capacities == reference.capacities
    assert measured.iterations == reference.iterations
    assert measured.zeta == reference.zeta
    assert measured.zeta_phi == reference.zeta_phi
    # Anti-vacuity: the measurement really moved the answer.
    unmeasured = Optimizer(_pair(), C, analyzer=CovFake(loop=None), **NAIVE).run()
    assert measured.capacities != unmeasured.capacities


def test_result_fields_come_from_the_final_evaluation():
    res = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None], final=[0.8, None]),
                    **NAIVE).run()
    st, S0, T0 = _pair()[0], res.capacities[0], res.sojourn_times[0]
    assert res.arrival_cov == [0.8, None]
    assert res.zeta[0] == st.zeta_from(T0, S0, cov_a=0.8)
    assert res.zeta_phi[0] == st.phi(S0, cov_a=0.8)
    assert res.zeta_phi[0] != st.phi(S0, cov_a=0.3)        # a mix-up would be visible


def test_without_a_final_evaluation_the_last_loop_measurement_is_reported():
    res = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None], final=[0.8, None]),
                    final_evaluation=False, **NAIVE).run()
    assert res.arrival_cov == [0.3, None]


def test_a_missing_final_measurement_is_not_filled_from_the_loop():
    # Per evaluation, never cached: a station unmeasured in the final evaluation is priced
    # at its constructor cov_a there, even though the loop measured it.
    res = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None], final=[None, None]),
                    **NAIVE).run()
    st, S0, T0 = _pair()[0], res.capacities[0], res.sojourn_times[0]
    assert res.arrival_cov == [None, None]
    assert res.zeta[0] == st.zeta_from(T0, S0)


def test_a_measurement_that_vanishes_mid_run_is_priced_at_the_constructor_then():
    # Iteration 2 loses md1's measurement. Its capacities must be those of a run that
    # measured nothing at iteration 2, which a cache of the last good value would break.
    script = [[0.3, None], [None, None], [0.3, None]]
    fake = CovFake(script=script)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        Optimizer(_pair(), C, analyzer=fake, **dict(NAIVE, max_iter=3)).run()
    S2, S3 = fake.seen[1], fake.seen[2]
    # S3 = eq 21 at the zeta of iteration 2, which must be the UNMEASURED zeta at S2.
    from qopt.allocator import allocate
    st = _pair()
    zeta2 = [s.zeta_from(s.sojourn_time(Si), Si) for s, Si in zip(st, S2)]
    assert S3 == allocate(st, C, zeta2)


def test_the_noise_floor_goes_through_the_same_override():
    fake = CovFake(loop=[0.3, None], half_width=0.02)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = Optimizer(_pair(), C, analyzer=fake, warm_start=False, damping=0.5,
                        max_iter=1, zeta_shape_tol=None).run()
    st, S = _pair(), fake.seen[0]
    ev = CovFake(loop=[0.3, None], half_width=0.02).evaluate(st, S)
    zeta = [st[0].zeta_from(ev.sojourn_times[0], S[0], cov_a=0.3),
            st[1].zeta_from(ev.sojourn_times[1], S[1])]
    dzeta = [st[0].zeta_from(0.02, S[0], cov_a=0.3), st[1].zeta_from(0.02, S[1])]
    assert res.noise_floor == noise_floor(st, C, zeta, dzeta)
    unmeasured = [st[0].zeta_from(0.02, S[0]), dzeta[1]]
    assert res.noise_floor != noise_floor(st, C, zeta, unmeasured)   # anti-vacuity


def test_nothing_measured_leaves_arrival_cov_empty():
    assert Optimizer(_pair(), C).run().arrival_cov == []
    assert Optimizer(_pair(), C, analyzer=CovFake(loop=None), **NAIVE).run().arrival_cov == []


def test_an_old_signature_phi_override_survives_a_measured_run():
    class Legacy(GG1Station):
        reads_arrival_cov = False               # a user type qopt knows nothing about
        def phi(self, S):
            return super().phi(S)

    stations = [
        Legacy(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=0.0, name="legacy", zeta_mode=ZETA_SLOPE),
        GG1Station(1.2, 3.0, c=0.5, cov_a=1.0, cov_s=0.0, name="g", zeta_mode=ZETA_SLOPE),
    ]
    budget = 2.0 * min_feasible_budget(stations)
    res = Optimizer(stations, budget, analyzer=CovFake(loop=[None, 0.3]), **NAIVE).run()
    assert res.arrival_cov == [None, 0.3]
