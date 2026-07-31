import math

import pytest

from qopt.allocator import min_feasible_budget
from qopt.analyzer import AnalyticAnalyzer, Analyzer, Evaluation
from qopt.network import Network, Route
from qopt.optimizer import Optimizer, Result
from qopt.station import ForkJoinStation, GG1Station


def _stations():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


def _network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0)


class DeterministicFake(Analyzer):
    """Mirrors sojourn_time exactly, but declares itself stochastic.

    Lets the stochastic code path (warm start, damping, sim_calls, final evaluation) be
    exercised with zero randomness, so equivalence can be asserted bitwise.
    """

    is_stochastic = True

    def __init__(self, half_width=None):
        self.half_width = half_width
        self.calls = 0
        self.fresh_calls = 0

    def evaluate(self, stations, S, *, fresh_seed=False):
        self.calls += 1
        if fresh_seed:
            self.fresh_calls += 1
        T = [st.sojourn_time(Si) for st, Si in zip(stations, S)]
        ci = None
        if self.half_width is not None:
            ci = [(t - self.half_width, t + self.half_width) for t in T]
        return Evaluation(sojourn_times=T, ci=ci,
                          extras={"system_response_time": (sum(T), (0.0, 1.0))})


BUDGET = 15.600000000000001
LEGACY_S = [2.9601176145885644, 3.644844988735743, 3.017459891043565]
LEGACY_OBJECTIVE = 1.1669333832717816


# --- backward compatibility --------------------------------------------------

def test_default_construction_is_analytic_and_undamped():
    opt = Optimizer(_stations(), budget=BUDGET)
    assert isinstance(opt.analyzer, AnalyticAnalyzer)
    assert opt.damping == 1.0
    assert opt.max_iter == 1000


def test_analytic_defaults_reproduce_the_legacy_numbers_bitwise():
    result = Optimizer(_stations(), budget=BUDGET).run()
    assert result.capacities == LEGACY_S
    assert result.objective == LEGACY_OBJECTIVE
    assert result.iterations == 6
    assert result.converged is True
    assert result.stop_reason == "tol"
    assert result.sojourn_ci is None
    assert result.noise_floor is None
    assert result.warm_start_iterations == 0
    assert result.degraded == []
    assert result.sim_calls == 0


def test_result_new_fields_all_default():
    r = Result(capacities=[1.0], sojourn_times=[1.0], zeta=[1.0], objective=1.0,
               iterations=1, converged=True, residual=0.0)
    assert r.sojourn_ci is None
    assert r.noise_floor is None
    assert r.stop_reason == "tol"
    assert r.warm_start_iterations == 0
    assert r.degraded == []
    assert r.system_response_time is None
    assert r.sim_calls == 0


def test_result_degraded_default_is_per_instance():
    a = Result(capacities=[], sojourn_times=[], zeta=[], objective=0.0,
               iterations=0, converged=True, residual=0.0)
    b = Result(capacities=[], sojourn_times=[], zeta=[], objective=0.0,
               iterations=0, converged=True, residual=0.0)
    a.degraded.append("x")
    assert b.degraded == []


# --- Network as the first argument -------------------------------------------

def test_optimizer_accepts_a_network():
    network = _network()
    opt = Optimizer(network, budget=BUDGET)
    assert opt.network is network
    assert opt.stations == network.stations
    assert opt.run().capacities == LEGACY_S


def test_optimizer_still_accepts_a_bare_station_sequence():
    opt = Optimizer(_stations(), budget=BUDGET)
    assert opt.network is None
    assert len(opt.stations) == 3


# --- naive equivalence (spec 8, 6.6) ----------------------------------------

NAIVE_KNOBS = dict(warm_start=False, damping=1.0, noise_kappa=0.0, max_iter=1000)


def test_naive_equivalence_is_bit_identical():
    baseline = Optimizer(_stations(), budget=BUDGET).run()
    fake = DeterministicFake()
    simulated = Optimizer(
        _stations(), budget=BUDGET, analyzer=fake, **NAIVE_KNOBS
    ).run()
    assert simulated.capacities == baseline.capacities
    assert simulated.sojourn_times == baseline.sojourn_times
    assert simulated.zeta == baseline.zeta
    assert simulated.objective == baseline.objective
    assert simulated.iterations == baseline.iterations
    assert simulated.residual == baseline.residual
    assert simulated.converged == baseline.converged
    # One POST per iteration, plus the final evaluation (spec 6.3 cost model).
    assert simulated.sim_calls == simulated.iterations + 1
    assert fake.fresh_calls == 1


def test_naive_equivalence_without_a_final_evaluation():
    baseline = Optimizer(_stations(), budget=BUDGET).run()
    fake = DeterministicFake()
    simulated = Optimizer(
        _stations(), budget=BUDGET, analyzer=fake, final_evaluation=False, **NAIVE_KNOBS
    ).run()
    assert simulated.capacities == baseline.capacities
    assert simulated.iterations == baseline.iterations
    assert simulated.residual == baseline.residual
    assert simulated.sim_calls == simulated.iterations
    assert fake.fresh_calls == 0


# --- warm start --------------------------------------------------------------

def test_warm_start_costs_zero_simulation_calls_and_is_counted_separately():
    fake = DeterministicFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake, damping=1.0,
                       noise_kappa=0.0, max_iter=5).run()
    assert result.warm_start_iterations == 6        # the analytic pre-solve
    assert result.sim_calls == result.iterations + 1
    assert result.capacities == pytest.approx(LEGACY_S, rel=1e-9)


def test_warm_start_starts_the_loop_at_the_analytic_answer():
    fake = DeterministicFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake, damping=1.0,
                       noise_kappa=0.0, max_iter=20).run()
    # Already converged before the first simulated iteration, so it stops immediately.
    assert result.iterations == 1
    assert result.stop_reason == "tol"


def test_warm_start_off_skips_the_pre_solve():
    fake = DeterministicFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake,
                       **NAIVE_KNOBS).run()
    assert result.warm_start_iterations == 0


# --- damping -----------------------------------------------------------------

def test_stochastic_defaults_are_damped_and_capped():
    opt = Optimizer(_stations(), budget=BUDGET, analyzer=DeterministicFake())
    assert opt.damping == 0.5
    assert opt.max_iter == 20


def test_damping_slows_movement_but_reaches_the_same_point():
    result = Optimizer(_stations(), budget=BUDGET, analyzer=DeterministicFake(),
                       warm_start=False, damping=0.5, noise_kappa=0.0,
                       max_iter=500, tol=1e-12).run()
    assert result.capacities == pytest.approx(LEGACY_S, rel=1e-9)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_damping_validated(bad):
    with pytest.raises(ValueError, match="damping"):
        Optimizer(_stations(), budget=BUDGET, damping=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_noise_kappa_validated(bad):
    with pytest.raises(ValueError, match="noise_kappa"):
        Optimizer(_stations(), budget=BUDGET, noise_kappa=bad)


# --- CI-aware stopping (spec 6.4) -------------------------------------------

def test_stop_reason_flips_to_noise_floor_as_ci_widens():
    narrow = Optimizer(_stations(), budget=BUDGET,
                       analyzer=DeterministicFake(half_width=1e-15),
                       warm_start=False, damping=1.0, max_iter=200).run()
    assert narrow.stop_reason == "tol"

    wide = Optimizer(_stations(), budget=BUDGET,
                     analyzer=DeterministicFake(half_width=0.05),
                     warm_start=False, damping=1.0, max_iter=200).run()
    assert wide.stop_reason == "noise-floor"
    assert wide.noise_floor > wide.residual
    assert wide.converged is True


def test_threshold_scales_by_damping_so_kappa_means_what_it_says():
    # A DISCRIMINATING case for the theta scaling, not merely a consistent one. At
    # damping=0.5 with half_width=0.02 the first step is residual = 0.098019 while the
    # noise floor is 0.137007, so the two candidate thresholds straddle it:
    #   scaled   kappa*theta*floor = 0.068504  ->  0.098019 >= it, the loop keeps going
    #   unscaled kappa*floor       = 0.137007  ->  0.098019 <  it, the loop would stop
    # So reverting the scaling flips this run from a max_iter exit to a noise-floor stop.
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = Optimizer(_stations(), budget=BUDGET,
                           analyzer=DeterministicFake(half_width=0.02),
                           warm_start=False, damping=0.5, noise_kappa=1.0,
                           max_iter=1).run()
    assert result.noise_floor == pytest.approx(0.137007, rel=1e-3)
    assert result.residual == pytest.approx(0.098019, rel=1e-3)
    # The scaled threshold sits below the residual and the unscaled one above it.
    assert 1.0 * 0.5 * result.noise_floor < result.residual < 1.0 * result.noise_floor
    assert result.stop_reason == "max_iter"     # unscaled would report "noise-floor"
    assert result.converged is False


def test_stop_reason_is_tol_when_tol_was_met_despite_a_wide_noise_floor():
    # A DISCRIMINATING case for the labelling: residual < tol < kappa*theta*floor.
    # Warm-starting lands the loop on the analytic fixed point, so iteration 1's step is
    # ~1.5e-11, while a wide CI puts the noise threshold at ~0.35. The run met `tol`
    # outright and must say so; labelling it "noise-floor" would tell a caller to buy
    # simulation replications they do not need. Labelling by which term won the max()
    # reports "noise-floor" here, so this test fails if the fix is reverted.
    result = Optimizer(_stations(), budget=BUDGET,
                       analyzer=DeterministicFake(half_width=0.05),
                       warm_start=True, damping=1.0, max_iter=20).run()
    assert result.iterations == 1
    assert result.residual < 1e-9               # tol was genuinely met
    assert result.noise_floor > 1e-9            # and the floor was the larger term
    assert result.stop_reason == "tol"


def test_noise_floor_is_computed_at_the_capacities_the_ci_was_measured_at():
    # Guards the loop's ordering: the floor must use the S handed to evaluate(), not the
    # already-damped S_new. Swapping those is invisible at damping=1.0, so damp here.
    from qopt.allocator import noise_floor as nf

    seen = []

    class RecordingFake(DeterministicFake):
        def evaluate(self, stations, S, *, fresh_seed=False):
            seen.append(list(S))
            return super().evaluate(stations, S, fresh_seed=fresh_seed)

    # max_iter=1 cannot converge, so the non-convergence warning is expected here.
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = Optimizer(_stations(), budget=BUDGET,
                           analyzer=RecordingFake(half_width=0.02),
                           warm_start=False, damping=0.5, max_iter=1).run()

    stations = _stations()
    S_eval = seen[0]                            # the S the single loop iteration evaluated
    ev = DeterministicFake(half_width=0.02).evaluate(stations, S_eval)
    zeta = [st.zeta_from(T, Si) for st, T, Si in zip(stations, ev.sojourn_times, S_eval)]
    dzeta = [0.5 * (hi - lo) * (Si * st.mu - st.gamma)
             for st, Si, (lo, hi) in zip(stations, S_eval, ev.ci)]
    assert result.noise_floor == nf(stations, BUDGET, zeta, dzeta)
    assert result.capacities != S_eval          # the returned S really is the damped one


def test_kappa_zero_restores_naive_stopping():
    result = Optimizer(_stations(), budget=BUDGET,
                       analyzer=DeterministicFake(half_width=0.05),
                       warm_start=False, damping=1.0, noise_kappa=0.0,
                       max_iter=200).run()
    assert result.stop_reason == "tol"
    assert result.noise_floor is None


def test_ci_and_system_response_time_reach_the_result():
    result = Optimizer(_stations(), budget=BUDGET,
                       analyzer=DeterministicFake(half_width=0.01),
                       warm_start=False, damping=1.0, max_iter=200).run()
    assert len(result.sojourn_ci) == 3
    for (lo, hi), t in zip(result.sojourn_ci, result.sojourn_times):
        assert lo < t < hi
    assert result.system_response_time is not None


# --- degraded accounting and strict -----------------------------------------

class DegradingFake(DeterministicFake):
    def evaluate(self, stations, S, *, fresh_seed=False):
        ev = super().evaluate(stations, S, fresh_seed=fresh_seed)
        ev.degraded.append(f"call {self.calls}: synthetic degradation")
        return ev


def test_degraded_entries_accumulate_per_iteration():
    fake = DegradingFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake, warm_start=False,
                       damping=1.0, noise_kappa=0.0, max_iter=1000).run()
    assert len(result.degraded) == fake.calls
    assert all("synthetic degradation" in d for d in result.degraded)


def test_strict_raises_at_the_end_with_the_whole_audit_trail():
    from qopt.exceptions import SimulationQualityError

    with pytest.raises(SimulationQualityError, match="synthetic degradation"):
        Optimizer(_stations(), budget=BUDGET, analyzer=DegradingFake(),
                  warm_start=False, damping=1.0, noise_kappa=0.0, strict=True,
                  max_iter=1000).run()


# --- caveat warnings (spec 6.6) ---------------------------------------------

class FixedSeedFake(DeterministicFake):
    seed_policy = "fixed"


def test_fixed_seed_without_a_final_evaluation_warns():
    with pytest.warns(RuntimeWarning, match="common random numbers"):
        Optimizer(_stations(), budget=BUDGET, analyzer=FixedSeedFake(),
                  final_evaluation=False)


def test_fixed_seed_with_a_final_evaluation_does_not_warn(recwarn):
    Optimizer(_stations(), budget=BUDGET, analyzer=FixedSeedFake())
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_max_iter_exhaustion_still_warns_and_reports_max_iter():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = Optimizer(_stations(), budget=BUDGET,
                           analyzer=DeterministicFake(), warm_start=False,
                           damping=1.0, noise_kappa=0.0, tol=0.0, max_iter=3).run()
    assert result.stop_reason == "max_iter"
    assert result.converged is False
    assert math.isfinite(result.residual)
