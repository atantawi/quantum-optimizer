import math

import pytest

from qopt.allocator import min_feasible_budget
from qopt.exceptions import InfeasibleBudgetError
from qopt.analyzer import AnalyticAnalyzer, Analyzer, Evaluation
from qopt.network import Network, Route
from qopt.optimizer import Optimizer, Result
from qopt.forkjoin_policy import (
    R_STAR_EQUAL_RATE,
    R_STAR_INVARIANT_R,
    R_STAR_TUNED,
    optimal_ray,
)
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


def test_noise_threshold_compares_a_target_space_step_so_kappa_means_what_it_says():
    # Guards the kappa arm against comparing a DAMPED step to a target-space floor. At
    # damping=0.5 with half_width=0.02 the first iteration gives residual = 0.098019, so
    # the target-space step is 0.196038, while the noise floor is 0.137007:
    #   normalized  step 0.196038 vs kappa*floor 0.137007 -> >= it, the loop keeps going
    #   un-normalized residual 0.098019 vs the same 0.137007 -> < it, it would stop
    # So dropping the `/ self.damping` flips this run from a max_iter exit to a
    # noise-floor stop, and the expected "did not converge" warning never fires.
    #
    # It does NOT discriminate this form from the interim `max(tol, kappa*theta*floor)`
    # one: those two are algebraically identical on this arm (theta*d < kappa*theta*f is
    # d < kappa*f), so there is no behavioural difference for any test to catch. What
    # changed between them was the `tol` arm, which the next test covers.
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = Optimizer(_stations(), budget=BUDGET,
                           analyzer=DeterministicFake(half_width=0.02),
                           warm_start=False, damping=0.5, noise_kappa=1.0,
                           max_iter=1).run()
    assert result.noise_floor == pytest.approx(0.137007, rel=1e-3)
    assert result.residual == pytest.approx(0.098019, rel=1e-3)
    # The damped residual sits below the threshold while the target-space step sits above
    # it — the whole point of normalizing rather than comparing the iterate's movement.
    step = result.residual / 0.5
    assert result.residual < 1.0 * result.noise_floor < step
    assert result.stop_reason == "max_iter"     # un-normalized would say "noise-floor"
    assert result.converged is False


def test_tol_is_a_target_space_tolerance_at_every_damping():
    # DISCRIMINATING case for `tol`, which carried the same damped-vs-target mismatch the
    # noise term did. At damping=0.5 with kappa=0 the target-space step sequence passes
    # through 8.910166e-07 on iteration 19, so with tol = 6e-07:
    #   normalized    step_19 = 8.91e-07 >= tol -> keeps going, stops on iteration 20
    #   un-normalized residual_19 = 4.46e-07 < tol -> stops on iteration 19
    # The iteration count is therefore the discriminator, and `tol` means the same thing
    # here as it does on the analytic path.
    result = Optimizer(_stations(), budget=BUDGET, analyzer=DeterministicFake(),
                       warm_start=False, damping=0.5, noise_kappa=0.0,
                       tol=6e-07, max_iter=200).run()
    assert result.converged is True
    assert result.stop_reason == "tol"
    assert result.iterations == 20          # un-normalized would stop at 19
    # The target-space step is what cleared tol. The damped residual is below tol as well
    # — by a further factor of theta — which is exactly why comparing it directly stops
    # too early, one iteration sooner here.
    assert result.residual / 0.5 < 6e-07
    assert result.residual < 6e-07


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
    dzeta = [st.zeta_from(0.5 * (hi - lo), Si)
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


def test_missing_response_time_ci_does_not_crash_the_loop(sim_response):
    # Before the finding-1 fix, _noise_floor's zip(..., ci) unpacking a None entry as
    # (lower, upper) raised TypeError mid-loop, in the default configuration
    # (noise_kappa=1.0). This proves the run COMPLETES instead: mm1's response-time
    # measure has no CI, so its dzeta contribution is 0, but md1 and fj still have
    # theirs and the loop finishes normally.
    from conftest import FakeTransport
    from qopt.qsim.analyzer import SimulationAnalyzer
    from qopt.qsim.client import QsimClient

    network = _network()
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.6, "md1": 0.4, "fj": 0.5},
        system=1.16,
    )
    for m in response["measures"]:
        if m["station"] == "mm1" and m["type"] == "response-time":
            m["lower"] = None
            m["upper"] = None
    client = QsimClient("http://qsim.test", transport=FakeTransport((200, response)))
    analyzer = SimulationAnalyzer(network, client)

    with pytest.warns(RuntimeWarning, match="mm1"):
        result = Optimizer(network, budget=BUDGET, analyzer=analyzer).run()

    assert result.sojourn_ci[0] is None                   # mm1: no CI
    assert result.sojourn_ci[1] is not None               # md1, fj: still have theirs
    assert result.sojourn_ci[2] is not None
    assert result.noise_floor is not None                 # md1 + fj still contribute
    assert result.noise_floor >= 0.0


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


# --------------------------------------------------------------------------------------
# Tuned r_star: the nested fixed point (issue #10 item 3).
# --------------------------------------------------------------------------------------

def _tuned_pair():
    """A tuned fork-join priced like quantum_dominant, plus one ordinary queue."""
    fj = ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0,
                         r_star=R_STAR_TUNED, name="fj")
    return [fj, GG1Station.mm1(gamma=0.9, mu=1.0, c=1.0, name="q")]


def test_tuned_station_reaches_a_self_consistent_fixed_point():
    """All three equations must hold at once at the answer: eq 21 prices the station at
    its own r_star, eq 22 takes zeta at the ray it runs, and r_star is the local optimum
    for the spend eq 21 gave it. The last of those is what makes it a NESTED fixed point,
    and it is what fails if the loop is allowed to converge before r_star settles.
    """
    stations = _tuned_pair()
    fj = stations[0]
    C = 4.0 * min_feasible_budget(stations)
    res = Optimizer(stations, C).run()
    assert res.converged
    spend = res.capacities[0] * fj.alloc_cost
    assert fj.r_star == pytest.approx(
        optimal_ray(0.45, 1.0, 4.0, 4.0, 1.0, spend), rel=1e-9)
    assert fj.r_star != 4.0          # it actually moved off the incumbent ray


def test_tuning_spends_the_whole_budget_exactly():
    """The budget identity has to survive a policy that reprices its own station.

    This pins the invariant at the fixed point, where it holds for either of two reasons:
    r_star has stopped moving, so eq 21's own allocation already exhausts C, AND `retune`
    returns a spend-preserving capacity. It therefore catches a wrong rescale factor but
    not a missing one -- the mechanism itself is pinned at station level, by
    test_retune_preserves_the_stations_spend_exactly.
    """
    stations = _tuned_pair()
    C = 4.0 * min_feasible_budget(stations)
    res = Optimizer(stations, C).run()
    spent = sum(st.alloc_cost * S for st, S in zip(stations, res.capacities))
    assert spent == pytest.approx(C, rel=1e-12)


def test_tuning_beats_both_incumbent_policies_at_the_same_budget():
    def objective(r_star):
        stations = _tuned_pair()
        stations[0] = ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0,
                                      r_star=r_star, name="fj")
        C = 4.0 * min_feasible_budget(_tuned_pair())
        return Optimizer(stations, C).run().objective

    tuned = objective(R_STAR_TUNED)
    assert tuned < objective(R_STAR_INVARIANT_R)
    assert tuned < objective(R_STAR_EQUAL_RATE)


def test_tuning_does_not_degrade_convergence():
    """A tuned run must not cost materially more iterations than a fixed ray.

    What this actually guards is the INNER SOLVE's precision: replacing the bisection with
    a golden-section minimizer of `t_ul` takes 9 iterations here against bisection's 6 and
    the incumbent's 5, because r* then carries sqrt(epsilon) noise that keeps the outer
    step above `tol`. That is a different mechanism from findings section 7's 9 iterations
    under the inner-split embedding, which came from a mispriced zeta on a different
    network -- the numbers coincide, the causes do not.
    """
    stations = _tuned_pair()
    C = 4.0 * min_feasible_budget(stations)
    tuned = Optimizer(stations, C).run()
    incumbent = Optimizer(
        [ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, name="fj"),
         GG1Station.mm1(gamma=0.9, mu=1.0, c=1.0, name="q")], C).run()
    assert tuned.converged and incumbent.converged
    assert tuned.iterations <= incumbent.iterations + 2


def test_tuning_survives_the_stochastic_path_with_damping_and_a_warm_start():
    """r_star does carry simulation noise -- it is a function of the station's spend, and
    on this path that spend descends from a measured E[T] (injecting +/-2% noise into E[T]
    moves the converged r_star by ~6e-4 relative). It still needs no damping of its own,
    but for a different reason: the retune adds no noise of its OWN, being an exact
    function of an S that has already been damped, so damping here would attenuate one
    perturbation twice. r_star also has to survive the analytic warm start, which shares
    these station objects and therefore tunes them before the simulated phase begins.
    """
    stations = _tuned_pair()
    fj = stations[0]
    C = 4.0 * min_feasible_budget(stations)
    fake = DeterministicFake()
    res = Optimizer(stations, C, analyzer=fake).run()
    assert res.converged
    assert res.warm_start_iterations > 0        # the warm start ran, and tuned
    assert fj.r_star == pytest.approx(
        optimal_ray(0.45, 1.0, 4.0, 4.0, 1.0, res.capacities[0] * fj.alloc_cost),
        rel=1e-9)
    spent = sum(st.alloc_cost * S for st, S in zip(stations, res.capacities))
    assert spent == pytest.approx(C, rel=1e-12)


def test_a_descending_budget_sweep_may_reuse_tuned_stations():
    """A tuned station that has already run must not refuse a budget it can serve.

    `retune` mutates the ray, and the ray it lands on at a generous budget needs MORE
    budget to stay stable than the policy's own minimum at r_star = 1. A station reused at
    a lower budget must be served against the policy's floor, not that ray's -- which takes
    both halves of the fix: `min_spend` reports the policy floor to the feasibility check,
    and `reset_policy` puts the station back on the ray eq 21 is then priced at.
    """
    stations = _tuned_pair()
    floor = min_feasible_budget(stations)
    Optimizer(stations, 20.0 * floor).run()
    stale = sum(s.alloc_cost * (s.gamma / s.mu) for s in stations)  # the ended-on rays
    assert stale > floor          # the high-budget ray really does need more
    C = 0.5 * (floor + stale)     # feasible for the policy, not for the ray it ended on
    reused = Optimizer(stations, C).run()
    fresh = Optimizer(_tuned_pair(), C).run()
    assert reused.converged
    assert reused.capacities == fresh.capacities        # bit-for-bit
    assert reused.objective == fresh.objective


# The single tuned station of the descending-sweep report, kept separate from
# `_tuned_pair` so the two floors are the station's own and can be pinned as numbers.
FJ_SWEEP = dict(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0)


def test_the_descending_sweep_figures_on_record():
    """Pins the measured figures docs/forkjoin-s2-policy/implementation.md cites for this,
    so that document stays executable rather than transcribed."""
    st = ForkJoinStation(**FJ_SWEEP, r_star=R_STAR_TUNED, name="fj")
    floor = min_feasible_budget([st])
    assert floor == pytest.approx(1.9125, rel=1e-12)
    Optimizer([st], 20.0 * floor).run()
    assert st.r_star == pytest.approx(2.6925527241298357, rel=1e-9)
    # The floor of the ray it ended on. `min_feasible_budget` deliberately no longer
    # reports this -- see test_the_exported_floor_agrees_with_run... below.
    assert st.alloc_cost * (st.gamma / st.mu) == pytest.approx(
        2.102912181464607, rel=1e-9)

    C = 2.008125          # above the policy's floor, below the ray it ended on
    reused = Optimizer([st], C).run()
    control = [ForkJoinStation(**FJ_SWEEP, r_star=R_STAR_TUNED, name="fj")]
    fresh = Optimizer(control, C).run()
    assert reused.converged
    assert reused.capacities == fresh.capacities
    assert reused.objective == fresh.objective


def test_the_reset_must_precede_the_first_allocation():
    """A policy-aware floor is not a substitute for restoring the ray -- the two guard
    different steps, and this pins why both are needed.

    `min_spend` lets the feasibility check clear a budget the policy can serve. But eq 21
    prices each station at its CURRENT ray, so allocating before the reset on a ray left
    over from a generous run gives NEGATIVE slack -- `-0.09521` at the budget used here --
    and `allocate` refuses outright. The run would fail to start rather than serve a budget
    the check had just cleared. Pinning this keeps the reset from drifting below the first
    `allocate`, and keeps it from being "simplified" away as redundant now that the floor
    is policy-aware.
    """
    from qopt.allocator import allocate
    from qopt.exceptions import InfeasibleBudgetError

    st = ForkJoinStation(**FJ_SWEEP, r_star=R_STAR_TUNED, name="fj")
    policy_floor = min_feasible_budget([st])
    Optimizer([st], 20.0 * policy_floor).run()          # leaves the ray at ~2.69
    ray_floor = st.alloc_cost * (st.gamma / st.mu)
    C = 0.5 * (policy_floor + ray_floor)
    assert policy_floor < C < ray_floor                 # the check passes, the ray cannot
    assert C - ray_floor == pytest.approx(-0.09520609073230357, rel=1e-9)

    with pytest.raises(InfeasibleBudgetError):
        allocate([st], C, [st.default_zeta])

    # And the real path, which resets between the two, serves that same budget.
    assert Optimizer([st], C).run().converged


def test_the_exported_floor_agrees_with_run_on_a_reused_tuned_station():
    """`min_feasible_budget` is public, and the README derives budgets from it, so it must
    not report a floor the Optimizer would accept a smaller budget than.

    Previously it read the mutable ray: after a generous run it reported that ray's floor
    (2.10291 here) while `run()` still served every budget above 1.9125, because `run()`
    restores the ray first. Budgets derived from the helper were inflated by run history.
    """
    st = ForkJoinStation(**FJ_SWEEP, r_star=R_STAR_TUNED, name="fj")
    fresh = min_feasible_budget([st])
    Optimizer([st], 20.0 * fresh).run()
    assert st.r_star == pytest.approx(2.6925527241298357, rel=1e-9)
    assert min_feasible_budget([st]) == fresh            # bit-for-bit, not merely close

    # And the reported floor is tight in both directions, which is what makes it usable:
    # just above it converges, at it is rejected.
    assert Optimizer([st], 1.0000001 * fresh).run().converged
    with pytest.raises(InfeasibleBudgetError):
        Optimizer([st], fresh).run()


def test_a_run_rejected_at_preflight_leaves_the_tuned_ray_alone():
    """Validation must not mutate. A budget that never passes preflight used to reset the
    station anyway, discarding a previous run's converged ray -- which is a reported
    output, read by `r_star` and by per-unit capacity attribution.

    Now possible because the feasibility check reads the policy's floor rather than the
    ray's, so the reset no longer has to run before it.
    """
    st = ForkJoinStation(**FJ_SWEEP, r_star=R_STAR_TUNED, name="fj")
    floor = min_feasible_budget([st])
    Optimizer([st], 20.0 * floor).run()
    answer = (st.r_star, st.mu, st.r, st.alloc_cost)
    assert st.r_star != 1.0

    with pytest.raises(InfeasibleBudgetError):
        Optimizer([st], 0.5 * floor).run()
    assert (st.r_star, st.mu, st.r, st.alloc_cost) == answer

    with pytest.raises(ValueError):
        Optimizer([st], float("nan")).run()
    assert (st.r_star, st.mu, st.r, st.alloc_cost) == answer

    with pytest.raises(ValueError):
        Optimizer([st], 4.0 * floor, initial_zeta=[-1.0]).run()
    assert (st.r_star, st.mu, st.r, st.alloc_cost) == answer

    # A run that DOES pass preflight still resets, so its answer is not warm-started off
    # the ray preserved above.
    reused = Optimizer([st], 4.0 * floor).run()
    control = [ForkJoinStation(**FJ_SWEEP, r_star=R_STAR_TUNED, name="fj")]
    fresh = Optimizer(control, 4.0 * floor).run()
    assert reused.capacities == fresh.capacities
    assert st.r_star == control[0].r_star


def test_a_tuned_station_runs_the_noise_floor_path():
    """The only other stochastic tuned test declares `half_width=None`, so `ci is None`,
    `_noise_floor` short-circuits, and `allocator.noise_floor` is never called with a tuned
    station at all -- even though it runs `allocate` 2n times, which now has a precondition
    that can raise. This exercises that path and the `noise-floor` stopping rule.
    """
    stations = _tuned_pair()
    fj = stations[0]
    C = 4.0 * min_feasible_budget(stations)
    # `warm_start=False` on purpose: the analytic pre-solve lands exactly on the answer
    # here, so with it the loop takes ONE step, stops on `tol`, and never gets far enough
    # for the noise floor to bind -- which is how this whole path stayed uncovered.
    res = Optimizer(stations, C, analyzer=DeterministicFake(half_width=0.001),
                    noise_kappa=1.0, warm_start=False).run()
    assert res.converged
    assert res.noise_floor == pytest.approx(0.0032843, rel=1e-3)
    assert res.stop_reason == "noise-floor"
    assert res.iterations > 1              # it actually iterated, unlike the other one
    # The ray is still the local optimum for the spend it ended on, and the budget is still
    # exactly spent -- the two invariants the analytic tests assert, on the noisy path.
    spend = res.capacities[0] * fj.alloc_cost
    assert fj.r_star == pytest.approx(
        optimal_ray(0.45, 1.0, 4.0, 4.0, 1.0, spend), rel=1e-9)
    assert sum(st.alloc_cost * S for st, S in zip(stations, res.capacities)) == \
        pytest.approx(C, rel=1e-12)


def test_every_station_is_reset_not_just_the_first():
    """The reset is a loop over all stations, and nothing pinned that.

    Every other reuse fixture happens to put the tuned fork-join at index 0, so
    `stations[0].reset_policy()` passed the whole suite. This puts it LAST behind two
    ordinary stations, where only a real loop reaches it.
    """
    def trio():
        return [GG1Station.mm1(gamma=0.9, mu=1.0, c=1.0, name="q1"),
                GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="q2"),
                ForkJoinStation(**FJ_SWEEP, r_star=R_STAR_TUNED, name="fj")]
    stations = trio()
    floor = min_feasible_budget(stations)
    Optimizer(stations, 20.0 * floor).run()
    assert stations[-1].r_star != 1.0                  # it is off the starting ray
    reused = Optimizer(stations, 4.0 * floor).run()
    fresh = Optimizer(trio(), 4.0 * floor).run()
    assert reused.capacities == fresh.capacities       # bit-for-bit
    assert reused.iterations == fresh.iterations


def test_reusing_tuned_stations_reproduces_a_fresh_run_exactly():
    """A run is a pure function of (stations-as-constructed, budget), tuning included.

    Without the reset the second run warm-starts from the first's answer, which is a
    different iterate sequence -- measured: 5 iterations against 6, and capacities agreeing
    only to ~1e-13. Iterations are asserted too because the capacities alone converge to
    nearly the same place either way, so they are the weaker witness of the two.
    """
    stations = _tuned_pair()
    C = 4.0 * min_feasible_budget(stations)
    first = Optimizer(stations, C).run()
    second = Optimizer(stations, C).run()
    assert second.capacities == first.capacities       # bit-for-bit
    assert second.iterations == first.iterations


def test_tuned_and_fixed_at_the_tuned_ray_reach_the_same_answer():
    """The fixed point's defining property, stated as an equivalence: freezing r_star at
    the value tuning converged to must reproduce the same allocation. If it did not, the
    tuned run would have stopped somewhere that is not a fixed point of eq 21 at its own
    prices."""
    stations = _tuned_pair()
    C = 4.0 * min_feasible_budget(stations)
    tuned = Optimizer(stations, C).run()
    frozen_stations = [
        ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0,
                        r_star=stations[0].r_star, name="fj"),
        GG1Station.mm1(gamma=0.9, mu=1.0, c=1.0, name="q"),
    ]
    frozen = Optimizer(frozen_stations, C).run()
    assert frozen.capacities == pytest.approx(tuned.capacities, rel=1e-8)
    assert frozen.objective == pytest.approx(tuned.objective, rel=1e-9)
