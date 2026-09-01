"""Offline tests for the QCSC example (spec 2026-07-31-qcsc-example-network-design)."""

import pytest

from qopt import ForkJoinStation, min_feasible_budget

EXPECTED_NAMES = [
    "cpu_init_ps", "fj_pp", "qpu_psq", "gpu_psq", "gpu_psg", "qpu_psg", "cpu_term_ps",
    "cpu_init_sp", "qpu_ssq", "gpu_ssq", "gpu_ssg", "qpu_ssg", "fj_sp", "cpu_term_sp",
]

# lambda * p11 = 0.45 into each stream; the sequential phases split that again by p0.
EXPECTED_GAMMA = [
    0.45, 0.45, 0.225, 0.225, 0.225, 0.225, 0.45,
    0.45, 0.225, 0.225, 0.225, 0.225, 0.45, 0.45,
]


def test_topology_names_and_derived_gamma():
    """Every gamma is derived by solve_traffic, so this catches a routing slip."""
    from examples.qcsc_network import WORKLOADS, build_qcsc_network

    for workload in WORKLOADS:
        network = build_qcsc_network(workload)
        assert [st.name for st in network] == EXPECTED_NAMES
        assert [st.gamma for st in network] == EXPECTED_GAMMA


def test_no_station_name_can_collide_with_qsim_fork_join_internals():
    from examples.qcsc_network import build_qcsc_network

    for st in build_qcsc_network("balanced"):
        assert "__" not in st.name


def test_rates_rejects_an_unknown_workload():
    """The three workload names are string literals everywhere; a typo must not fall
    through to an implicit None return."""
    from examples.qcsc_network import rates

    with pytest.raises(ValueError, match="unknown workload"):
        rates("hybrid_dominant", 2.0)


def test_min_feasible_budget_matches_hand_computation():
    """An independent oracle: sum_i c_i * gamma_i / mu_i, from the spec's parameters.

    4 CPUs       : 1 * 0.45  / 20  = 0.0225 each -> 0.09
    2 fork-joins : (4+1) * 0.45 / 1.0 = 2.25 each -> 4.50   (alloc_cost = c1 + c2)
    4 QPU queues : 4 * 0.225 / mu_Q
    4 GPU queues : 1 * 0.225 / mu_G
    """
    from examples.qcsc_network import build_qcsc_network

    cpus, forkjoins = 0.09, 4.50
    expected = {
        "balanced":           cpus + forkjoins + 4 * (4 * 0.225 / 2.0) + 4 * (0.225 / 2.0),
        "quantum_dominant":   cpus + forkjoins + 4 * (4 * 0.225 / 2.0) + 4 * (0.225 / 8.0),
        "classical_dominant": cpus + forkjoins + 4 * (4 * 0.225 / 8.0) + 4 * (0.225 / 2.0),
    }
    for workload, want in expected.items():
        got = min_feasible_budget(build_qcsc_network(workload).stations)
        assert got == pytest.approx(want), workload


def test_fork_join_heterogeneity_ratio_per_workload():
    """balanced is r = 1 by definition; both dominant variants carry r = 4 (spec 5)."""
    from examples.qcsc_network import build_qcsc_network

    expected_r = {"balanced": 1.0, "quantum_dominant": 4.0, "classical_dominant": 4.0}
    for workload, want in expected_r.items():
        stations = {st.name: st for st in build_qcsc_network(workload)}
        for name in ("fj_pp", "fj_sp"):
            station = stations[name]
            assert isinstance(station, ForkJoinStation)
            assert station.r == want, (workload, name)
            assert station.mu == 1.0, (workload, name)   # B_PP = B_SP = 1.0, slower side


def test_fork_join_cost_follows_the_server_not_the_speed():
    """The QPU branch costs C_QPU whether or not it is the bottleneck (spec 5)."""
    from examples.qcsc_network import build_qcsc_network

    quantum = {st.name: st for st in build_qcsc_network("quantum_dominant")}["fj_pp"]
    classical = {st.name: st for st in build_qcsc_network("classical_dominant")}["fj_pp"]
    # quantum-dominant: QPU is slower, so it is server 1.
    assert (quantum.c1, quantum.c2) == (4.0, 1.0)
    # classical-dominant: GPU is slower, so the cheap server is server 1.
    assert (classical.c1, classical.c2) == (1.0, 4.0)
    # Either way the fork-join spends the same, which is why the floors differ only
    # through the single-server queues.
    assert quantum.alloc_cost == classical.alloc_cost == 5.0


def test_shared_budget_is_six_times_the_balanced_floor():
    from examples.qcsc_network import build_qcsc_network, shared_budget

    floor = min_feasible_budget(build_qcsc_network("balanced").stations)
    assert floor == pytest.approx(6.84)
    assert shared_budget() == pytest.approx(6.0 * floor)


def test_shared_budget_is_feasible_for_every_workload():
    from examples.qcsc_network import WORKLOADS, build_qcsc_network, shared_budget

    budget = shared_budget()
    for workload in WORKLOADS:
        floor = min_feasible_budget(build_qcsc_network(workload).stations)
        assert budget > floor, workload


# Regression pins. These are not independent oracles -- they are the values the spec
# recorded after verifying feasibility and convergence. If one fails, the topology, the
# rates, or the budget changed: find out which before touching the number.
EXPECTED_OBJECTIVE = {
    "balanced": 6.401439829328976,
    "quantum_dominant": 4.528843770190739,
    "classical_dominant": 3.463677319176492,
}


def test_every_station_name_carries_a_reporting_prefix():
    """capacity_by_unit groups on these prefixes, so the contract must hold."""
    from examples.qcsc_network import UNIT_PREFIXES, WORKLOADS, build_qcsc_network

    for workload in WORKLOADS:
        for st in build_qcsc_network(workload):
            assert st.name.startswith(UNIT_PREFIXES), st.name


def test_all_workloads_converge_and_stay_stable_at_the_shared_budget():
    from qopt import Optimizer

    from examples.qcsc_network import WORKLOADS, build_qcsc_network, shared_budget

    budget = shared_budget()
    for workload in WORKLOADS:
        network = build_qcsc_network(workload)
        result = Optimizer(network, budget=budget).run()
        assert result.converged, workload
        # A bound, not a pin: the exact count is 5 today, but it is a property of qopt's
        # tolerance and damping defaults, not of this network. Pinning it would make an
        # unrelated tuning change in the library fail an example test.
        assert result.iterations <= 10, (workload, result.iterations)
        for st, S in zip(network, result.capacities):
            assert S * st.mu > st.gamma, (workload, st.name)


def test_objectives_match_the_recorded_values():
    from qopt import Optimizer

    from examples.qcsc_network import build_qcsc_network, shared_budget

    budget = shared_budget()
    for workload, expected in EXPECTED_OBJECTIVE.items():
        result = Optimizer(build_qcsc_network(workload), budget=budget).run()
        assert result.objective == pytest.approx(expected, rel=1e-12), workload


def test_unit_costs_collapse_the_two_dominant_workloads():
    """Spec 5.1: the topology is symmetric in QPU/GPU, so only the cost vector separates
    quantum-dominant from classical-dominant. Under unit costs they are identical."""
    from qopt import Optimizer

    from examples.qcsc_network import build_qcsc_network, shared_budget

    unit_budget = shared_budget(c_qpu=1.0)
    quantum, classical = (
        Optimizer(build_qcsc_network(w, c_qpu=1.0), budget=unit_budget).run()
        for w in ("quantum_dominant", "classical_dominant")
    )
    # The symmetry claim itself, stated exactly: the per-station sojourn times of one are a
    # permutation of the other's. As `approx`, not bitwise -- every S_i is computed through
    # `slack` and `denom` (qopt/allocator.py), both left-folds over the stations, and the
    # two workloads present those summands in a different order. IEEE-754 addition is
    # commutative but not associative, so a permuted fold is not GUARANTEED to reproduce
    # the same bits, even though for these values it does.
    assert sorted(quantum.sojourn_times) == pytest.approx(
        sorted(classical.sojourn_times), rel=1e-12
    )
    # The objectives therefore agree, for the same reason and to the same kind of
    # tolerance: `objective` is itself a left-fold over station order (qopt/optimizer.py).
    assert quantum.objective == pytest.approx(classical.objective, rel=1e-15)

    real_budget = shared_budget()
    separated = [
        Optimizer(build_qcsc_network(w), budget=real_budget).run().objective
        for w in ("quantum_dominant", "classical_dominant")
    ]
    assert separated[0] != separated[1]        # C_QPU = 4 breaks the symmetry


def test_visit_ratio_weighted_total_differs_from_the_objective():
    """The optimized objective is the unweighted sum; the diagnostic is the mean job
    sojourn time. They must not be the same number (spec section 6)."""
    from qopt import Optimizer

    from examples.qcsc_network import (build_qcsc_network, shared_budget,
                                       visit_ratio_weighted)

    network = build_qcsc_network("balanced")
    result = Optimizer(network, budget=shared_budget()).run()
    weighted = visit_ratio_weighted(network, result.sojourn_times)
    assert weighted == pytest.approx(2.282953760142859, rel=1e-12)
    assert weighted < result.objective


def test_capacity_by_unit_rejects_a_station_with_no_known_prefix():
    """The grouping is exhaustive by contract, so an unrecognised name must raise rather
    than be silently dropped from every column."""
    from examples.qcsc_network import build_qcsc_network, capacity_by_unit

    network = build_qcsc_network("balanced")
    network.stations[0].name = "tpu_init_ps"
    with pytest.raises(ValueError, match="matches none of UNIT_PREFIXES"):
        capacity_by_unit(network, [1.0] * len(network.stations))


def test_capacity_by_unit_counts_a_fork_join_on_both_sides():
    """The default ray r_star = r, where the two servers do receive the same capacity."""
    from examples.qcsc_network import build_qcsc_network, capacity_by_unit

    network = build_qcsc_network("balanced")
    capacities = [1.0] * len(network.stations)     # one unit each, so sums are counts
    by_unit = capacity_by_unit(network, capacities)
    assert by_unit["cpu"] == 4.0                  # 4 CPU stations
    assert by_unit["qpu"] == 6.0                  # 4 QPU queues + 2 fork-join QPU servers
    assert by_unit["gpu"] == 6.0                  # 4 GPU queues + 2 fork-join GPU servers


def test_dot_flag_prints_the_topology_and_returns_none(capsys):
    from examples.qcsc_network import main

    assert main(["--dot"]) is None
    out = capsys.readouterr().out
    assert out.startswith("digraph")
    assert '"fj_pp"' in out and '"cpu_term_sp"' in out
    assert '"src" -> "cpu_init_ps" [label="0.5"]' in out
    # One edge per declared Route, no more: the diagram is only usable as a check on the
    # topology if it cannot silently lose or duplicate an edge.
    assert out.count(" -> ") == 18
    assert out.count("[shape=box3d") == 2           # exactly the two fork-joins


def test_main_runs_analytically_without_a_service(monkeypatch, capsys):
    monkeypatch.delenv("QOPT_QSIM_URL", raising=False)
    from examples.qcsc_network import WORKLOADS, main

    results = main([])
    assert sorted(results) == sorted(WORKLOADS)
    for workload, result in results.items():
        assert result.sim_calls == 0, workload
        assert result.converged, workload
    out = capsys.readouterr().out
    assert "QOPT_QSIM_URL" in out               # the hint is printed
    assert "cumulative capacity" in out


def test_stopping_rule_matches_the_existing_simulated_example():
    """Same rule as examples/simulated_mixed_network.py, so run times are comparable.

    That correspondence is by inspection, not enforced: the sibling's dict is inline
    inside its own main() and not importable, so this pins qcsc_network's STOPPING to a
    literal copy. If the sibling's dict changes, this test will not notice.
    """
    from examples.qcsc_network import STOPPING

    assert STOPPING == {
        "alpha": 0.05, "precision": 0.02, "minSamples": 100000,
        "maxSamples": 4000000, "maxWallClockSeconds": 300,
    }


def test_print_table_reports_intervals_a_diagnostic_and_a_failure_to_converge(capsys):
    """print_table's simulated-path branches, none of which the analytic run reaches.

    The analytic pass leaves `sojourn_ci` and `system_response_time` at None and always
    converges, so a live service was until now the only thing that exercised the CI
    columns, the missing-CI placeholder, the diagnostic line, or the NOT CONVERGED warning
    -- the branch that exists precisely to stop someone trusting an S* that did not
    converge.
    """
    from qopt.optimizer import Result

    from examples.qcsc_network import build_qcsc_network, print_table

    network = build_qcsc_network("balanced")
    n = len(network.stations)
    sojourn_ci = [(0.4, 0.6)] * n
    sojourn_ci[3] = None                     # a mean with no interval (spec 8.1)
    result = Result(
        capacities=[3.0] * n, sojourn_times=[0.5] * n, zeta=[1.0] * n,
        objective=7.0, iterations=20, residual=2.5e-3, converged=False,
        stop_reason="max_iter", sojourn_ci=sojourn_ci,
        system_response_time=(2.5, (2.4, 2.6)),
    )
    print_table("SIMULATED (fake)", network, result)
    out = capsys.readouterr().out

    assert "NOT CONVERGED" in out and "2.500e-03" in out
    assert "E[T] 95% CI" in out                        # the header gains its column
    assert "(0.400000, 0.600000)" in out
    assert "system response time = 2.500000 CI (2.400000, 2.600000)" in out
    placeholder = [ln for ln in out.splitlines() if ln.split()[:1] == [network.stations[3].name]]
    assert len(placeholder) == 1 and placeholder[0].endswith("--")
    assert "None" not in out

    # A system response time whose own interval is missing must degrade to a label, not
    # format None into the output.
    result.system_response_time = (2.5, (None, None))
    print_table("SIMULATED (fake, no interval)", network, result)
    out = capsys.readouterr().out
    assert "system response time = 2.500000 CI unavailable" in out
    assert "None" not in out


def test_print_gaps_handles_a_station_with_no_confidence_interval(capsys):
    """A missing CI must print, not raise (spec 8.1)."""
    from qopt.optimizer import Result

    from examples.qcsc_network import build_qcsc_network, print_gaps

    network = build_qcsc_network("balanced")
    n = len(network.stations)
    analytic = Result(
        capacities=[3.0] * n, sojourn_times=[0.5] * n, zeta=[1.0] * n,
        objective=7.0, iterations=3, residual=1e-7, converged=True, stop_reason="tol",
    )
    simulated = Result(
        capacities=[3.0] * n, sojourn_times=[0.6] * n, zeta=[1.0] * n,
        objective=8.4, iterations=3, residual=1e-7, converged=True, stop_reason="tol",
        sojourn_ci=[None] * n,
    )
    print_gaps("balanced", network, analytic, simulated)
    out = capsys.readouterr().out
    assert "cpu_init_ps" in out
    assert "None" not in out


def test_print_gaps_flags_a_gap_that_exceeds_its_ci_half_width(capsys):
    """The '> CI half-width?' verdict is the logic spec section 7's '2 of 42' count rests
    on. The test above only exercises the no-CI branch; this exercises both real
    branches of `abs(gap) > 0.5 * (upper - lower)`, so inverting that comparison would
    fail a test here rather than only silently inverting the documented conclusion."""
    from qopt.optimizer import Result

    from examples.qcsc_network import build_qcsc_network, print_gaps

    network = build_qcsc_network("balanced")
    n = len(network.stations)
    flagged, clear = network.stations[0].name, network.stations[1].name
    analytic = Result(
        capacities=[3.0] * n, sojourn_times=[0.5] * n, zeta=[1.0] * n,
        objective=7.0, iterations=3, residual=1e-7, converged=True, stop_reason="tol",
    )
    sojourn_times = [0.5] * n
    sojourn_times[0] = 0.52   # gap 0.02
    sojourn_times[1] = 0.51   # gap 0.01
    sojourn_ci = [(0.5, 0.5)] * n
    sojourn_ci[0] = (0.495, 0.505)   # half-width 0.005 < |gap| 0.02 -> flagged
    sojourn_ci[1] = (0.46, 0.56)     # half-width 0.05  > |gap| 0.01 -> clear
    simulated = Result(
        capacities=[3.0] * n, sojourn_times=sojourn_times, zeta=[1.0] * n,
        objective=8.4, iterations=3, residual=1e-7, converged=True, stop_reason="tol",
        sojourn_ci=sojourn_ci,
    )
    print_gaps("balanced", network, analytic, simulated)
    verdicts = {
        line.split()[0]: line.split()[-1]
        for line in capsys.readouterr().out.splitlines()
        if line.split() and line.split()[0] in (flagged, clear)
    }
    assert verdicts[flagged] == "yes"
    assert verdicts[clear] == "no"


# --------------------------------------------------------------------------------------
# Reporting under a non-default ray (issue #10). A fork-join's two servers receive the
# same capacity only at r_star = r, so the per-unit totals must split them.
# --------------------------------------------------------------------------------------

def test_capacity_by_unit_splits_a_fork_joins_two_servers():
    """At r_star = 1 server 2 buys S/r, not S. Counting S on both sides would overstate
    the faster unit's purchased capacity by a factor of r."""
    from examples.qcsc_network import build_qcsc_network, capacity_by_unit

    # quantum_dominant: the QPU is the slower unit, so it is server 1 and keeps S.
    network = build_qcsc_network("quantum_dominant", r_star=1.0)
    by_unit = capacity_by_unit(network, [1.0] * len(network.stations))
    assert by_unit["cpu"] == pytest.approx(4.0)
    assert by_unit["qpu"] == pytest.approx(4.0 + 2 * 1.0)
    assert by_unit["gpu"] == pytest.approx(4.0 + 2 * 0.25)   # S*r_star/r = 1/4

    # classical_dominant is the mirror image: the GPU is slower, so it is server 1.
    network = build_qcsc_network("classical_dominant", r_star=1.0)
    by_unit = capacity_by_unit(network, [1.0] * len(network.stations))
    assert by_unit["gpu"] == pytest.approx(4.0 + 2 * 1.0)
    assert by_unit["qpu"] == pytest.approx(4.0 + 2 * 0.25)


def test_fork_join_unit_labels_agree_with_its_cost_and_rate_assignment():
    """The server -> unit map drives every fork-join row of the reporting table, and it
    is not recoverable from the station's own fields (balanced has mu_q == mu_g, and the
    unit-cost runs have c_qpu == c_gpu). Pin it against both hardware facts."""
    from examples.qcsc_network import B_PP, C_GPU, C_QPU, build_qcsc_network, rates

    cost = {"qpu": C_QPU, "gpu": C_GPU}
    for workload in ("balanced", "quantum_dominant", "classical_dominant"):
        mu_q, mu_g = rates(workload, B_PP)
        rate = {"qpu": mu_q, "gpu": mu_g}
        fj = {st.name: st for st in build_qcsc_network(workload)}["fj_pp"]
        u1, u2 = fj.units
        assert (fj.c1, fj.c2) == (cost[u1], cost[u2]), workload
        assert fj.mu == rate[u1] and rate[u1] <= rate[u2], workload


def test_r_star_reaches_the_fork_joins_and_leaves_everything_else_alone():
    from examples.qcsc_network import build_qcsc_network

    stations = {st.name: st for st in build_qcsc_network("quantum_dominant", r_star=1.0)}
    for name in ("fj_pp", "fj_sp"):
        assert stations[name].r_star == 1.0, name
        assert stations[name].alloc_cost == pytest.approx(4.0 + 1.0 / 4.0), name
    # The default is unchanged, which is what keeps every recorded objective valid.
    default = {st.name: st for st in build_qcsc_network("quantum_dominant")}
    assert default["fj_pp"].r_star == 4.0
    assert default["fj_pp"].alloc_cost == pytest.approx(5.0)


def test_capacity_by_unit_refuses_a_multi_unit_station_that_cannot_split():
    """Falling back to "add S to every column" would silently overstate the faster unit
    on any ray but r_star = r, so a station spanning two units has to be able to split."""
    from qopt import ForkJoinStation
    from examples.qcsc_network import build_qcsc_network, capacity_by_unit

    network = build_qcsc_network("balanced")
    network.stations[1] = ForkJoinStation(
        gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, name="fj_pp")
    with pytest.raises(ValueError, match="cannot attribute its capacity"):
        capacity_by_unit(network, [1.0] * len(network.stations))


# --------------------------------------------------------------------------------------
# The nested r_star fixed point on the real 14-station network (issue #10 item 3). Every
# reference figure is from findings section 7, produced at this same budget C = 41.04 by an
# independent probe that patched the example's own topology.
# --------------------------------------------------------------------------------------

# findings section 7: objective under the paper's policy, and under the best ray found by
# sweeping r_star on a 0.02 grid. Published to 6dp, so compared with a matching absolute
# tolerance -- unlike EXPECTED_OBJECTIVE above, which carries the incumbent's full floats.
# Half a unit of the last printed digit would be 5e-7, and `quantum_dominant`'s
# equal-rate objective sits 4.99e-7 from its published figure -- passing with 0.2% of the
# tolerance to spare, so any float-level reordering elsewhere would flip it red for
# reasons unrelated to the policy. Doubled, which is still far tighter than any regression
# in the ray this pins would produce.
PUBLISHED_DP = 1e-6
FINDINGS_SECTION_7 = {
    "balanced":           dict(equal=6.401440, best=6.249439),
    "quantum_dominant":   dict(equal=4.776428, best=4.431693),
    "classical_dominant": dict(equal=2.613335, best=2.613335),
}

# findings section 7: the best ray found by sweeping r_star on a 0.02 grid. Compared with
# that grid's own resolution as the tolerance, which is the only defensible figure -- a
# sweep cannot locate an optimum finer than its step.
#
# Deliberately NOT compared against section 7's other triple, 1.4457 / 2.3195 / 1.0000.
# Those are the local condition evaluated at the spend the INNER-SPLIT embedding converged
# to, and that embedding is the one section 10 item 4 says not to use -- it converged at a
# different spend (7.96 against 7.49), so its rays are not this computation's target. They
# agree to 1.2e-3 and 1.5e-3, which is corroboration, not a specification.
# What THIS implementation converges to, at full precision. Pinned separately from the
# published figures above because the assertions against those are inequalities -- "no worse
# than the sweep found", "within the sweep's own grid" -- which by construction cannot
# notice the tuned result drifting to a slightly different, still-better point. These are
# the regression pin on the headline result of issue #10 item 3.
TUNED = {
    "balanced":           (6.2494291783037355, 1.447382318551769),
    "quantum_dominant":   (4.431691350142377,  2.316117567446687),
    "classical_dominant": (2.6133350927675547, 1.0000000000000002),
}

FINDINGS_BEST_RAY = {
    "balanced": 1.440, "quantum_dominant": 2.320, "classical_dominant": 1.000,
}
SWEEP_GRID = 0.02


def _qcsc_run(workload, r_star):
    from qopt import Optimizer

    from examples.qcsc_network import build_qcsc_network, shared_budget

    net = build_qcsc_network(workload, r_star=r_star)
    result = Optimizer(net, budget=shared_budget()).run()
    rays = [st.r_star for st in net.stations if st.name.startswith("fj")]
    return net, result, rays


@pytest.mark.parametrize("workload", sorted(FINDINGS_SECTION_7))
def test_named_policies_reproduce_the_recorded_objectives(workload):
    """The two incumbents, selected by name, must land on their published numbers."""
    from qopt import R_STAR_EQUAL_RATE, R_STAR_INVARIANT_R

    assert _qcsc_run(workload, R_STAR_INVARIANT_R)[1].objective == \
        pytest.approx(EXPECTED_OBJECTIVE[workload], rel=1e-12)
    assert _qcsc_run(workload, R_STAR_EQUAL_RATE)[1].objective == \
        pytest.approx(FINDINGS_SECTION_7[workload]["equal"], abs=PUBLISHED_DP)


@pytest.mark.parametrize("workload", sorted(TUNED))
def test_tuned_objective_and_ray_are_pinned(workload):
    """Exact regression pin. The other tuned assertions are inequalities against published
    figures, so they cannot catch this result moving to a different still-better point."""
    from qopt import R_STAR_TUNED

    want_objective, want_ray = TUNED[workload]
    _, result, rays = _qcsc_run(workload, R_STAR_TUNED)
    assert result.objective == pytest.approx(want_objective, rel=1e-12)
    assert rays == pytest.approx([want_ray, want_ray], rel=1e-9)


@pytest.mark.parametrize("workload", sorted(FINDINGS_SECTION_7))
def test_tuning_reaches_the_swept_optimum_without_being_told_where_it_is(workload):
    """The nested fixed point must find what a 0.02-grid sweep of r_star found.

    Asserted as "no worse than", because the two are not the same computation: the sweep
    picked the best point of a grid, while this solves the local condition at its own
    converged spend, so it is free to land BETWEEN grid points and score marginally
    better. It may not land worse.
    """
    from qopt import R_STAR_TUNED

    want = FINDINGS_SECTION_7[workload]
    _, result, rays = _qcsc_run(workload, R_STAR_TUNED)
    assert result.converged
    assert result.objective <= want["best"] + PUBLISHED_DP
    # Both fork-joins share gamma and rates, so they must agree exactly.
    assert rays[0] == pytest.approx(rays[1], rel=1e-12)
    assert rays[0] == pytest.approx(FINDINGS_BEST_RAY[workload], abs=SWEEP_GRID)


def test_tuning_recovers_the_papers_rule_exactly_where_it_is_optimal():
    """`classical_dominant` is the one workload where r_star = 1 is the true optimum, and
    where the incumbent loses to it by 24.55%. Tuning has to find that on its own -- and
    find it exactly, not merely nearby, since r_star = 1 is `t_bot`'s kink."""
    from qopt import R_STAR_EQUAL_RATE, R_STAR_TUNED

    _, tuned, rays = _qcsc_run("classical_dominant", R_STAR_TUNED)
    _, paper, _ = _qcsc_run("classical_dominant", R_STAR_EQUAL_RATE)
    assert rays == pytest.approx([1.0, 1.0], abs=1e-9)
    assert tuned.objective == pytest.approx(paper.objective, rel=1e-9)


@pytest.mark.parametrize("workload", sorted(FINDINGS_SECTION_7))
def test_tuning_spends_the_whole_shared_budget(workload):
    """Pins the budget identity at the fixed point. Like its unit-level twin it catches a
    WRONG rescale but not a missing one -- once r_star settles, `alloc_cost` stops changing
    and eq 21's own allocation exhausts C either way. The mechanism is pinned by
    test_retune_preserves_the_stations_spend_exactly."""
    from qopt import R_STAR_TUNED

    from examples.qcsc_network import shared_budget

    net, result, _ = _qcsc_run(workload, R_STAR_TUNED)
    spent = sum(st.alloc_cost * S for st, S in zip(net.stations, result.capacities))
    assert spent == pytest.approx(shared_budget(), rel=1e-12)


@pytest.mark.parametrize("workload", sorted(FINDINGS_SECTION_7))
def test_capacity_by_unit_follows_a_tuned_station(workload):
    """The reporting contract has to track a ray chosen during the run, not at
    construction -- `capacity_by_unit` reads `r_star` when asked, so it does."""
    from qopt import R_STAR_TUNED

    from examples.qcsc_network import build_qcsc_network, capacity_by_unit

    net, result, rays = _qcsc_run(workload, R_STAR_TUNED)
    stations = list(net.stations)
    fj = next(st for st in stations if st.name == "fj_pp")
    S = result.capacities[stations.index(fj)]
    # Server 2's share moved with the tuned ray, and the report has to use the moved one.
    assert fj.server_capacities(S)[1] == pytest.approx(S * rays[0] / fj.r_base, rel=1e-12)
    tuned_total = capacity_by_unit(net, result.capacities)
    frozen = capacity_by_unit(
        build_qcsc_network(workload, r_star=fj.r_base), result.capacities)
    slow, fast = fj.units
    assert tuned_total[fast] != pytest.approx(frozen[fast], rel=1e-9)
    assert tuned_total[slow] == pytest.approx(frozen[slow], rel=1e-12)
