"""The paper's QCSC network (docs/analysis.pdf section 2, Figure 5 p. 30).

Fourteen stations, sixteen single-server queues, two fork-joins, one open chain:

    src --p11--> cpu_init_ps -> [fj_pp] --p0--> qpu_psq -> gpu_psq --+
                                          --1-p0--> gpu_psg -> qpu_psg --+-> cpu_term_ps -> snk
        --1-p11-> cpu_init_sp --p0--> qpu_ssq -> gpu_ssq --+
                              --1-p0--> gpu_ssg -> qpu_ssg --+-> [fj_sp] -> cpu_term_sp -> snk

Stream 1 is the parallel-sequential application class (a fork-join parallel phase, then a
sequential phase); stream 2 is sequential-parallel (the same phases in the opposite
order). Every gamma is DERIVED from this topology by the traffic equations.

Three workloads differ only in service rates (section 2 p. 6): balanced (mu_Q = mu_G),
quantum-dominant (mu_Q < mu_G), classical-dominant (mu_Q > mu_G). Same topology, same
arrival rate, same budget, so the printed differences are attributable to the workload.

Not every station's arrivals are Poisson, which is what the analytic per-station form
assumes:

  * cpu_init_ps and cpu_init_sp receive a Bernoulli split of a Poisson stream, which is
    still Poisson. Their cov_a = 1 is exact here.
  * Stream 1's sequential queues sit DOWNSTREAM OF A FORK-JOIN, so their arrivals are
    join completions (a max over two branches), not Poisson -- the analytic form is
    weakest there. Both fork-joins run at r = 4, so t_ul's heterogeneous-server
    approximation adds a bias of its own at fj_pp and fj_sp; fj_sp also takes a
    SUPERPOSITION of two tandem departure streams rather than a Poisson input.
  * Stream 2's sequential queues are fed by a CPU whose own input is Poisson, so they
    should track the analytic values closely.

The two streams therefore run the same phases in opposite order, which makes them a
controlled comparison for arrival-process coupling.

That reasoning was measured against a live simulated pass and the predicted MAGNITUDE did
not hold: no station row differs by more than ~1.1%, and the two streams are not cleanly
separated. A small consistent bias is there -- the simulator runs below the closed form in
31 of 42 rows -- but nothing like the predicted effect, most likely because the optimizer
allocates 6-7.5x the minimum feasible budget, so every station runs at modest utilization
where arrival-process shape matters far less than it does near saturation. Spec section 7
holds the numbers, the statistics, and what they do and do not license; that is the single
place this story is maintained, so read it there rather than trusting a summary.

Run `python -m examples.qcsc_network` for the analytic tables; add QOPT_QSIM_URL for the
simulated pass, or `--dot` to print the topology as Graphviz DOT.
"""

import os
import sys

from qopt import (
    ForkJoinStation,
    GG1Station,
    Network,
    Optimizer,
    QsimClient,
    Route,
    SimulationAnalyzer,
    min_feasible_budget,
)

# --- workload and system parameters (spec section 4) -------------------------------
LAMBDA = 0.9          # arrival rate
P11 = 0.5             # P[parallel-sequential class]
P0 = 0.5              # P[sequential phase starts with the quantum task]
R = 4.0               # heterogeneity ratio, shared by both dominant workloads

B_PP = B_SP = 1.0                        # parallel-phase base level
B_PSQ = B_PSG = B_SSQ = B_SSG = 2.0      # sequential-phase base levels
MU_CPU = 20.0                            # mu_I, mu_T >> all others (section 2)

C_QPU = 4.0           # cost per unit of QPU capacity
C_GPU = 1.0
C_CPU = 1.0

BUDGET_MULTIPLE = 6.0

WORKLOADS = ("balanced", "quantum_dominant", "classical_dominant")

STOPPING = {
    "alpha": 0.05,
    "precision": 0.02,
    "minSamples": 100000,
    "maxSamples": 4000000,
    "maxWallClockSeconds": 300,
}
"""Same stopping rule as examples/simulated_mixed_network.py."""


def rates(workload, b):
    """(mu_Q, mu_G) for a phase whose base level is `b`.

    The entire difference between the three workloads. `balanced` is necessarily ratio 1:
    the paper defines it as mu_Q = mu_G, so it cannot carry R.
    """
    if workload == "balanced":
        return b, b
    if workload == "quantum_dominant":
        return b, R * b            # QPU slower -> QPU is the bottleneck server
    if workload == "classical_dominant":
        return R * b, b            # GPU slower
    raise ValueError(f"unknown workload {workload!r}, expected one of {WORKLOADS}")


def _fork_join(workload, b, name, c_qpu, c_gpu):
    """ForkJoinStation for a parallel phase: mu is the slower server, r = fast/slow.

    Costs attach to the SERVER, not to the speed, so the QPU branch costs c_qpu whether
    or not it is the bottleneck. That asymmetry is what distinguishes the two dominant
    workloads (spec section 5.1).
    """
    mu_q, mu_g = rates(workload, b)
    if mu_q <= mu_g:
        return ForkJoinStation(mu=mu_q, r=mu_g / mu_q, c1=c_qpu, c2=c_gpu, name=name)
    return ForkJoinStation(mu=mu_g, r=mu_q / mu_g, c1=c_gpu, c2=c_qpu, name=name)


def build_qcsc_network(workload, *, c_qpu=C_QPU, c_gpu=C_GPU, c_cpu=C_CPU):
    """The 14-station QCSC network for one workload. Costs are overridable so that the
    QPU/GPU symmetry of the topology can be exercised under unit costs (spec 5.1)."""
    q_psq, g_psq = rates(workload, B_PSQ)
    q_psg, g_psg = rates(workload, B_PSG)
    q_ssq, g_ssq = rates(workload, B_SSQ)
    q_ssg, g_ssg = rates(workload, B_SSG)
    stations = [
        GG1Station.mm1(mu=MU_CPU, c=c_cpu, name="cpu_init_ps"),
        _fork_join(workload, B_PP, "fj_pp", c_qpu, c_gpu),
        GG1Station.mm1(mu=q_psq, c=c_qpu, name="qpu_psq"),   # p0 branch: quantum first
        GG1Station.mm1(mu=g_psq, c=c_gpu, name="gpu_psq"),
        GG1Station.mm1(mu=g_psg, c=c_gpu, name="gpu_psg"),   # 1-p0 branch: classical first
        GG1Station.mm1(mu=q_psg, c=c_qpu, name="qpu_psg"),
        GG1Station.mm1(mu=MU_CPU, c=c_cpu, name="cpu_term_ps"),
        GG1Station.mm1(mu=MU_CPU, c=c_cpu, name="cpu_init_sp"),
        GG1Station.mm1(mu=q_ssq, c=c_qpu, name="qpu_ssq"),
        GG1Station.mm1(mu=g_ssq, c=c_gpu, name="gpu_ssq"),
        GG1Station.mm1(mu=g_ssg, c=c_gpu, name="gpu_ssg"),
        GG1Station.mm1(mu=q_ssg, c=c_qpu, name="qpu_ssg"),
        _fork_join(workload, B_SP, "fj_sp", c_qpu, c_gpu),
        GG1Station.mm1(mu=MU_CPU, c=c_cpu, name="cpu_term_sp"),
    ]
    routes = [
        Route(Network.SOURCE, "cpu_init_ps", P11),
        Route(Network.SOURCE, "cpu_init_sp", 1.0 - P11),
        # stream 1: parallel phase, then sequential phase
        Route("cpu_init_ps", "fj_pp"),
        Route("fj_pp", "qpu_psq", P0),
        Route("fj_pp", "gpu_psg", 1.0 - P0),
        Route("qpu_psq", "gpu_psq"),
        Route("gpu_psq", "cpu_term_ps"),
        Route("gpu_psg", "qpu_psg"),
        Route("qpu_psg", "cpu_term_ps"),
        Route("cpu_term_ps", Network.SINK),
        # stream 2: sequential phase, then parallel phase
        Route("cpu_init_sp", "qpu_ssq", P0),
        Route("cpu_init_sp", "gpu_ssg", 1.0 - P0),
        Route("qpu_ssq", "gpu_ssq"),
        Route("gpu_ssq", "fj_sp"),
        Route("gpu_ssg", "qpu_ssg"),
        Route("qpu_ssg", "fj_sp"),
        Route("fj_sp", "cpu_term_sp"),
        Route("cpu_term_sp", Network.SINK),
    ]
    return Network(stations, routes, arrival_rate=LAMBDA, name=f"qcsc-{workload}")


def shared_budget(*, c_qpu=C_QPU, c_gpu=C_GPU, c_cpu=C_CPU):
    """One absolute budget for all three workloads: BUDGET_MULTIPLE x the balanced floor.

    Deliberately not each workload's own floor. Sharing one number is what makes the
    three E[T] columns comparable -- same money, different workload (spec section 6).
    """
    balanced = build_qcsc_network("balanced", c_qpu=c_qpu, c_gpu=c_gpu, c_cpu=c_cpu)
    return BUDGET_MULTIPLE * min_feasible_budget(balanced.stations)


UNITS_OF_PREFIX = {
    "cpu_": ("cpu",),
    "qpu_": ("qpu",),
    "gpu_": ("gpu",),
    "fj_": ("qpu", "gpu"),
}
"""Station-name prefix -> the processing units its capacity counts toward.

The single source of truth for both halves of the reporting contract: capacity_by_unit
groups on this, and every station name must carry one of these prefixes. A fork-join maps
to BOTH 'qpu' and 'gpu' because in qopt both of its servers receive the same S (see spec
section 10 -- the paper instead sets S_2 = S_1/r, which qopt deliberately does not do).
"""

UNIT_PREFIXES = tuple(UNITS_OF_PREFIX)
"""The prefixes themselves, derived so the two cannot drift apart."""


def capacity_by_unit(network, capacities):
    """Cumulative allocated capacity per unit type, grouped by UNITS_OF_PREFIX."""
    totals = {unit: 0.0 for units in UNITS_OF_PREFIX.values() for unit in units}
    for st, S in zip(network.stations, capacities):
        for prefix, units in UNITS_OF_PREFIX.items():
            if st.name.startswith(prefix):
                for unit in units:
                    totals[unit] += S
                break
        else:
            raise ValueError(
                f"station {st.name!r} matches none of UNIT_PREFIXES {UNIT_PREFIXES}"
            )
    return totals


def visit_ratio_weighted(network, sojourn_times):
    """sum_i (gamma_i / LAMBDA) * E[T_i] -- the mean end-to-end job sojourn time.

    A diagnostic only. The OPTIMIZED objective uses omega_i = 1 (the paper's default),
    which is the plain sum of the 14 expected sojourn times and is a different quantity.
    """
    return sum(
        (st.gamma / network.arrival_rate) * t
        for st, t in zip(network.stations, sojourn_times)
    )


def print_table(title, network, result):
    print(f"\n{title}")
    print(f"  stop_reason = {result.stop_reason}   iterations = {result.iterations}"
          f"   sim_calls = {result.sim_calls}"
          f"   warm_start_iterations = {result.warm_start_iterations}"
          f"   converged = {result.converged}")
    if not result.converged:
        print(f"  NOT CONVERGED: residual = {result.residual:.3e} -- do not trust S*")
    header = f"  {'station':12s} {'gamma':>7s} {'S*':>9s} {'E[T]':>9s} {'zeta':>9s}"
    if result.sojourn_ci is not None:
        header += f" {'E[T] 95% CI':>24s}"
    print(header)
    for i, (st, S, t, z) in enumerate(zip(
        network.stations, result.capacities, result.sojourn_times, result.zeta
    )):
        row = f"  {st.name:12s} {st.gamma:7.4f} {S:9.4f} {t:9.4f} {z:9.4f}"
        if result.sojourn_ci is not None:
            entry = result.sojourn_ci[i]
            if entry is None:                    # no CI for this station (spec 8.1)
                row += f"   {'--':>24s}"
            else:
                lower, upper = entry
                row += f"   ({lower:.6f}, {upper:.6f})"
        print(row)
    print(f"  objective (sum w*E[T], w = 1)      = {result.objective:.6f}")
    print(f"  mean job sojourn (visit-weighted)  = "
          f"{visit_ratio_weighted(network, result.sojourn_times):.6f}   [diagnostic]")
    if result.system_response_time is not None:
        mean, (lower, upper) = result.system_response_time
        interval = (
            "CI unavailable" if lower is None or upper is None
            else f"CI ({lower:.6f}, {upper:.6f})"
        )
        print(f"  system response time = {mean:.6f} {interval}   [diagnostic]")


def print_summary(rows, budget):
    """One block comparing the workloads: objective and cumulative capacity per unit.

    Analytic only, regardless of whether a simulated pass also ran -- `rows` here always
    comes from run_analytic. The cumulative QPU and GPU capacities are the two axes of
    the paper's Figure 2.
    """
    print(f"\nANALYTIC SUMMARY at the shared budget C = {budget:.4f}")
    print(f"  {'workload':20s} {'C/floor':>8s} {'objective':>11s} {'mean job':>10s} "
          f"{'cum QPU':>9s} {'cum GPU':>9s} {'cum CPU':>9s}")
    for workload, network, result in rows:
        floor = min_feasible_budget(network.stations)
        by_unit = capacity_by_unit(network, result.capacities)
        print(f"  {workload:20s} {budget / floor:8.2f} {result.objective:11.6f} "
              f"{visit_ratio_weighted(network, result.sojourn_times):10.6f} "
              f"{by_unit['qpu']:9.4f} {by_unit['gpu']:9.4f} {by_unit['cpu']:9.4f}")


def run_analytic(budget):
    """One analytic optimization per workload. Returns [(workload, network, result)]."""
    rows = []
    for workload in WORKLOADS:
        network = build_qcsc_network(workload)
        floor = min_feasible_budget(network.stations)
        # Optimizer.__init__ raises on an infeasible budget too. This one is deliberately
        # redundant: `shared_budget` is the balanced floor scaled up, so when it fails it
        # fails for one specific workload, and the library's message cannot say which.
        if budget <= floor:
            raise ValueError(
                f"budget {budget} is not feasible for workload {workload!r}: its "
                f"minimum feasible budget is {floor}"
            )
        result = Optimizer(network, budget=budget).run()
        rows.append((workload, network, result))
        print_table(f"ANALYTIC  ({workload})", network, result)
    return rows


def print_gaps(workload, network, analytic, simulated):
    """simulated - analytic, per station, each at its own pass's S*.

    Where the gap exceeds the simulated CI half-width it is coupling the per-station
    closed form cannot see, not sampling noise (spec section 7) -- but only when it
    recurs at a specific, pre-specified station. Over these 14 rows at alpha = 0.05,
    ~0.7 flags per workload are expected by chance alone, so a single flag here should be
    read as noise, not coupling.
    """
    print(f"\nDIFFERENCE ({workload}): simulated - analytic, at each pass's own S*")
    print(f"  {'station':12s} {'analytic':>12s} {'simulated':>12s} {'gap':>10s} "
          f"{'gap %':>8s} {'> CI half-width?':>18s}")
    for i, (st, a, s) in enumerate(zip(
        network.stations, analytic.sojourn_times, simulated.sojourn_times
    )):
        gap = s - a
        entry = simulated.sojourn_ci[i] if simulated.sojourn_ci is not None else None
        if entry is None:
            verdict = "no CI"
        else:
            lower, upper = entry
            verdict = "yes" if abs(gap) > 0.5 * (upper - lower) else "no"
        print(f"  {st.name:12s} {a:12.6f} {s:12.6f} {gap:10.6f} "
              f"{100.0 * gap / a:7.2f}% {verdict:>18s}")


def run_simulated(url, budget, analytic_rows):
    """One simulated optimization per workload, on fresh Networks.

    Fresh Network objects rather than the analytic pass's: bind_gamma is idempotent for
    an equal value so reuse would work, but keeping the station objects independent means
    neither pass can observe the other's mutable state.
    """
    client = QsimClient(url, stopping=STOPPING, preflight=True)
    results = {}
    for workload, _, analytic in analytic_rows:
        network = build_qcsc_network(workload)
        simulated = Optimizer(
            network, budget=budget,
            analyzer=SimulationAnalyzer(network, client),
        ).run()
        results[workload] = simulated
        print_table(f"SIMULATED  ({workload})", network, simulated)
        print_gaps(workload, network, analytic, simulated)
        if simulated.degraded:
            print("  DEGRADED")
            for entry in simulated.degraded:
                print(f"    - {entry}")
    print("\n  fj_pp and fj_sp are excluded from the gamma-conservation check "
          "(qsim-service#8), so 2 of 14 stations have no throughput witness here.")
    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--dot" in argv:
        print(build_qcsc_network("balanced").to_dot(), end="")
        return None

    print(__doc__.strip().split("\n\n")[0])
    budget = shared_budget()
    print(f"\nlambda = {LAMBDA}   p11 = {P11}   p0 = {P0}   r = {R}   "
          f"costs: QPU {C_QPU:g} / GPU {C_GPU:g} / CPU {C_CPU:g}")
    print(f"shared budget C = {BUDGET_MULTIPLE:g} x the balanced floor = {budget:.4f}")

    rows = run_analytic(budget)
    print_summary(rows, budget)
    print("\n  (cumulative capacity: a fork-join's S counts on both sides, since both "
          "of its servers receive it)")

    url = os.environ.get("QOPT_QSIM_URL")
    if not url:
        print("\nSet QOPT_QSIM_URL=http://localhost:8080 to add the simulated pass. "
              "Analytic results only.")
        return {workload: result for workload, _, result in rows}
    return run_simulated(url, budget, rows)


if __name__ == "__main__":
    main()
