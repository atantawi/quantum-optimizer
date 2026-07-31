"""The spec 4.1.1 network solved analytically and by simulation, side by side.

    source (lambda_0 = 1.0)
       +- 0.6 -> mm1 --+- 0.5 -> fj - 1.0 -> sink
       +- 0.4 -> md1 --+
                       +- 0.5 -> sink

gamma is DERIVED from this topology on both paths, so the printed difference is
attributable to variability propagation alone rather than to differing arrival rates.

A prediction, stated up front and then demonstrated:

  * mm1 and md1 receive a Poisson stream split by Bernoulli probabilities, which is
    still Poisson. Their cov_a = 1 is exactly right, so analytic and simulated should
    agree closely.
  * fj receives a thinned SUPERPOSITION of two departure streams, which is not Poisson
    — yet t_ul takes a Poisson arrival rate. So fj is precisely where the analytic
    approximation is unjustified and where simulation should visibly diverge.

This example also prints the HETEROGENEOUS t_ul-vs-simulation comparison, with the
simulated CI alongside each number so a reader can judge whether a gap is approximation
bias or sampling noise. It is printed and discussed rather than asserted: t_ul is exact
only for equal branch rates, so the heterogeneous gap is genuine bias, and any bracket
test on it either fails a correct run (tight precision) or passes regardless (loose
precision). See spec 8.2.

Set QOPT_QSIM_URL to add the simulated columns; without it the analytic table is printed
alone.
"""

import os

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
from qopt.forkjoin_approx import t_ul

BUDGET_MULTIPLE = 6.0


def build_network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6),
        Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5),
        Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5),
        Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def _print_table(title, network, result):
    print(f"\n{title}")
    print(f"  stop_reason = {result.stop_reason}   iterations = {result.iterations}"
          f"   sim_calls = {result.sim_calls}"
          f"   warm_start_iterations = {result.warm_start_iterations}")
    header = f"  {'station':6s} {'gamma':>7s} {'S*':>9s} {'E[T]':>9s} {'zeta':>9s}"
    if result.sojourn_ci is not None:
        header += f" {'E[T] 95% CI':>22s}"
    print(header)
    for i, (st, S, t, z) in enumerate(zip(
        network.stations, result.capacities, result.sojourn_times, result.zeta
    )):
        row = f"  {st.name:6s} {st.gamma:7.4f} {S:9.4f} {t:9.4f} {z:9.4f}"
        if result.sojourn_ci is not None:
            entry = result.sojourn_ci[i]
            if entry is None:
                row += f"   {'--':>22s}"     # no CI for this station (spec 7.1)
            else:
                lower, upper = entry
                row += f"   ({lower:.6f}, {upper:.6f})"
        print(row)
    print(f"  objective (sum w*E[T]) = {result.objective:.6f}")
    if result.system_response_time is not None:
        mean, (lower, upper) = result.system_response_time
        interval = (                             # bounds absent: measures.extract warned
            "CI unavailable" if lower is None or upper is None
            else f"CI ({lower:.6f}, {upper:.6f})"
        )
        print(f"  system response time = {mean:.6f} "
              f"{interval}   [diagnostic, not optimized]")


def main():
    print(__doc__.strip().split("\n\n")[0])

    network = build_network()
    budget = BUDGET_MULTIPLE * min_feasible_budget(network.stations)
    print(f"\nbudget = {budget:.4f}   derived gamma = "
          f"{tuple(st.gamma for st in network)}")

    analytic = Optimizer(network, budget=budget).run()
    _print_table("ANALYTIC (independent stations)", network, analytic)

    url = os.environ.get("QOPT_QSIM_URL")
    if not url:
        print("\nSet QOPT_QSIM_URL=http://localhost:8080 to add the simulated "
              "comparison. Analytic results only.")
        return analytic

    # A fresh Network: not because gamma can't be rebound (bind_gamma is idempotent for
    # an equal value, so reusing `network`'s stations would succeed) but to keep the two
    # runs' station objects fully independent, so neither run can observe the other's
    # mutable state.
    simulated_network = build_network()
    client = QsimClient(url, stopping={
        "alpha": 0.05, "precision": 0.02, "minSamples": 100000,
        "maxSamples": 4000000, "maxWallClockSeconds": 300,
    }, preflight=True)
    simulated = Optimizer(
        simulated_network, budget=budget,
        analyzer=SimulationAnalyzer(simulated_network, client),
    ).run()
    _print_table("SIMULATED (whole network)", simulated_network, simulated)

    print("\nDIFFERENCE (simulated - analytic), at each path's own S*")
    print(f"  {'station':6s} {'E[T] analytic':>14s} {'E[T] simulated':>15s} "
          f"{'gap':>10s} {'gap %':>8s}")
    for st, a, s in zip(
        network.stations, analytic.sojourn_times, simulated.sojourn_times
    ):
        print(f"  {st.name:6s} {a:14.6f} {s:15.6f} {s - a:10.6f} "
              f"{100.0 * (s - a) / a:7.2f}%")

    # The heterogeneous t_ul cross-check: printed and discussed, never asserted (8.2).
    fj = simulated_network.stations[-1]
    S_fj = simulated.capacities[-1]
    approximation = t_ul(fj.gamma, S_fj * fj.mu, S_fj * fj.r * fj.mu)
    measured = simulated.sojourn_times[-1]
    fj_ci = simulated.sojourn_ci[-1]
    gap = measured - approximation
    print(f"\nFORK-JOIN: t_ul vs simulation at S* = {S_fj:.6f} "
          f"(branch rates {S_fj * fj.mu:.4f} and {S_fj * fj.r * fj.mu:.4f}, r = {fj.r:g})")
    print(f"  t_ul (heterogeneous, approximate) = {approximation:.6f}")
    print(f"  gap                               = {gap:+.6f} "
          f"({100.0 * gap / approximation:+.2f}%)")
    if fj_ci is None:
        print(f"  simulated                         = {measured:.6f} CI unavailable")
        print("  No CI for 'fj', so this run cannot separate approximation bias from "
              "sampling noise (spec 7.1).")
    else:
        lower, upper = fj_ci
        half_width = 0.5 * (upper - lower)
        print(f"  simulated                         = {measured:.6f} "
              f"CI ({lower:.6f}, {upper:.6f}), half-width {half_width:.6f}")
        if abs(gap) > half_width:
            print("  The gap exceeds the CI half-width, so it is approximation bias, not "
                  "noise: t_ul is exact only for equal branch rates (r = 1), and r "
                  f"= {fj.r:g} here.")
        else:
            print("  The gap is within the CI half-width, so this run cannot separate "
                  "approximation bias from sampling noise. Tighten `precision` to see it.")
    print("  This comparison is deliberately not an acceptance test — see spec 8.2.")

    if simulated.degraded:
        print("\nDEGRADED")
        for entry in simulated.degraded:
            print(f"  - {entry}")
    return simulated


if __name__ == "__main__":
    main()
