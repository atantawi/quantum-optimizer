"""Variability propagation with the fewest moving parts: M/D/1 -> M/M/1 in series.

A Poisson source at lambda_0 = 1.0 feeds a deterministic-service station, whose
departure process is NOT Poisson, which then feeds an exponential-service station:

    source (lambda_0 = 1.0) -> shape (M/D/1) -> serve (M/M/1) -> sink

The analytic path evaluates each station independently from the cov_a it was given, so
`serve` is analyzed as if its arrivals were Poisson. They are not: a deterministic
server smooths the stream it passes on. Simulating the whole network captures that
coupling, and the printed difference at `serve` is exactly the effect per-station
analysis cannot represent.

Set QOPT_QSIM_URL to compare against a live qsim-service; without it the example prints
the analytic table alone.
"""

import os

from qopt import (
    GG1Station,
    Network,
    Optimizer,
    QsimClient,
    Route,
    SimulationAnalyzer,
    min_feasible_budget,
)

BUDGET_MULTIPLE = 3.0


def build_network():
    stations = [
        GG1Station.md1(mu=1.0, c=1.0, name="shape"),
        GG1Station.mm1(mu=1.0, c=1.0, name="serve"),
    ]
    routes = [
        Route(Network.SOURCE, "shape"),
        Route("shape", "serve"),
        Route("serve", Network.SINK),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-simulated-tandem")


def _print_table(title, network, result, budget):
    print(f"\n{title}")
    print(f"  budget = {budget:.4f}   stop_reason = {result.stop_reason}"
          f"   iterations = {result.iterations}   sim_calls = {result.sim_calls}")
    header = f"  {'station':10s} {'gamma':>7s} {'S*':>9s} {'E[T]':>9s} {'zeta':>9s}"
    if result.sojourn_ci is not None:
        header += f" {'E[T] 95% CI':>22s}"
    print(header)
    for i, (st, S, t, z) in enumerate(zip(
        network.stations, result.capacities, result.sojourn_times, result.zeta
    )):
        row = f"  {st.name:10s} {st.gamma:7.4f} {S:9.4f} {t:9.4f} {z:9.4f}"
        if result.sojourn_ci is not None:
            entry = result.sojourn_ci[i]
            if entry is None:
                row += f"   {'--':>22s}"     # no CI for this station (spec 7.1)
            else:
                lower, upper = entry
                row += f"   ({lower:.4f}, {upper:.4f})"
        print(row)
    print(f"  objective (sum w*E[T]) = {result.objective:.6f}")


def result_budget(network):
    return BUDGET_MULTIPLE * min_feasible_budget(network.stations)


def main():
    print(__doc__.strip().split("\n\n")[0])

    network = build_network()
    budget = result_budget(network)
    analytic = Optimizer(network, budget=budget).run()
    _print_table("ANALYTIC (independent stations)", network, analytic, budget)

    url = os.environ.get("QOPT_QSIM_URL")
    if not url:
        print("\nSet QOPT_QSIM_URL=http://localhost:8080 to add the simulated "
              "comparison. Analytic results only.")
        return analytic

    # A fresh Network: not because gamma can't be rebound (bind_gamma is idempotent for
    # an equal value, so reusing `network`'s stations would succeed) but to keep the two
    # runs' station objects fully independent, so neither run can observe the other's
    # mutable state (e.g. gamma binding order, or any future per-station mutation).
    simulated_network = build_network()
    client = QsimClient(url, preflight=True)
    analyzer = SimulationAnalyzer(simulated_network, client)
    simulated = Optimizer(
        simulated_network, budget=budget, analyzer=analyzer
    ).run()
    _print_table("SIMULATED (whole network)", simulated_network, simulated, budget)

    print("\nDIFFERENCE (simulated - analytic)")
    print(f"  {'station':10s} {'E[T] analytic':>14s} {'E[T] simulated':>15s} {'gap':>10s}")
    for st, a, s in zip(
        network.stations, analytic.sojourn_times, simulated.sojourn_times
    ):
        print(f"  {st.name:10s} {a:14.6f} {s:15.6f} {s - a:10.6f}")
    print("\n'shape' sees genuinely Poisson arrivals, so its cov_a = 1 is exact and the "
          "two paths should agree closely. 'serve' does not: its arrivals are the "
          "departures of a deterministic server, so any gap there is variability "
          "propagation, which per-station analysis cannot represent.")
    if simulated.degraded:
        print("\nDEGRADED")
        for entry in simulated.degraded:
            print(f"  - {entry}")
    return simulated


if __name__ == "__main__":
    main()
