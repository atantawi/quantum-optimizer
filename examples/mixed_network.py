"""Sample mixed network (spec 4.1.1): two single-server queues feeding a fork-join.

    source (lambda_0 = 1.0)
       +- 0.6 -> mm1 --+- 0.5 -> fj - 1.0 -> sink
       +- 0.4 -> md1 --+
                       +- 0.5 -> sink

The gammas (0.6, 0.4, 0.5) are the traffic-equation solution of this topology, so they
are derived here rather than hand-supplied. Every printed number is unchanged from the
version that supplied them by hand; only the station labels differ.

The labels were shortened by choice, not by requirement: 4.2 rejects only names that are
empty, non-unique, contain `__`, or collide with the reserved `src`/`snk`, so the previous
`"ingest (M/M/1)"` would have validated fine. Short identifiers simply read better once a
name doubles as a JSON node key and a DOT identifier.
"""

from qopt import ForkJoinStation, GG1Station, Network, Optimizer, Route, min_feasible_budget


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


def main():
    network = build_network()
    budget = 6 * min_feasible_budget(network.stations)
    result = Optimizer(network, budget=budget).run()

    print(f"budget = {budget:.4f}   converged = {result.converged} "
          f"in {result.iterations} iterations")
    print(f"{'station':22s} {'gamma':>8s} {'S*':>10s} {'E[T]':>10s} {'zeta':>10s}")
    for st, S, t, z in zip(
        network.stations, result.capacities, result.sojourn_times, result.zeta
    ):
        print(f"{st.name:22s} {st.gamma:8.4f} {S:10.4f} {t:10.4f} {z:10.4f}")
    print(f"objective (sum w*E[T]) = {result.objective:.6f}")
    return result


if __name__ == "__main__":
    main()
