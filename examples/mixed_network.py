"""Sample mixed network: two single-server queues and one fork-join station."""

from qopt import (
    ForkJoinStation,
    GG1Station,
    Optimizer,
    min_feasible_budget,
)


def build_network():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="ingest (M/M/1)"),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="transform (M/D/1)"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fork-join"),
    ]


def main():
    stations = build_network()
    budget = 6 * min_feasible_budget(stations)
    result = Optimizer(stations, budget=budget).run()

    print(f"budget = {budget:.4f}   converged = {result.converged} "
          f"in {result.iterations} iterations")
    print(f"{'station':22s} {'S*':>10s} {'E[T]':>10s} {'zeta':>10s}")
    for st, S, t, z in zip(
        stations, result.capacities, result.sojourn_times, result.zeta
    ):
        print(f"{st.name:22s} {S:10.4f} {t:10.4f} {z:10.4f}")
    print(f"objective (sum w*E[T]) = {result.objective:.6f}")
    return result


if __name__ == "__main__":
    main()
