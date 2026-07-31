# Queueing Network Capacity Allocation Optimizer

A Python optimizer that allocates resource capacities `S` across a **network of queues** to
minimize the sum of weighted expected sojourn times subject to a budget and stability
constraints. It implements the Section-5 fixed-point iteration of the analysis paper
(`docs/analysis.pdf`, "Optimization and Performance Analysis of Resource Allocation in
Quantum-Centric Supercomputing Environments").

## Model

The network is a collection of **stations**. Two types:

- **Single-server queue** — a G/G/1 queue analyzed with the Kingman / Allen–Cunneen
  mean-value approximation, parameterized by the coefficients of variation of interarrival
  (`cov_a`) and service (`cov_s`) times. M/M/1 (`cov_a=1, cov_s=1`) and M/D/1
  (`cov_a=1, cov_s=0`) are presets; the approximation is exact for any M/G/1.
- **Fork-join queue** — two parallel servers (ratio `r ≥ 1`), analyzed with the UL
  (upper–lower bound interpolation) approximation.

Arrival rates `γ` are fixed per-station constants; the optimizer iterates on the capacity
vector `S` until the optimal `S*` is reached.

## Architecture

```mermaid
flowchart LR
    NET["<b>Network</b><br/>topology + solve_traffic<br/>→ derived γ"] --> IN
    IN["Stations (γ, μ, weight)<br/>+ budget"] --> OPT

    subgraph LOOP["fixed-point loop — until ‖ΔS‖∞ &lt; tol"]
        direction LR
        OPT["<b>Optimizer</b><br/>driver / convergence"]
        ALLOC["<b>allocator</b><br/>eq 21 · min_feasible_budget"]
        AN["<b>Analyzer</b> seam<br/>E[T]ᵢ, ζᵢ for all stations"]
        OPT -- "allocate(ζ)" --> ALLOC
        ALLOC -- "capacities S" --> OPT
        OPT -- "S" --> AN
        AN -- "ζᵢ = E[T]ᵢ·(Sᵢμᵢ − γᵢ)" --> OPT
    end

    OPT --> RES["<b>Result</b><br/>S* · E[T] · objective · converged"]

    AN -. implementation .-> ANLY["<b>AnalyticAnalyzer</b><br/>per-station closed form"]
    ANLY -. subtypes .-> GG1["GG1Station<br/>M/M/1 · M/D/1"]
    ANLY -. subtypes .-> FJ["ForkJoinStation<br/>→ t_ul (UL bound)"]
    AN -. implementation .-> SIM["<b>SimulationAnalyzer</b><br/>qopt/qsim/: spec.py →<br/>client.py → measures.py"]
    SIM -- "HTTP/JSON POST /simulate" --> QSIM(["<b>qsim-service</b><br/>external, GPL v2<br/>reached only over HTTP"]):::external

    classDef external stroke-dasharray: 4 4,fill:#f6f6f6;
```

Each iteration re-allocates capacities from the current `ζ` (eq 21), then recomputes `ζ`
from the resulting capacities (eq 22); the loop repeats until the capacity vector stops
moving. Stations are the pluggable analyzer layer — each owns its own queueing math behind
the `Station` interface. Whole-network simulation plugs in one level *up*, at a network-level
`Analyzer` seam, because a simulation answers for every station in a single run rather than
station by station: `AnalyticAnalyzer` calls each station's own closed form, while
`SimulationAnalyzer` serializes the whole `Network` (topology and derived `γ`) into a
qsim-service request, issues one `POST /simulate` per optimizer iteration, and translates
the response's measures back into the same `(E[T], ζ)` shape — so the allocator and the loop
never know which analyzer is running.

## Scope & limitations

By default, each station is analyzed **independently** (`AnalyticAnalyzer`) from its own
arrival rate and coefficients of variation — exact for M/M/1 and M/G/1, an
**approximation** for a general network.

- **Network coupling is supported via simulation.** Per-station closed forms cannot
  capture how one station's departure process shapes the arrival variability of the
  stations downstream of it. `SimulationAnalyzer` obtains `E[T]` from a discrete-event
  simulation of the whole network (one `POST /simulate` per optimizer iteration) via
  [`qsim-service`](https://github.com/atantawi/qsim-service), so that coupling is
  captured directly. `qopt` speaks HTTP/JSON only and declares zero runtime
  dependencies; the service is GPL v2 and stays behind that boundary.
- **Single open chain only.** One customer chain enters from a source and departs to a
  sink, so every `γᵢ` is exogenously determined and fixed across iterations. Closed
  chains would make `λ` depend on `S` through throughput, moving eq 21's budget floor
  underneath the optimizer; multi-class networks need a per-class notion eq 21 does not
  have. Both are honest open limitations, not oversights.
- **Fork-join measures other than `response-time` are diagnostics only.** JMT defines
  just two fork-join region measures, so `utilization`, `queue-length`, and friends
  report join-station numbers at a fork-join node
  ([qsim-service#8](https://github.com/atantawi/qsim-service/issues/8)). Nothing
  outside `response-time` enters eq 22, so this constrains reporting, not results.

See
[`docs/superpowers/specs/2026-07-29-simulation-support-design.md`](docs/superpowers/specs/2026-07-29-simulation-support-design.md)
for the full design.

## Status

Implemented. Core library (`qopt`) with single-server (G/G/1) and fork-join stations,
the eq-21 allocator, and the fixed-point `Optimizer`. Test suite passes.

## Usage

Install the package (editable) so `qopt` is importable:

```
pip install -e .
```

```python
from qopt import GG1Station, ForkJoinStation, Optimizer, min_feasible_budget

stations = [
    GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="ingest"),
    GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="transform"),
    ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fork-join"),
]
result = Optimizer(stations, budget=6 * min_feasible_budget(stations)).run()
print(result.capacities, result.objective, result.converged)
```

See `examples/mixed_network.py` for a runnable version. Running it prints the optimal
allocation (illustrative output; regenerate with `python -m examples.mixed_network`):

```
budget = 15.6000   converged = True in 6 iterations
station                        S*       E[T]       zeta
ingest (M/M/1)             2.9601     0.4237     1.0000
transform (M/D/1)          3.6448     0.2913     0.9451
fork-join                  3.0175     0.4520     1.1378
objective (sum w*E[T]) = 1.166933
```

Note the M/M/1 station's `zeta = 1.0000` exactly, while the M/D/1 and fork-join stations
have load-dependent `zeta` — which is what makes the fixed-point iteration necessary.

If the fixed point is not reached within `max_iter`, `run()` still returns a `Result` (the
last iterate) but sets `converged=False`, records the final `residual` (`‖Sₖ₊₁−Sₖ‖∞`), and
emits a `RuntimeWarning` — inspect `result.converged` before trusting the allocation.

### Simulated evaluation

```python
from qopt import (GG1Station, Network, Optimizer, QsimClient, Route,
                  SimulationAnalyzer, min_feasible_budget)

network = Network(
    [GG1Station.md1(mu=1.0, c=1.0, name="shape"),
     GG1Station.mm1(mu=1.0, c=1.0, name="serve")],
    [Route(Network.SOURCE, "shape"), Route("shape", "serve"),
     Route("serve", Network.SINK)],
    arrival_rate=1.0,
)                                        # gamma is derived from the topology
client = QsimClient("http://localhost:8080", preflight=True)
result = Optimizer(
    network,
    budget=3 * min_feasible_budget(network.stations),
    analyzer=SimulationAnalyzer(network, client),
).run()
print(result.capacities, result.sojourn_ci, result.sim_calls, result.stop_reason)
```

Runnable versions: `examples/simulated_tandem.py` and
`examples/simulated_mixed_network.py`. Both fall back to analytic-only output when
`QOPT_QSIM_URL` is unset.

See also:

- `docs/superpowers/specs/2026-07-10-optimizer-design.md` — authoritative design spec.
- `docs/superpowers/specs/2026-07-29-simulation-support-design.md` — simulation support
  (topology, `Analyzer` seam, `qsim-service` client). Implemented; see `SimulationAnalyzer`.
- `docs/optimizer-brainstorm-summary.md` — problem statement and design rationale.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
