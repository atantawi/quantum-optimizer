# QCSC Example Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `examples/qcsc_network.py` — the paper's 14-station Quantum-Centric
SuperComputing network at one operating point under three workload variants — with offline
tests and one gated live-simulation test.

**Architecture:** One self-contained example module holding named parameter constants, a
`rates()` helper that is the entire difference between the three workloads, a
`build_qcsc_network()` factory returning a `qopt.Network` with all 14 stations and 18
routes, and a `main()` that runs the analytic optimizer for each workload and — when
`QOPT_QSIM_URL` is set — a second simulated pass. No change to `qopt/` itself.

**Tech Stack:** Python 3.12, `qopt` (this repo, editable install), `pytest`. Zero runtime
dependencies. Simulation reaches `qsim-service` over HTTP/JSON only.

**Spec:** `docs/superpowers/specs/2026-07-31-qcsc-example-network-design.md` — read it
before starting. Section references below (§3, §5.1, …) point into that spec.

## Global Constraints

- **No new runtime dependencies.** `qopt` declares zero; the example may import only
  `qopt` and the standard library (`os`, `sys`).
- **Do not modify `qopt/`.** The library is used exactly as it stands. If something seems
  to require a library change, stop and report rather than editing it.
- **Station names use single underscores only.** `Network._validate` rejects any name
  containing `__` (it would collide with qsim's internal `<node>__b0` / `<node>__join`).
- **Name prefixes are load-bearing.** Every station name starts with `cpu_`, `qpu_`,
  `gpu_`, or `fj_`; the reporting code groups on these prefixes. Task 2 tests that
  contract.
- **Tests must pass offline.** Everything in `tests/test_example_qcsc.py` runs without a
  simulation service. Live tests go in `tests/test_integration_qsim.py`, which is skipped
  unless `QOPT_QSIM_URL` is set.
- **Exact parameter values** (§4 of the spec), copied verbatim into the module:
  `LAMBDA = 0.9`, `P11 = 0.5`, `P0 = 0.5`, `R = 4.0`, `B_PP = B_SP = 1.0`,
  `B_PSQ = B_PSG = B_SSQ = B_SSG = 2.0`, `MU_CPU = 20.0`, `C_QPU = 4.0`, `C_GPU = 1.0`,
  `C_CPU = 1.0`, `BUDGET_MULTIPLE = 6.0`.
- **Run tests with:** `python -m pytest tests/ -q` from the repo root. If a virtualenv is
  present use its interpreter (`.venv/bin/python -m pytest ...`).

---

### Task 1: Topology, parameters, and the three workloads

Builds `build_qcsc_network()` and everything it needs. Deliverable: the network can be
constructed for all three workloads and every derived `γ` is correct. No optimizer, no
reporting yet.

**Files:**
- Create: `examples/qcsc_network.py`
- Test: `tests/test_example_qcsc.py`

**Interfaces:**
- Consumes: `qopt.{GG1Station, ForkJoinStation, Network, Route, min_feasible_budget}`.
- Produces, for Tasks 2–4:
  - `WORKLOADS: tuple[str, ...]` — `("balanced", "quantum_dominant", "classical_dominant")`
  - `rates(workload: str, b: float) -> tuple[float, float]` — returns `(mu_Q, mu_G)`
  - `build_qcsc_network(workload: str, *, c_qpu: float = C_QPU, c_gpu: float = C_GPU,
    c_cpu: float = C_CPU) -> qopt.Network`
  - `shared_budget(*, c_qpu=C_QPU, c_gpu=C_GPU, c_cpu=C_CPU) -> float`
  - Module constants `LAMBDA, P11, P0, R, B_PP, B_SP, B_PSQ, B_PSG, B_SSQ, B_SSG,
    MU_CPU, C_QPU, C_GPU, C_CPU, BUDGET_MULTIPLE`

The cost keyword arguments exist because §5.1's symmetry test must rebuild the same
topology under unit costs. Do not drop them in favour of reading module globals — the test
would then have to monkeypatch globals, which is worse.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_example_qcsc.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_example_qcsc.py -q`
Expected: every test FAILS with `ModuleNotFoundError: No module named 'examples.qcsc_network'`.

- [ ] **Step 3: Write the module**

Create `examples/qcsc_network.py`. The docstring's prediction paragraph is required, not
decorative — it is §7 of the spec and the reason the simulated pass exists.

```python
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

A prediction, stated up front and then demonstrated by the simulated pass:

  * cpu_init_ps and cpu_init_sp receive a Bernoulli split of a Poisson stream, which is
    still Poisson. Their cov_a = 1 is exactly right, so analytic and simulated agree.
  * Stream 1's sequential queues sit DOWNSTREAM OF A FORK-JOIN, so their arrivals are
    join completions (a max over two branches), not Poisson. This is where the analytic
    per-station form is unjustified.
  * fj_sp receives a SUPERPOSITION of two tandem departure streams while t_ul takes a
    Poisson arrival rate -- and at r = 4 t_ul carries heterogeneity bias of its own. Two
    error sources stacked.
  * Stream 2's sequential queues are fed by a CPU whose own input is Poisson, so they
    should track the analytic values closely.

The two streams therefore run the same phases in opposite order, which makes them a
controlled comparison for arrival-process coupling.

Run `python -m examples.qcsc_network` for the analytic tables; add QOPT_QSIM_URL for the
simulated pass, or `--dot` to print the topology as Graphviz DOT.
"""

from qopt import ForkJoinStation, GG1Station, Network, Route, min_feasible_budget

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
```

Import only what this task uses: `ForkJoinStation`, `GG1Station`, `Network`, `Route`,
`min_feasible_budget`. `Optimizer` arrives in Task 2 and `os` in Task 3, each in the task
that first needs it — no imports ahead of their use.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_example_qcsc.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 5: Run the whole suite for regressions**

Run: `python -m pytest tests/ -q`
Expected: PASS. No existing test touches the new module, so nothing should change.

- [ ] **Step 6: Commit**

```bash
git add examples/qcsc_network.py tests/test_example_qcsc.py
git commit -m "feat: the QCSC network topology and its three workloads"
```

---

### Task 2: Analytic optimization, reporting, and the symmetry test

Adds `main()`'s analytic path, the per-workload table, the cross-workload summary, and the
`--dot` flag. Deliverable: `python -m examples.qcsc_network` prints three complete tables.

**Files:**
- Modify: `examples/qcsc_network.py` (append; add `import sys` and extend the `qopt`
  import with `Optimizer`)
- Modify: `tests/test_example_qcsc.py` (append)

**Interfaces:**
- Consumes from Task 1: `WORKLOADS`, `build_qcsc_network`, `shared_budget`, `LAMBDA`,
  the cost constants.
- Produces for Task 3:
  - `UNIT_PREFIXES: tuple[str, ...]` — `("cpu_", "qpu_", "gpu_", "fj_")`
  - `capacity_by_unit(network, capacities) -> dict[str, float]` — keys `"cpu"`, `"qpu"`,
    `"gpu"`; a fork-join's `S` counts toward **both** `"qpu"` and `"gpu"`, because both
    of its servers receive that same `S`
  - `visit_ratio_weighted(network, sojourn_times) -> float` — `sum (gamma_i/LAMBDA) * E[T_i]`
  - `print_table(title, network, result) -> None`
  - `print_summary(rows, budget) -> None` where `rows` is
    `list[tuple[str, Network, Result]]`
  - `run_analytic(budget) -> list[tuple[str, Network, Result]]`
  - `main(argv=None) -> dict[str, Result] | None` (`None` for `--dot`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_example_qcsc.py`:

```python
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
        assert result.iterations == 5, workload
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
    objectives = [
        Optimizer(build_qcsc_network(w, c_qpu=1.0), budget=unit_budget).run().objective
        for w in ("quantum_dominant", "classical_dominant")
    ]
    # CORRECTED IN REVIEW -- do not copy this line. Asserting bit equality here is wrong:
    # `allocate`'s `slack` and `denom` are left-folds over the stations, the two variants
    # present those summands in a different order, and IEEE-754 addition is not
    # associative, so the bit equality holds for these values rather than by construction.
    # As implemented, the test asserts `sorted(...) == pytest.approx(sorted(...))` on the
    # sojourn vectors -- the permutation, which is the actual claim -- plus `approx` on the
    # objective. See spec 5.1 and tests/test_example_qcsc.py.
    assert objectives[0] == objectives[1]      # bitwise: a permutation of the same model

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


def test_capacity_by_unit_counts_a_fork_join_on_both_sides():
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_example_qcsc.py -q`
Expected: the 8 new tests FAIL with `ImportError: cannot import name 'UNIT_PREFIXES'` /
`'main'` etc. The 7 from Task 1 still PASS.

- [ ] **Step 3: Append the reporting and analytic driver**

Append to `examples/qcsc_network.py`:

```python
UNIT_PREFIXES = ("cpu_", "qpu_", "gpu_", "fj_")
"""Station-name prefixes. Load-bearing: capacity_by_unit groups on them."""


def capacity_by_unit(network, capacities):
    """Cumulative allocated capacity per processing-unit type.

    A fork-join's S counts toward BOTH 'qpu' and 'gpu': in qopt both servers of a
    fork-join receive that same S (see spec section 10 -- the paper instead sets
    S_2 = S_1/r, which qopt deliberately does not do).
    """
    totals = {"cpu": 0.0, "qpu": 0.0, "gpu": 0.0}
    for st, S in zip(network.stations, capacities):
        if st.name.startswith("fj_"):
            totals["qpu"] += S
            totals["gpu"] += S
        elif st.name.startswith("qpu_"):
            totals["qpu"] += S
        elif st.name.startswith("gpu_"):
            totals["gpu"] += S
        else:
            totals["cpu"] += S
    return totals


def visit_ratio_weighted(network, sojourn_times):
    """sum_i (gamma_i / LAMBDA) * E[T_i] -- the mean end-to-end job sojourn time.

    A diagnostic only. The OPTIMIZED objective uses omega_i = 1 (the paper's default),
    which is the plain sum of the 14 expected sojourn times and is a different quantity.
    """
    return sum(
        (st.gamma / LAMBDA) * t for st, t in zip(network.stations, sojourn_times)
    )


def print_table(title, network, result):
    print(f"\n{title}")
    print(f"  stop_reason = {result.stop_reason}   iterations = {result.iterations}"
          f"   sim_calls = {result.sim_calls}   converged = {result.converged}")
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

    The cumulative QPU and GPU capacities are the two axes of the paper's Figure 2.
    """
    print(f"\nSUMMARY at the shared budget C = {budget:.4f}")
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
        if budget <= floor:
            raise ValueError(
                f"budget {budget} is not feasible for workload {workload!r}: its "
                f"minimum feasible budget is {floor}"
            )
        result = Optimizer(network, budget=budget).run()
        rows.append((workload, network, result))
        print_table(f"ANALYTIC  ({workload})", network, result)
    return rows


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

    print("\nSet QOPT_QSIM_URL=http://localhost:8080 to add the simulated pass. "
          "Analytic results only.")
    return {workload: result for workload, _, result in rows}


if __name__ == "__main__":
    main()
```

`main()` deliberately does not consult `QOPT_QSIM_URL` in this task: the simulated pass
arrives whole in Task 3, which adds the environment check and `run_simulated` together.
Do **not** add a stub that raises `NotImplementedError`, and do not read the environment
variable and ignore it — either would be dead code or a silent no-op. This task's
deliverable is the complete analytic path, and that is a coherent thing on its own.

`import os` is therefore not needed until Task 3; add it there. Import `sys` here (the
`--dot` flag uses it).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_example_qcsc.py -q`
Expected: PASS, 15 tests. `test_main_runs_analytically_without_a_service` never reaches
the stub because `QOPT_QSIM_URL` is deleted.

- [ ] **Step 5: Run the example by hand and read the output**

Run: `python -m examples.qcsc_network`
Expected: three 14-row tables and a summary block. Check by eye: every `gamma` column
matches §3.1 of the spec, `converged = True` everywhere, and the summary's `C/floor`
column reads about 6.00 / 6.31 / 7.48.

Run: `python -m examples.qcsc_network --dot`
Expected: a `digraph` with 16 nodes (14 stations plus `src` and `snk`).

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add examples/qcsc_network.py tests/test_example_qcsc.py
git commit -m "feat: analytic optimization and reporting for the QCSC example"
```

---

### Task 3: The simulated pass

Replaces the Task 2 stub with the real simulated run: one `SimulationAnalyzer` per
workload on a fresh `Network`, then a per-station gap table. Deliverable: with
`QOPT_QSIM_URL` set the example prints simulated tables and the analytic-vs-simulated
gaps; without it, behaviour is unchanged.

**Files:**
- Modify: `examples/qcsc_network.py` (replace the `run_simulated` stub)
- Modify: `tests/test_example_qcsc.py` (append)

**Interfaces:**
- Consumes from Task 2: `print_table`, `run_analytic`'s row shape
  `(workload, network, result)`, `WORKLOADS`, `build_qcsc_network`.
- Produces: `STOPPING: dict`, `run_simulated(url, budget, analytic_rows) -> dict[str, Result]`,
  `print_gaps(workload, network, analytic, simulated) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_example_qcsc.py`:

```python
def test_stopping_rule_matches_the_existing_simulated_example():
    """Same rule as examples/simulated_mixed_network.py, so run times are comparable."""
    from examples.qcsc_network import STOPPING

    assert STOPPING == {
        "alpha": 0.05, "precision": 0.02, "minSamples": 100000,
        "maxSamples": 4000000, "maxWallClockSeconds": 300,
    }


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_example_qcsc.py -q`
Expected: the 2 new tests FAIL with `ImportError: cannot import name 'STOPPING'` /
`'print_gaps'`.

Note: the keyword arguments above were checked against the real dataclass, whose fields
are `capacities`, `sojourn_times`, `zeta`, `objective`, `iterations`, `converged`,
`residual`, `degraded` (all required except `degraded`, which has a default factory),
plus `sojourn_ci=None`, `noise_floor=None`, `stop_reason="tol"`,
`warm_start_iterations=0`, `system_response_time=None`, `sim_calls=0`. The existing
`tests/test_example_simulated.py::test_mixed_network_table_prints_a_system_measure_with_no_ci`
constructs one the same way and is the reference.

- [ ] **Step 3: Add the simulated pass**

In `examples/qcsc_network.py`, add `import os` above `import sys` and extend the `qopt`
import with the simulation names:

```python
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
```

Add near the other constants:

```python
STOPPING = {
    "alpha": 0.05,
    "precision": 0.02,
    "minSamples": 100000,
    "maxSamples": 4000000,
    "maxWallClockSeconds": 300,
}
"""Same stopping rule as examples/simulated_mixed_network.py."""
```

Add these two functions above `main()`:

```python
def print_gaps(workload, network, analytic, simulated):
    """simulated - analytic, per station, each at its own pass's S*.

    Where the gap exceeds the simulated CI half-width it is coupling the per-station
    closed form cannot see, not sampling noise (spec section 7).
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
```

Then change the tail of `main()` — Task 2 left it printing the hint unconditionally — so
that it consults the environment and dispatches to `run_simulated`:

```python
    url = os.environ.get("QOPT_QSIM_URL")
    if not url:
        print("\nSet QOPT_QSIM_URL=http://localhost:8080 to add the simulated pass. "
              "Analytic results only.")
        return {workload: result for workload, _, result in rows}
    return run_simulated(url, budget, rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_example_qcsc.py -q`
Expected: PASS, 17 tests.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/qcsc_network.py tests/test_example_qcsc.py
git commit -m "feat: the simulated pass for the QCSC example"
```

- [ ] **Step 7: Run the example against a live service, in the background**

This is the slow one: roughly 18 POSTs, up to ~90 minutes at the wall-clock cap. Start
`qsim-service` first (its Docker image), then run it detached and read the log when it
lands — do not block on it:

```bash
QOPT_QSIM_URL=http://localhost:8080 python -m examples.qcsc_network > /tmp/qcsc_sim.log 2>&1
```

Expected in the log: three SIMULATED tables and three gap tables. Check the §7
prediction against the `gap %` column — stream 1's sequential queues and `fj_sp` should
show the larger gaps, the two `cpu_init_*` stations the smallest. Report what the numbers
actually say, including if they contradict the prediction.

---

### Task 4: Gated live test and documentation

**Files:**
- Modify: `tests/test_integration_qsim.py` (append one test)
- Modify: `README.md` (the "Simulated evaluation" section's list of runnable examples,
  around line 163)

**Interfaces:**
- Consumes: `examples.qcsc_network.{build_qcsc_network, shared_budget}`, and the module-level
  `QSIM_URL` / `STOPPING` / `pytestmark` already in `tests/test_integration_qsim.py`.
- Produces: nothing.

- [ ] **Step 1: Write the gated test**

Append to `tests/test_integration_qsim.py`:

```python
def test_qcsc_network_evaluates_all_fourteen_stations():
    """Fourteen nodes with two fork-joins is new ground for the serializer: one POST."""
    from examples.qcsc_network import build_qcsc_network, shared_budget
    from qopt.optimizer import Optimizer

    network = build_qcsc_network("quantum_dominant")
    analytic = Optimizer(network, budget=shared_budget()).run()

    client = QsimClient(QSIM_URL, stopping=STOPPING, preflight=True)
    evaluation = SimulationAnalyzer(network, client).evaluate(
        network.stations, analytic.capacities
    )
    assert len(evaluation.sojourn_times) == 14
    assert all(t > 0 for t in evaluation.sojourn_times)
```

- [ ] **Step 2: Verify it is skipped without the service**

Run: `python -m pytest tests/test_integration_qsim.py -q`
Expected: all tests SKIPPED, reason "set QOPT_QSIM_URL to run live qsim-service tests".

- [ ] **Step 3: Run it against a live service**

Run: `QOPT_QSIM_URL=http://localhost:8080 python -m pytest tests/test_integration_qsim.py::test_qcsc_network_evaluates_all_fourteen_stations -v`
Expected: PASS. If it fails, report the failure — do not weaken the assertion.

- [ ] **Step 4: Update the README**

In `README.md`, in the "Simulated evaluation" section, extend the runnable-versions line
to name the new example:

```markdown
Runnable versions: `examples/simulated_tandem.py`,
`examples/simulated_mixed_network.py`, and `examples/qcsc_network.py` — the paper's
14-station QCSC network under three workloads (balanced, quantum-dominant,
classical-dominant). All fall back to analytic-only output when `QOPT_QSIM_URL` is unset.
```

And add to the "See also" list:

```markdown
- `docs/superpowers/specs/2026-07-31-qcsc-example-network-design.md` — the QCSC example
  network (topology, workloads, budget). Implemented; see `examples/qcsc_network.py`.
```

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, with the live tests skipped.

- [ ] **Step 6: Commit**

```bash
git add tests/test_integration_qsim.py README.md
git commit -m "test: a gated live check of the 14-station QCSC model, and README"
```

---

## Self-Review

**Spec coverage.** §3 topology → Task 1. §4 parameters → Task 1 (constants) and the
Global Constraints. §5 workloads → Task 1 (`rates`, `_fork_join`). §5.1 symmetry → Task 2
(`test_unit_costs_collapse_the_two_dominant_workloads`). §6 budget and weights → Task 1
(`shared_budget`) and Task 2 (`visit_ratio_weighted`, the objective pins). §7 prediction →
Task 1 docstring, checked against real numbers in Task 3 Step 7. §8 output → Task 2
(`print_table`, `print_summary`, `--dot`) and Task 3 (`print_gaps`). §8.1 degradation →
Task 2 (infeasibility raise, non-convergence line) and Task 3 (`None`-CI guard, degraded
list, fork-join conservation note). §9 tests → Tasks 1–4. §10 divergences → documented in
`capacity_by_unit`'s docstring, which is where the both-servers-share-S rule shows up in
behaviour.

**Placeholders.** None. The one deliberate stub (`run_simulated` raising
`NotImplementedError` at the end of Task 2) exists only so Task 2 is independently
testable, and Task 3 Step 3 replaces it.

**Type consistency.** `build_qcsc_network(workload, *, c_qpu, c_gpu, c_cpu)` and
`shared_budget(*, c_qpu, c_gpu, c_cpu)` take the same keywords, and Task 2's symmetry test
calls both with `c_qpu=1.0` only. `run_analytic` returns
`[(workload, network, result)]`, which is exactly what `print_summary` and
`run_simulated` consume. `capacity_by_unit` returns keys `"cpu"`, `"qpu"`, `"gpu"`, used
in `print_summary` as `by_unit['qpu']`. `print_gaps(workload, network, analytic,
simulated)` is called with that argument order in both `run_simulated` and its test.

**One risk worth naming.** The `EXPECTED_OBJECTIVE` and `visit_ratio_weighted` pins came
from a prototype of this same topology, not from an independent derivation, so they are
regression guards rather than oracles — the comment above them says so. The independent
checks are `test_min_feasible_budget_matches_hand_computation` (literal arithmetic from
the spec's parameters), `test_topology_names_and_derived_gamma` (hand-computed traffic
equations), and the two symmetry assertions (exact relations, no magic numbers). If a pin
and an oracle ever disagree, the oracle wins.
