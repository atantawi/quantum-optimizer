# Handoff — Simulation Support (session state, 2026-07-29)

Resume point for the simulation-support work. The **spec is finished and committed**; the
**implementation plan is not yet written**. That is the next action.

## Where things stand

| | |
|---|---|
| Branch | `docs/simulation-support-design` (clean, pushed) |
| Spec | `docs/superpowers/specs/2026-07-29-simulation-support-design.md` — complete, 3 commits |
| PR | [#3](https://github.com/atantawi/quantum-optimizer/pull/3) → `main`, docs-only, open |
| Tracking issue | [#2](https://github.com/atantawi/quantum-optimizer/issues/2) |
| External blocker | [qsim-service#5](https://github.com/atantawi/qsim-service/issues/5) — fork-join measure wiring |
| Tests | `48 passed` (unchanged; all work so far is docs) |

**Next action:** write `docs/superpowers/plans/2026-07-29-simulation-support.md` using the
`superpowers:writing-plans` skill, then offer subagent-driven vs. inline execution.

Nothing has been implemented. No source file under `qopt/` has been modified.

## Decisions locked with the user (do not re-litigate)

| Question | Decision |
|---|---|
| Where does `γ` come from? | **Derived from topology.** Single write point in `Network.__init__` |
| Convergence under noise | **CI-aware stop + damping**, warm-started from the analytic fixed point. Every mechanism must have an off switch that reproduces today's naive loop |
| Topology vocabulary | **Single open class, arbitrary probabilistic routing** including feedback cycles |
| "Open chain" | Means **open (not closed) class** — terminology only. *Not* a linear-series restriction |
| Degraded qsim results | **Warn + proceed, record in `Result`**; `strict=True` raises |
| Topology format | **Mirror qsim's `model` block.** No NetworkX — keeps `dependencies = []` |
| Fork-join blocker | **Fix `qsim-service` first.** Sequence FJ measure reading last in the plan, gated on qsim-service#5 |

## Facts verified against the code (don't re-derive)

**Call-site safety for the planned signature changes:**

- All `Station` subclass construction in `tests/` and `examples/` is **keyword-based**
  (`grep` found only `ForkJoinStation(**kwargs)`). So `__init__(self, gamma=None, mu=None, ...)`
  with an explicit "mu is required" guard is backward compatible.
- `Optimizer` is called as `Optimizer(stations, budget=...)` — **first arg positional** in every
  call site. So keep the parameter *named* `stations` and let it accept either a `Station`
  sequence or a `Network`; do not rename it to `target`.
- `Result(...)` is constructed in exactly one place, `qopt/optimizer.py:92`. Appending new
  fields **with defaults** is safe.

**§4.1.1 arithmetic — confirmed by running the real code, not asserted:**

- Traffic solve for `src →0.6 mm1 / →0.4 md1`, both `→0.5 fj / →0.5 snk`, `fj →1.0 snk`
  converges to exactly `(0.6, 0.4, 0.5)` in **3 iterations**.
- `min_feasible_budget = 2.6`; `6 × 2.6 = 15.6` — the committed README budget.
- Analytic result reproduces the README table digit-for-digit:
  `S* = 2.9601 / 3.6448 / 3.0175`, `E[T] = 0.4237 / 0.2913 / 0.4520`,
  `ζ = 1.0000 / 0.9451 / 1.1378`, `objective = 1.166933`.

**Cross-repo findings (qsim-service):**

- `MeasureMapper.SUPPORTED` = `response-time, residence-time, queue-time, queue-length,
  utilization, throughput, drop-rate, system-response-time`. **No `fork-join-response-time`.**
- `JsimgWriter.writeForkJoin` expands one fork-join node into `fj` (Fork: `Queue` +
  `ServiceTunnel` + `Fork`, **no server**), branch Server stations `fj__b0`/`fj__b1` at `S·µ`
  and `S·r·µ`, and `fj__join` (Join).
- `joinStationName` / `branchStationName` are referenced **only inside `JsimgWriter`** — the
  "so the measure mapper can remap onto it" comment describes an unimplemented intention.
- **JMT does have the metric**: `SimConstants.FORK_JOIN_RESPONSE_TIME = 27`,
  `FORK_JOIN_NUMBER_OF_JOBS = 26`, and the JSIMG `type` string `"Fork Join Response Time"` is in
  the jar. So the spec's `fork-join-response-time` mapping is correct; only qsim's wiring is
  missing. (An interim spec revision wrongly retargeted this to plain `response-time` — that was
  corrected in commit `ecc1380`.)
- `SIMmodeldefinition.xsd` `<measure>` attrs: `name`, `alpha`, `precision`, `verbose`, `type`,
  `referenceNode`, `referenceUserClass`, optional `nodeType` (`station`|`region`), `serverType`.
  System-level measures use `nodeType="" referenceNode=""`.
- `FixtureIntegrationTest.qopt3StationRunsAndMeasuresEveryStation` never asserts anything about
  the `fj` station's value — which is why the fork-join gap went unnoticed. Its
  `noneMatch(station.contains("__"))` assertion passes *by construction*, since `MeasureMapper`
  only ever iterates domain `model.nodes()`.

## Interfaces settled for the plan (write these verbatim)

```python
# qopt/exceptions.py  — add
TopologyError(QOptError)
SimulationError(QOptError)
  SimulationTransportError / SimulationRequestError / SimulationEngineError
  SimulationQualityError / MeasureMissingError

# qopt/station.py  — modify
Station.__init__(self, gamma=None, mu=None, weight=1.0, *, name=None)   # mu required via guard
Station.gamma                      -> property; raises ValueError if unset
Station.bind_gamma(value)          -> idempotent; raises if gamma was explicit, or on conflict
Station.zeta_from(T, S)            -> T * (S * self.mu - self.gamma)
Station.zeta(S)                    -> self.zeta_from(self.sojourn_time(S), S)
Station.sim_measure_type           -> abstract property
Station.sim_node(S, job_class)     -> abstract; returns a qsim node dict
  SingleServerStation.sim_measure_type == "response-time"
  ForkJoinStation.sim_measure_type    -> raises NotImplementedError naming qsim-service#5
  ForkJoinStation.sim_node            -> works now (branches at S*mu and S*r*mu, join="all")

# qopt/traffic.py  — new
solve_traffic(nodes, edges, arrival_rate, source, sink, *, tol=1e-12, max_iter=10_000)
    -> (dict[str, float], iterations);  raises TopologyError if the cap is hit

# qopt/network.py  — new
Route(src, dst, probability=1.0)                      # frozen dataclass
Network(stations, routes, arrival_rate, *, name="qopt-network",
        arrival_scv=1.0, job_class="jobs")
Network.SOURCE / Network.SINK                          # reserved endpoint sentinels
Network.to_model_dict(S) -> dict                       # emitted node names: "src", "snk"
Network.to_dot() -> str
# NO from_model_dict — S is not recoverable from the emitted S*mu product

# qopt/analyzer.py  — new
Evaluation(sojourn_times, ci=None, degraded=[], extras={})     # dataclass
Analyzer.is_stochastic : bool
Analyzer.evaluate(stations, S, *, fresh_seed=False) -> Evaluation
AnalyticAnalyzer  # is_stochastic = False; ci = None

# qopt/qsim/client.py  — new
QsimClient(base_url, *, timeout=None, stopping=None, transport=None, preflight=False)
    transport(url, body: bytes, timeout: float) -> (status: int, body: bytes)
    # guard at construction: timeout > stopping["maxWallClockSeconds"] + margin
QsimClient.post_simulate(request: dict) -> dict     # maps 400/422/500 to the exception tree

# qopt/qsim/spec.py  — new
build_request(network, S, *, seed, stopping, measures) -> dict

# qopt/qsim/measures.py  — new
extract_sojourn(response, stations, job_class) -> (T, ci, degraded, extras)
    # MeasureMissingError when a needed (station, class, type) triple is absent

# qopt/qsim/analyzer.py  — new
SimulationAnalyzer(network, client, *, seed=20260729, seed_policy="fixed", strict=False)
    # is_stochastic = True; pre-checks S*mu > gamma BEFORE issuing the POST

# qopt/allocator.py  — add
noise_floor(stations, C, zeta_vec, dzeta) -> float
    # ANTI-CORRELATED perturbation: component i up, all others down, plus mirror.
    # Uniform scaling of zeta is a no-op in eq 21, so uniform perturbation is NOT worst case.
    # Clamp perturbed zeta to a small positive value before sqrt.

# qopt/optimizer.py  — modify
Optimizer(stations, budget, *, analyzer=None, tol=1e-9, max_iter=None, initial_zeta=None,
          damping=None, noise_kappa=1.0, final_evaluation=True, strict=False, warm_start=True)
    # stations: Station sequence OR Network
    # defaults by analyzer kind: damping 1.0 / max_iter 1000 analytic;
    #                            damping 0.5 / max_iter 20 stochastic
    # RuntimeWarning when seed_policy=="fixed" and final_evaluation is False
Result  # + sojourn_ci, noise_floor, stop_reason, warm_start_iterations,
        #   degraded, system_response_time, sim_calls   (all defaulted)
```

Conventions: `job_class = "jobs"`; emitted source/sink node names `"src"` / `"snk"`; station
names must be non-empty, unique, free of `__`, and not `src`/`snk`.

Golden fixture: the §4.1.1 topology at `S = (3.0, 4.0, 5.0)` ⇒ `mm1` exponential rate `3.0`,
`md1` deterministic `0.25`, `fj` branches exponential `5.0` and `10.0`. Station names must be
renamed to routing-safe identifiers (the current `"ingest (M/M/1)"` has spaces and parens), so
the regression test compares **numbers**, not labels.

## Planned task decomposition (11 tasks)

Follow the prior plan's format: `## Task N:` at H2, real code in every step, TDD order
(failing test → run → implement → run → commit). See
`docs/superpowers/plans/2026-07-11-optimizer-implementation.md` as the style reference.

1. Exceptions + `Station.zeta_from` + optional `γ` / `bind_gamma`
2. `qopt/traffic.py` — `solve_traffic`
3. `qopt/network.py` — `Route`, `Network`, §4.2 validation, γ binding, §4.1.1 regression test
4. `Station.sim_node` / `sim_measure_type` + `Network.to_model_dict` + golden fixture + `to_dot`
5. `qopt/analyzer.py` — `Analyzer`, `Evaluation`, `AnalyticAnalyzer`
6. `qopt/qsim/client.py` — transport, POST, error mapping, health preflight, timeout coherence
7. `qopt/qsim/spec.py` + `qopt/qsim/measures.py` — request envelope, response → `E[T]`/CI/degraded
8. `qopt/qsim/analyzer.py` — `SimulationAnalyzer`, stability pre-check, seed policy
9. `allocator.noise_floor` + `Optimizer` loop knobs + extended `Result` + naive-equivalence test
10. `examples/simulated_tandem.py` + README + `__init__` exports + gated integration test
11. **Gated on qsim-service#5** — fork-join measure reading + `examples/simulated_mixed_network.py`

Tasks 1–10 ship a working simulated path for single-server networks. Task 11 needs the sister
repo fixed first.

## Open items

- **qsim-service#5 must land** before Task 11. Its fix is small (wire
  `fork-join-response-time` → `"Fork Join Response Time"` for fork-join nodes), but the
  `referenceNode`/`nodeType` pairing has to be settled empirically on the qsim side. That work
  needs its own session in that repo.
- **Unverified inference:** the `fj` Fork node's response time is *expected* to be ≈ 0 (no server
  in that node's sections), which is what would make the current silent answer catastrophic
  rather than merely wrong. Confirmable by running the `qopt-3station.json` fixture through the
  Java suite; not yet done.
- **PR #3 body** describes the first two commits; commit `ecc1380` (fork-join dependency) landed
  afterwards and is summarized in a PR comment rather than folded into the body.
