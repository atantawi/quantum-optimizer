# Handoff — Simulation Support (session state, 2026-07-29, revised 2026-07-30)

> ## SUPERSEDED — the feature is implemented
>
> This note captured the resume point *before* implementation. It is kept because its
> cross-repo findings and locked decisions are still the best record of why the design is
> what it is, and because reconstructing them meant reading the GPL sister repo's Java
> source and decompiling the bundled JMT jar.
>
> **Everything below describing work as "not yet written" or "not started" is stale.** The
> plan was written and all 11 of its tasks were implemented on branch
> `feat/simulation-support`. For current state read, in order:
>
> 1. `docs/superpowers/plans/2026-07-29-simulation-support.md` — the plan as executed
> 2. `.superpowers/sdd/2026-07-29-simulation-support/progress.md` — per-task ledger, the
>    three human rulings, and the 12 triaged review findings
>
> **Three items below were open questions and are now answered:**
>
> - The `station: ""` key for system measures is **verified**, not inferred — a live run
>   returned `{"station": "", "type": "system-response-time"}` with nothing under `"system"`.
> - The fork-join identity holds **exactly**: `response-time == system-response-time ==
>   0.2884507654809945`, matching `qsim-service`'s own committed fixture digit-for-digit.
> - Spec §6.3's stopping rule needed a θ factor it did not have; the spec has been amended
>   and the implementation departs from the original formula deliberately.
>
> Fork-join throughput remains exempt from the γ-conservation check pending
> [qsim-service#8](https://github.com/atantawi/qsim-service/issues/8) — still accurate.

Resume point for the simulation-support work (as of 2026-07-29). The **spec is finished and
committed**; the **implementation plan is not yet written**. That was the next action then.

## Where things stand

| | |
|---|---|
| Branch | `docs/simulation-support-design` (clean, pushed) |
| Spec | `docs/superpowers/specs/2026-07-29-simulation-support-design.md` — complete, 5 commits |
| PR | [#3](https://github.com/atantawi/quantum-optimizer/pull/3) → `main`, docs-only, open |
| Tracking issue | [#2](https://github.com/atantawi/quantum-optimizer/issues/2) |
| External blockers | **None.** [qsim-service#7](https://github.com/atantawi/qsim-service/pull/7) merged (`51a99c7`), closing #5/#6. [qsim-service#8](https://github.com/atantawi/qsim-service/issues/8) is open but constrains diagnostics only (spec §10) |
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
| Fork-join blocker | **Resolved upstream** by qsim-service#7. FJ measure reading is still sequenced last in the plan, but as an ordering choice — it needs a live service and a non-trivial oracle — not a dependency |
| Requested measures | **Closed list of three**, always explicit: `response-time`, `system-response-time`, `throughput`. Omitting it lets qsim substitute `DEFAULTS`, two of which are join-station numbers on a FJ node (spec §5.4) |
| `throughput`'s purpose | γ-conservation check — an independent witness that `solve_traffic` and `to_model_dict` describe the same network. Warn + record; `strict=True` raises. FJ stations exempt per qsim-service#8 (spec §6.8) |
| FJ validation oracles | The `system-response-time` **identity** (exact, CI-independent) plus a **symmetric** `r=1` `t_ul` bracket. Heterogeneous `t_ul` bracketing is self-defeating — see spec §8.2 before re-adding it |

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

**Cross-repo findings (qsim-service), re-verified against `51a99c7` on 2026-07-30:**

- `MeasureMapper.SUPPORTED` = `response-time, residence-time, queue-time, queue-length,
  utilization, throughput, drop-rate, system-response-time`. **No `fork-join-response-time`** —
  requesting that literal is a **400**. (`SUPPORTED` folds in `FORK_JOIN_STATION.keySet()`, but
  that is a no-op: the FJ override's key is `response-time`.)
- `MeasureMapper.FORK_JOIN_STATION = {"response-time" → "Fork Join Response Time"}`, applied
  **only** when the node `instanceof ForkJoinNode`. So `response-time` means station response time
  at a queue and fork-to-join sojourn at a fork-join node — one domain type, two semantics.
- `MeasureMapper.map()` emits a measure for **every** node × served class, per requested type.
  qopt cannot request a type for just one station; `servedClasses` returns empty for source/sink,
  so those yield nothing.
- `MeasureMapper.DEFAULTS = [response-time, utilization, throughput, queue-length]` — used when
  `measures` is null or empty. **Two of those four are join-station numbers on a FJ node.** This is
  why spec §5.4 makes the list mandatory rather than relying on a default.
- `JsimgWriter.expandedMeasureNode` skips the `__join` remap for `MeasureMapper.FORK_JOIN_TYPES`,
  leaving them on the domain name (already the fork station). Other types still remap to the join.
- `SolutionsParser.REVERSE` maps `"Fork Join Response Time"` → `response-time`, and
  `domainStation()` strips `__join` / `__bN` suffixes, so no internal name reaches qopt.
- `JsimgWriter` now rejects a fork-join measure type on a non-fork-join node as **422**. qopt
  cannot trigger it through `/simulate` (only `MeasureMapper` builds the specs), but it is one more
  input to the `SimulationRequestError` branch.
- **JMT defines exactly two fork-join region measures**: `FORK_JOIN_RESPONSE_TIME` and
  `FORK_JOIN_NUMBER_OF_JOBS`. That ceiling is why qsim-service#8 can at best fix `queue-length`;
  the other five need invented semantics. Bounds how far spec §5.4's list can ever widen.
- `SIMmodeldefinition.xsd` `<measure>` attrs: `name`, `alpha`, `precision`, `verbose`, `type`,
  `referenceNode`, `referenceUserClass`, optional `nodeType` (`station`|`region`), `serverType`.
  System-level measures use `nodeType="" referenceNode=""`.
- **Not verified:** that a system measure comes back as `station: ""` in the JSON response. It
  follows from `referenceNode=""` plus `domainStation("")` passing through, but `src/test/resources/
  results/` has **no system-measure fixture** (only `mm1`, `fork-join`, `inverted-bounds`, none
  with a system measure). Spec §5.3 gotcha 2 flags this as an inference.
- Fixture `results/fork-join.solutions.xml` **does** pin `measureType="Fork Join Response Time"
  station="fj"` with `meanValue="0.2884507654809945"` — the qopt-side contract for the response
  shape.

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
Station.SIM_MEASURE_TYPE           = "response-time"   # base constant, NOT abstract
Station.sim_conservation_checked   = True              # class attr, overridable
Station.sim_node(S, job_class)     -> abstract; returns a qsim node dict
  ForkJoinStation.sim_conservation_checked = False     # qsim-service#8
  ForkJoinStation.sim_node   -> branches at S*mu and S*r*mu, join="all"
  # Post qsim-service#7 a FJ node's "response-time" IS the fork-to-join sojourn, so the
  # measure type no longer varies by station type. Do not reintroduce it as a property.

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
MEASURES = ("response-time", "system-response-time", "throughput")   # closed; never empty
build_request(network, S, *, seed, stopping, measures=MEASURES) -> dict

# qopt/qsim/measures.py  — new
extract(response, stations, job_class) -> (T, ci, degraded, extras)
    # extras["throughput"]: name -> (mean, (lo, hi));  extras["system_response_time"]
    # MeasureMissingError ONLY for a missing station "response-time" (eq 22 has no input).
    # Missing system-response-time / throughput -> None + RuntimeWarning + degraded entry.
    # System measures are keyed on station == "" (inferred, not yet pinned — spec §5.3).

# qopt/qsim/analyzer.py  — new
SimulationAnalyzer(network, client, *, seed=20260729, seed_policy="fixed", strict=False)
    # is_stochastic = True; pre-checks S*mu > gamma BEFORE issuing the POST
    # runs the γ-conservation check each evaluate(): simulated throughput CI must bracket
    # station.gamma for every station with sim_conservation_checked; miss -> RuntimeWarning
    # + degraded entry, or SimulationQualityError under strict=True

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
renamed to short identifiers by choice — §4.2 would have accepted `"ingest (M/M/1)"`, since it
rejects only empty, non-unique, `__`-containing, and `src`/`snk` names — so
the regression test compares **numbers**, not labels.

## Planned task decomposition (11 tasks)

Follow the prior plan's format: `## Task N:` at H2, real code in every step, TDD order
(failing test → run → implement → run → commit). See
`docs/superpowers/plans/2026-07-11-optimizer-implementation.md` as the style reference.

1. Exceptions + `Station.zeta_from` + optional `γ` / `bind_gamma`
2. `qopt/traffic.py` — `solve_traffic`
3. `qopt/network.py` — `Route`, `Network`, §4.2 validation, γ binding, §4.1.1 regression test
4. `Station.sim_node` + `SIM_MEASURE_TYPE` / `sim_conservation_checked` + `Network.to_model_dict`
   + golden fixture + `to_dot`
5. `qopt/analyzer.py` — `Analyzer`, `Evaluation`, `AnalyticAnalyzer`
6. `qopt/qsim/client.py` — transport, POST, error mapping, health preflight, timeout coherence
7. `qopt/qsim/spec.py` + `qopt/qsim/measures.py` — request envelope with the closed three-measure
   list (§5.4, test-pinned), response → `E[T]`/CI/throughput/degraded
8. `qopt/qsim/analyzer.py` — `SimulationAnalyzer`, stability pre-check, seed policy,
   γ-conservation check (§6.8)
9. `allocator.noise_floor` + `Optimizer` loop knobs + extended `Result` + naive-equivalence test
10. `examples/simulated_tandem.py` + README + `__init__` exports + gated integration tests
    (M/M/1 bracket, γ-conservation on the §4.1.1 network)
11. Fork-join measure reading + `examples/simulated_mixed_network.py` + the two FJ integration
    oracles (identity, symmetric `t_ul` bracket — §8.2)

Tasks 1–10 ship a working simulated path for single-server networks. **Task 11 is last by choice,
not by dependency** — it is the one path whose verification needs a live service and a non-trivial
oracle.

## Open items

- **`station: ""` for system measures is unverified** (spec §5.3 gotcha 2). No `qsim-service`
  fixture pins it and that repo's own spec §5.2 example says `"system"` instead. `measures.py`
  keys on `""`; Task 10's first live run settles it. If wrong, the symptom is
  `Result.system_response_time is None` plus a `RuntimeWarning` — and Task 11's identity test
  failing outright, since it uses that measure as its oracle.
- **Fork-join throughput is exempt from the γ-conservation check** pending
  [qsim-service#8](https://github.com/atantawi/qsim-service/issues/8). Under `join: "all"` it
  *ought* to equal λ and one probe measured `0.985` vs λ = 1.0, but that is one measurement, not an
  upstream guarantee, and it is unverified beyond two branches. When #8 lands, deleting
  `ForkJoinStation.sim_conservation_checked = False` is the whole change.
- **PR #3 body** describes the first two commits only. `ecc1380` (fork-join dependency) and
  `0b374c0` (§5.3 corrected to the as-built contract) are summarized in PR comments, and the
  2026-07-30 revision — closed measure list, γ-conservation check, oracle shapes — is not yet
  reflected there at all. Worth folding into the body before the PR is reviewed.
