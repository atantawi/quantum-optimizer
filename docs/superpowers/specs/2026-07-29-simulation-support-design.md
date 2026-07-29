# Design Spec: Simulation Support — network topology and a `qsim-service`-backed analyzer

Date: 2026-07-29
Status: Approved design — pending final review before implementation planning.

Companion specs:

- `docs/superpowers/specs/2026-07-10-optimizer-design.md` — the analytic optimizer this
  extends. Its eq 21 / eq 22 / fixed-point loop are unchanged here.
- `docs/superpowers/specs/2026-07-25-qsim-service-design.md` — pointer to the `qsim-service`
  repo, which holds the authoritative JSON contract this spec consumes.

---

## 1. Goal

Let the optimizer obtain `E[T_i]` from a **discrete-event simulation of the whole network**
instead of from independent per-station closed-form approximations, by driving the
`qsim-service` HTTP/JSON API from inside the existing fixed-point loop.

This closes the gap the current README names as future work: per-station analysis cannot
capture how one station's *departure* process shapes the *arrival* variability of the
stations downstream of it. Simulating the network captures that coupling directly.

Two things this requires that the analytic path does not:

1. **A topology.** Closed-form analysis needs only each station's own `γ` and coefficients
   of variation. A simulation needs to know what is connected to what.
2. **Batch evaluation.** A simulation answers for every station in one run. Evaluation must
   become vector-in / vector-out — one POST per optimizer iteration, not one per station.

### 1.1 What does not change

The mathematics is untouched. Eq 21 (`allocate`), eq 22 (`ζ = E[T]·(Sµ − γ)`), the
objective `Σ ωᵢ E[Tᵢ]`, and the fixed-point loop structure are all identical. Only the
*source* of `E[T]` differs. This is deliberate: it makes analytic-vs-simulated results
directly comparable, because both run through the same allocator and the same loop.

### 1.2 Stated assumptions

- **Single open chain.** The network carries one customer chain that enters from a source
  and departs to a sink. Consequently every `γᵢ` is exogenously determined and **fixed
  across iterations**.

  Read "chain" in the queueing-theory sense — *open* as opposed to *closed*. It is **not** a
  restriction to a linear series of stations. Arbitrary probabilistic routing is supported,
  including branching and feedback cycles; the traffic equations (§4) handle all of them.

  This assumption is load-bearing, not cosmetic. Eq 21's base term `γᵢ/µᵢ` and
  `min_feasible_budget` both assume `γ` is constant while `S` varies. For a **closed** chain
  that fails: population is fixed, so λ derives from throughput, which depends on `S` — `γ`
  would shift every iteration and the budget-feasibility floor would move underneath the
  optimizer. Closed chains and multi-class networks are therefore out of scope (§10).

- **Fixed topology.** Routing and exogenous arrival rate are inputs. The optimizer never
  modifies them; it modifies only `S`, which enters the simulation as per-station service
  rates.

- **`cov_a` is analytic-only.** Under simulation, arrival variability at every station is
  *produced by the network*, so no station's `cov_a` is consulted at all. Exogenous arrivals
  are owned by the `Network` (`arrival_rate` + `arrival_scv`, §3), which is the single source
  for the source node's inter-arrival distribution. `GG1Station.cov_a` is retained because the
  analytic path and the warm start still need it. That the simulated path ignores it is the
  point of simulating, not a limitation.

## 2. Architecture

```
qopt/network.py          Network: stations + routing + exogenous rate.
                         Solves traffic equations → γ per station. Validates structure.
qopt/analyzer.py         Analyzer ABC + Evaluation record + AnalyticAnalyzer.
qopt/qsim/spec.py        Network + S → request dict (seed / stopping / measures envelope).
qopt/qsim/client.py      Transport (stdlib urllib default, injectable), POST, HTTP → exception.
qopt/qsim/measures.py    Response → per-station (E[T], CI, quality flags).
qopt/qsim/analyzer.py    SimulationAnalyzer = spec + client + measures.
qopt/optimizer.py        + warm start, damping, CI-aware stopping, extended Result.
qopt/station.py          + zeta_from(T, S), sim_node(S), sim_measure_type; γ now optional.
qopt/exceptions.py       + TopologyError, SimulationError hierarchy.
```

Data flow, one optimizer iteration:

```
S ──► Network.to_model_dict(S) ──► build_request ──► POST /simulate ──► [E[T]ᵢ, CIᵢ]
                                                                            │
        allocate(ζ) ◄── [ζᵢ = station.zeta_from(E[T]ᵢ, Sᵢ)] ◄───────────────┘
```

### 2.1 Why an `Analyzer` seam

`Optimizer.run()` currently pulls per station: `[st.zeta(Si) for st, Si in zip(...)]`. A
simulator cannot answer that way. Three options were considered:

| Option | Verdict |
|---|---|
| **`Analyzer` protocol** — network-level `evaluate(S) → E[T]`; analytic and simulated implementations behind one interface | **Chosen.** One stated seam; both paths share the loop verbatim, so comparisons are apples-to-apples; every piece independently testable |
| Per-station seam with a shared memo — `sojourn_time(S)` consults a cache keyed on the whole `S` vector, first miss triggers the POST | Rejected. A station's method silently depends on sibling state; the one-POST-per-iteration property is emergent rather than stated; float-vector cache keys are fragile |
| A parallel `SimOptimizer` class | Rejected. Duplicates convergence / damping / warm-start logic, which then drifts; warm-starting from the analytic solution requires reaching into the other class |

## 3. Topology representation

**Chosen format: mirror `qsim-service`'s `model` block.** There is no standard
queueing-network interchange format. The Python graph ecosystem offers NetworkX (node-link
JSON, GraphML, DOT), but those describe *graphs* — they carry no service distributions,
classes, or capacities — and NetworkX would add a runtime dependency to a currently
zero-dependency Apache-2.0 package while contributing nothing to the traffic solve. Reusing
qsim's own vocabulary means one format across both repos and makes `spec.py` a
transliteration rather than a translation layer.

```python
Network(
    stations,                    # list[Station]; .name required and unique
    routes,                      # list[Route(src, dst, probability=1.0)]
    arrival_rate,                # exogenous λ₀
    *, name="qopt-network", arrival_scv=1.0,
)
```

- `Route` uses `src` / `dst` in Python (`from` is a reserved word) and serializes to qsim's
  `{"from", "to", "probability"}`.
- `Network.SOURCE` and `Network.SINK` are reserved endpoint names, emitted as qsim
  `source` / `sink` nodes. The source node's inter-arrival distribution is built from
  `arrival_rate` and `arrival_scv` alone (`scv == 1` ⇒ exponential ⇒ Poisson arrivals),
  never from a station's `cov_a`.
- A **fork-join station is a single node** in the routing graph: λ in equals λ out. Both
  branches internally see that λ; the branch service rates live in the node fragment (§5.2).

**Serialization.** `Network.to_model_dict(S)` emits exactly qsim's `model` block — topology
plus per-station service at capacity `S`. `Network.from_model_dict` gives the round trip.
`qsim/spec.py` wraps the model with the `seed` / `stopping` / `measures` envelope, so
`Network` owns the *model* vocabulary and knows nothing of the *request* envelope.
`Network.to_dot()` emits Graphviz DOT for diagrams — a small string emitter, no dependency.

## 4. Traffic equations and the derivation of γ

```
λ = λ_ext + Pᵀλ        i.e.   λᵢ = λ₀·p_{SOURCE,i} + Σ_j λ_j·p_{ji}
```

Solved by fixed-point iteration from `λ = 0`, which converges geometrically for any open
chain. Stop at `max|Δλ| < 1e-12` with an iteration cap; exceeding the cap means flow is
trapped in a closed subnetwork, which is a `TopologyError`, not a warning. Roughly twenty
lines; no numpy, no dependency.

### 4.1 How γ reaches the stations

`Network.__init__` runs the solve and assigns each station its derived `γ`. Every existing
consumer — `allocate`, `min_feasible_budget`, `Station.zeta` — then works **unchanged**,
since all of them read `st.gamma`. One write point, so the emitted JSON and eq 21 cannot
disagree about arrival rates.

`Station.__init__`'s `gamma` therefore becomes optional:

| Case | Behavior |
|---|---|
| `gamma` passed, no network | Today's standalone analytic station. Existing tests and `examples/mixed_network.py` keep working unchanged |
| `gamma` omitted, bound to a `Network` | Filled from the traffic solve |
| `gamma` passed **and** bound to a `Network` | `ValueError`. γ is derived-only; there is no silent override path |
| `gamma` omitted, never bound | `ValueError` on first use — never a `None` propagating into the math |

This also resolves an inconsistency worth recording: `examples/mixed_network.py` uses
`γ = 0.6, 0.4, 0.5` for three stations, while `qsim-service`'s `qopt-3station.json` fixture
wires those same three stations as a tandem chain fed by one source at rate 1.0 — in which
every station sees λ = 1.0. Both cannot describe the same network. Under this design the
example remains a *standalone analytic* network (no topology, so no contradiction), and the
chain becomes a new example (§9).

### 4.2 Structural validation at construction

| Check | Rationale |
|---|---|
| Station names non-empty and unique | names are the routing keys |
| Every route endpoint resolves to a station or `SOURCE`/`SINK` | catches typos locally instead of as a qsim 422 |
| `SOURCE` has no in-edges; `SINK` has no out-edges | well-formed open chain |
| Per-node out-edge probabilities sum to 1 (single edge defaults to 1.0) | matches qsim contract invariant §5.3 |
| Every station reachable from `SOURCE` | an unreachable station gets λ = 0 and is trivially "optimal" |
| `SINK` reachable from every station | flow black holes break conservation |
| `arrival_rate` finite and > 0 | same guard style as existing `Station` fields |

Failures raise `TopologyError`; scalar parameter guards stay `ValueError`, matching current
conventions.

## 5. Station changes

### 5.1 `zeta_from(T, S)`

`ζᵢ = E[Tᵢ]·(Sᵢµᵢ − γᵢ)` is pure station arithmetic, independent of where `E[T]` came from.
So the station keeps owning it and simply accepts an externally supplied `E[T]`:

```python
def zeta_from(self, T, S):        # new: accepts an externally supplied E[T]
    ...

def zeta(self, S):                # unchanged public behavior
    return self.zeta_from(self.sojourn_time(S), S)
```

`sojourn_time` remains each station's analytic implementation, used by the analytic path,
by the warm start, and by every existing test.

### 5.2 Stations own their JSON node fragment

`Station.sim_node(S)` returns that station's qsim node dict, and `Station.sim_measure_type`
names the measure to read back. This follows the existing design philosophy — a station
already owns `alloc_cost`, `default_zeta`, and its queueing math — so adding a station type
stays a one-file change instead of growing an `isinstance` ladder in `spec.py`. The accepted
cost is that `station.py` knows the qsim schema shape. (The alternative, a type-keyed
registry in `spec.py`, keeps `station.py` schema-free at the price of splitting each station
type across two files.)

**Service emission under capacity `S`.** Station `i` emits a service distribution with mean
`1/(Sᵢ·µᵢ)`:

| Condition | Emitted form |
|---|---|
| `cov_s == 1` | `{"type": "exponential", "rate": S·µ}` |
| `cov_s == 0` | `{"type": "deterministic", "value": 1/(S·µ)}` |
| otherwise | `{"mean": 1/(S·µ), "scv": cov_s²}` — qsim maps to Gamma, exact on both moments |

`ForkJoinStation` emits two branches at `S·µ` and `S·r·µ` with `"join": "all"`, mirroring its
existing shared-capacity semantics (both servers receive capacity `S`, preserving the ratio
`r` for all `S`).

**Measure mapping.** Eq 22 needs *per-visit* sojourn time against the *total* arrival rate
`γᵢ` — which is the convention that makes `ζ = 1` come out exactly for M/M/1. So:

| Station type | qsim measure |
|---|---|
| single-server queue | per-station `response-time` |
| fork-join | `fork-join-response-time` |

`system-response-time` is recorded as a diagnostic on `Result`, **not** optimized — the
objective stays `Σ ωᵢ E[Tᵢ]` for continuity with the analytic path.

## 6. The optimizer loop

### 6.1 Analyzer interface

```python
class Analyzer(ABC):
    is_stochastic: bool                      # drives warm-start and damping defaults
    def evaluate(self, stations, S) -> Evaluation

@dataclass
class Evaluation:
    sojourn_times: list      # E[T] per station, aligned to station order
    ci: list | None          # (lower, upper) per station; None when analytic
    degraded: list           # audit strings: which measures came back weak
    extras: dict             # system-response-time, wallClockSeconds, seed
```

`AnalyticAnalyzer` (`is_stochastic = False`) delegates to `station.sojourn_time(S_i)` and
returns `ci = None`.

### 6.2 Construction

`Optimizer(stations, budget)` keeps working exactly as today — it defaults to
`AnalyticAnalyzer` with `damping = 1.0`, giving bit-identical behavior and leaving existing
tests untouched. The new form is `Optimizer(network, budget, analyzer=SimulationAnalyzer(...))`.

### 6.3 Loop

1. **Warm start.** If the analyzer is stochastic, first run the analytic loop to convergence
   — deterministic, roughly six iterations, zero simulation calls — and take its `S*` as
   `S₀`. Free, and it starts the expensive phase near the answer.
2. Each iteration: `E[T] ← analyzer.evaluate(S)`; `ζᵢ ← station.zeta_from(E[T]ᵢ, Sᵢ)`;
   `S_target ← allocate(stations, budget, ζ)`; then damp:
   `S ← (1−θ)·S + θ·S_target`.
3. Stop when `‖ΔS‖∞ < max(tol, κ·noise_floor)`.
4. If `final_evaluation`, evaluate once more at the converged `S*` with a fresh seed (§6.5);
   those numbers become the reported metrics.

**Cost model.** The warm start costs zero simulation calls (it is purely analytic), each loop
iteration costs exactly one POST, and the final evaluation costs one more. So
`sim_calls = iterations + (1 if final_evaluation else 0)`, and `warm_start_iterations` is
counted separately precisely because it is free.

### 6.4 Estimating the noise floor

`allocate` is closed-form and pure, so estimating how much of `ΔS` is attributable to
simulation noise costs **zero** extra simulation calls. Propagate each reported CI
half-width `hᵢ` into `ζ`:

```
δζᵢ = hᵢ · (Sᵢµᵢ − γᵢ)
```

then re-run `allocate` on perturbed `ζ` and measure the resulting spread in `S`.

**The perturbation direction matters.** Eq 21 is **invariant under uniform positive scaling
of ζ**: scaling every `ζ` by `k` multiplies both `numᵢ = √(ωᵢζᵢ/(cᵢµᵢ))` and
`denom = Σ√(ω_jζ_jc_j/µ_j)` by `√k`, which cancels in the ratio. So perturbing all stations
upward together is nearly a no-op rather than a worst case. The worst case is
**anti-correlated**. For each station `i`, evaluate `allocate` with component `i` at
`ζᵢ+δζᵢ` and all others at `ζ_j−δζ_j`, plus the mirror:

```
noise_floor = maxᵢ |Sᵢ(ζ⁺) − Sᵢ(ζ⁻)| / 2
```

That is `2n+1` closed-form evaluations — negligible against one simulation run. (The same
invariance is why an all-M/M/1 network converges in a single step: every `ζ` is identically
1, and uniform values are a fixed point of the scaling-invariant map.)

### 6.5 Seed policy

Two defensible choices, with a real trade-off:

- **Varying seed per iteration** — unbiased, but `ΔS` mixes genuine movement with seed
  noise, so the residual is noise-dominated and convergence detection is weak.
- **Fixed seed across iterations** (common random numbers) — `ΔS` reflects only the actual
  change in `S`, so the residual becomes signal-dominated and the loop converges crisply.
  But it converges to the fixed point *of that one sample path*, which is biased.

**Chosen: both, in sequence.** Use common random numbers *during* iteration for stable
convergence, then run **one final evaluation at `S*` with a fresh seed** to produce the
reported `E[T]`, CIs, and objective. `Result` records the loop's converged `S*` alongside
independently-seeded metrics at that point.

### 6.6 Knobs, and reproducing the naive loop

Every mechanism above has an off switch.

| Knob | Default (simulated) | Off means |
|---|---|---|
| `warm_start` | `True` | skip the analytic pre-solve; start from `initial_zeta` as today |
| `damping` (θ) | `0.5` | `1.0` = undamped, full jump to `allocate(ζ)` |
| `noise_kappa` (κ) | `1.0` | `0.0` = ignore the noise floor, plain `tol` only |
| `seed_policy` | `"fixed"` | `"vary"` = seed + iteration; `None` = omit `seed`, let qsim choose |
| `final_evaluation` | `True` | report the last loop iterate's numbers, no fresh-seed re-run |
| `strict` | `False` | `True` = raise on degraded results |
| `max_iter` | `20` | each iteration is a full simulation run, so not `1000` |

With `warm_start=False, damping=1.0, noise_kappa=0.0, seed_policy="vary",
final_evaluation=False`, the loop **is** today's `Optimizer.run()` with a simulated `E[T]`
substituted and nothing else added. This is an explicit acceptance test (§8), not a claim.

Two documented caveats:

- `seed_policy="fixed"` together with `final_evaluation=False` reports metrics from the CRN
  sample path — biased numbers that look clean. This pairing emits a `RuntimeWarning`.
- With `noise_kappa=0.0` against a stochastic analyzer, `converged=False` /
  `stop_reason="max_iter"` is the expected normal outcome, not a malfunction.

### 6.7 Extended `Result`

All new fields are defaulted, so existing construction sites are unaffected.

| Field | Meaning |
|---|---|
| `sojourn_ci` | per-station `(lower, upper)`; `None` on the analytic path |
| `noise_floor` | final `ΔS` attributable to simulation noise; `None` analytically |
| `stop_reason` | `"tol"` \| `"noise-floor"` \| `"max_iter"` |
| `warm_start_iterations` | analytic iterations consumed before the simulated phase |
| `degraded` | per-iteration audit of weak measures |
| `system_response_time` | qsim diagnostic; not optimized |
| `sim_calls` | POSTs issued — the real cost meter |

The existing `converged` field is retained for backward compatibility;
`stop_reason` is the finer-grained signal.

## 7. Error handling

### 7.1 Exception hierarchy

Extends the existing `QOptError` root:

```
QOptError
├── TopologyError                   structural Network failures (§4.2)
└── SimulationError
    ├── SimulationTransportError    refused / timeout / DNS
    ├── SimulationRequestError      400 / 422 — our JSON was wrong: a spec.py bug
    │                               or an invalid network
    ├── SimulationEngineError       500
    ├── SimulationQualityError      degraded result; raised only under strict=True
    └── MeasureMissingError         response lacked a measure a station needs
```

`MeasureMissingError` is a hard error rather than a warning: there is no number to proceed
with, so warn-and-proceed does not apply.

### 7.2 Degraded results

qsim may return `completed: false` (a cap fired before all CIs converged) or a per-measure
`success: false` (that measure's CI target was missed). Default policy is **warn and
proceed**: use the reported mean, emit a `RuntimeWarning` naming the station and measure,
and record the degradation per-iteration in `Result.degraded` so a caller can audit which
iterates rested on weak numbers. This matches how the current `Optimizer` handles
non-convergence. `strict=True` converts it to `SimulationQualityError`.

### 7.3 Fail-fast guards

- **Stability pre-check before POSTing.** If any `Sᵢ·µᵢ ≤ γᵢ` that station saturates and the
  entire run is garbage. Raise `InstabilityError` *before* issuing the request rather than
  spending minutes of simulation to discover it.
- **Timeout coherence at construction.** The client timeout must exceed
  `stopping.maxWallClockSeconds` plus a margin, or the client kills runs the server would
  have completed. Validated when `SimulationAnalyzer` is built, not on first use.
- **Optional `/health` preflight.** One GET at analyzer construction, so a misconfigured URL
  fails immediately with a clear message instead of on iteration 1.

### 7.4 Transport and dependencies

`qopt` stays at `dependencies = []`. The default transport is stdlib `urllib.request`; a
`transport` callable is injectable so tests use a fake and users can supply `httpx`, retry
policies, or authentication. The GPL/Apache boundary is preserved exactly as the
`qsim-service` spec intends: `qopt` speaks HTTP/JSON and never links JMT code.

## 8. Testing

The entire simulation path is unit-testable with no Java, no network, and no container.

| Test | What it pins |
|---|---|
| Fake transport returning canned JSON | spec build, measure mapping, degraded handling, every error branch |
| Golden request fixture | a **qopt-authored** `tests/fixtures/qopt_chain_request.json`; `Network.to_model_dict(S)` at a known `S` must reproduce it byte-for-byte. Locks our emission against accidental drift (see §8.1 on why qsim's own fixture is not the golden file) |
| Traffic equations | tandem (λ equal throughout), branch (λ splits by probability), feedback loop (λ = λ₀/(1−p)) — all closed-form expected values |
| ζ-scaling invariance | asserts `allocate` really is invariant to uniform ζ scaling — the property §6.4 depends on |
| Noise floor | synthetic analyzer with dialed CI widths; `stop_reason` flips `"tol"` → `"noise-floor"` as widths grow; κ=0 restores naive behavior |
| Naive-equivalence | all knobs off plus a deterministic fake analyzer mirroring `sojourn_time` ⇒ bit-identical to `Optimizer(stations, budget)` |
| Topology validation | one test per §4.2 row |
| M/M/1 bracket *(integration)* | single-station network against the live service: the simulated CI must bracket the analytic `1/(Sµ − γ)`. Gated on `QOPT_QSIM_URL`, skipped by default |

The M/M/1 bracket test is the actual validation of the idea; everything above it is
plumbing correctness.

### 8.1 Why qsim's `qopt-3station.json` is not the golden file

That fixture is a hand-authored *translation-layer* test on the qsim side, and it
deliberately exercises input forms qopt does not emit. Its fork-join branches use moment form
(`{"mean": 0.2, "scv": 1.5}` and `{"mean": 0.1, "scv": 0.5}`), but `ForkJoinStation` has no
per-branch `cov_s` and its `t_ul` approximation is built from `1/(µ − λ)` terms — i.e. it
assumes **exponential** servers. qopt can therefore only emit exponential fork-join branches,
and that fixture is not reproducible from any `Network`.

Its single-server nodes *are* consistent with §5.2 (`exponential rate 3.0` ⇒ `S·µ = 3`;
`deterministic 0.25` ⇒ `S·µ = 4`), and its tandem-at-λ=1.0 topology is what §4.1 refers to.
So it remains a useful cross-repo reference for the shared vocabulary — just not a
byte-comparison target. Per-branch `cov_s` on `ForkJoinStation` would require a fork-join
approximation that admits non-exponential servers, which is a modeling change, not plumbing
(§10).

## 9. Documentation deliverables

- `examples/simulated_chain.py` — the three-station chain solved analytically and by
  simulation, side by side, so the variability-propagation difference is visible.
- README: replace the dashed `future: simulation analyzer` box in the architecture diagram
  with the real path, and update **Scope & limitations** — network coupling moves from
  "future work" to "supported via simulation", with closed/multi-class remaining the honest
  open limitations.
- `examples/mixed_network.py` stays as-is: the standalone analytic example.

## 10. Out of scope

| Deferred | Why |
|---|---|
| **Closed chains** | λ would depend on `S` via throughput, so `γ` shifts each iteration and eq 21's budget floor moves under the optimizer (§1.2). Needs a math extension, not plumbing |
| **Multi-class networks** | Eq 21 has no per-class notion — `γᵢ`, `µᵢ`, and `ωᵢ` are scalar per station. Needs aggregate λ and a class-mix-weighted service rate, i.e. a math change |
| **Delay (infinite-server) nodes** | Cheap on the qsim side, but a delay node has no capacity to allocate, so it sits outside the optimization as pass-through latency. Additive later |
| **Non-exponential fork-join branches** | `t_ul` is built from `1/(µ − λ)` terms and assumes exponential servers (§8.1). Per-branch `cov_s` would need a fork-join approximation admitting general service — a modeling change |
| **Independent replications per iteration** | qsim already targets a per-measure CI internally, so single-run CIs suffice for §6.4. `qsim-service` spec §9 documents the aggregation method when this is wanted |
| **Optimizing the topology** | Routing is a fixed input by assumption (§1.2) |
| **Parallel/concurrent simulation calls** | The loop is inherently sequential — iteration `k+1` needs `S` from `k` |

## 11. Acceptance criteria

1. `Optimizer(stations, budget)` behaves bit-identically to the current implementation; the
   existing test suite passes unmodified.
2. `Network` derives `γ` for tandem, branching, and feedback topologies, matching
   closed-form expected values.
3. `Network.to_model_dict(S)` reproduces the committed `tests/fixtures/qopt_chain_request.json`
   golden fixture byte-for-byte.
4. The naive-equivalence test (§8) passes: knobs off ⇒ today's loop with a substituted
   `E[T]`.
5. Against a live `qsim-service`, a single-station M/M/1 network's simulated CI brackets the
   analytic `1/(Sµ − γ)`.
6. `qopt` still declares zero runtime dependencies.
7. Every §4.2 validation row and every §7.1 exception branch has a test.
