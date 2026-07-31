# Design Spec: Simulation Support — network topology and a `qsim-service`-backed analyzer

Date: 2026-07-29 (revised 2026-07-30; amended post-implementation 2026-07-30)
Status: **Implemented.** The `qsim-service` measure contract in §5.2–§5.4 is **as-built** against
[qsim-service#7](https://github.com/atantawi/qsim-service/pull/7) (merged `51a99c7`), re-verified
against that merged Java source on 2026-07-30. Re-verify if `MeasureMapper`,
`JsimgWriter.expandedMeasureNode`, or `SolutionsParser.REVERSE` change.
Implementation plan: `docs/superpowers/plans/2026-07-29-simulation-support.md`.

**Post-implementation amendments.** Two things this spec asserted turned out to need correction
once the code ran against a live service. Both are marked inline below:

- **§6.3's stopping rule compared a damped step against target-space thresholds**, making both
  `tol` and `κ` off by `1/θ`. The implementation deliberately departs from the original formula:
  it normalizes the step to target space once. Amended twice — see the note in §6.3/§6.4.
- **§5.3 gotcha 2's `station: ""` inference is now verified**, not inferred.

Everything else was confirmed as written, including the §4.1.1 arithmetic, the closed measure
list, the γ-conservation check, and both fork-join oracle shapes in §8.2.

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
qopt/qsim/spec.py        Network + S → request dict (seed / stopping / closed measure list §5.4).
qopt/qsim/client.py      Transport (stdlib urllib default, injectable), POST, HTTP → exception.
qopt/qsim/measures.py    Response → per-station (E[T], CI, throughput, quality flags).
qopt/qsim/analyzer.py    SimulationAnalyzer = spec + client + measures + γ-conservation (§6.8).
qopt/optimizer.py        + warm start, damping, CI-aware stopping, extended Result.
qopt/station.py          + zeta_from(T, S), sim_node(S), SIM_MEASURE_TYPE,
                         sim_conservation_checked; γ now optional.
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
plus per-station service at capacity `S`. `qsim/spec.py` wraps the model with the `seed` /
`stopping` / `measures` envelope, so `Network` owns the *model* vocabulary and knows nothing of
the *request* envelope. `Network.to_dot()` emits Graphviz DOT for diagrams — a small string
emitter, no dependency.

**No `from_model_dict`.** A round trip is not well-defined: the emitted service rate is the
*product* `S·µ`, and `S` is not recoverable from it, so stations cannot be reconstructed from a
model dict. Deserialization would therefore have to be topology-only with caller-supplied
stations, which no consumer needs yet — deferred as YAGNI rather than shipped half-defined.

**Station naming rules** (enforced in §4.2). Names become JSON node names, routing keys, and DOT
identifiers, so they must be non-empty, unique, and must not contain `__` — qsim's `JsimgWriter`
mints internal fork-join station names as `<node>__b0` / `<node>__join`, and a domain name
containing `__` could collide with them. `src` and `snk` are reserved for the emitted source and
sink nodes.

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

### 4.1.1 The existing example is a realizable topology

`examples/mixed_network.py` uses `γ = 0.6, 0.4, 0.5`. Those are not arbitrary — they are
exactly the traffic-equation solution of a Poisson source at λ₀ = 1.0 feeding `mm1` and `md1`
in **parallel**, both then routing into the fork-join station:

```
source (λ₀ = 1.0)
   ├─ 0.6 ─► mm1 ──┬─ 0.5 ─► fj ─ 1.0 ─► sink
   └─ 0.4 ─► md1 ──┤
                   └─ 0.5 ─► sink
```

```
λ_mm1 = 1.0 · 0.6                = 0.6
λ_md1 = 1.0 · 0.4                = 0.4
λ_fj  = 0.6 · 0.5 + 0.4 · 0.5    = 0.5
```

Out-edge probabilities sum to 1 at every node, so this satisfies §4.2 as written. There is one
degree of freedom — any `(p_mm1, p_md1)` with `0.6·p_mm1 + 0.4·p_md1 = 0.5` reproduces the same
γ — and this spec pins the symmetric choice `p_mm1 = p_md1 = 0.5` so the golden fixture (§8) is
deterministic.

**Why this is the right example to build on.** `min_feasible_budget` for these stations is
`2.0(0.6) + 1.0(0.4) + 2.0(0.5) = 2.6`, and `6 × 2.6 = 15.6` — the exact budget in the current
README output. So the network form with *derived* γ reproduces today's analytic table
bit-for-bit, making it a **regression test** rather than a new example needing its own
baseline (§8, §11).

It also isolates the phenomenon this whole feature exists to capture, because γ is identical on
both paths — so any analytic-vs-simulated difference is attributable to variability propagation
alone, not to differing arrival rates:

- A Poisson source split by Bernoulli probabilities yields streams into `mm1` and `md1` that
  **are** Poisson. Their `cov_a = 1` is therefore exactly right, and analytic should closely
  match simulated at those two stations.
- `fj` receives a thinned **superposition of two departure streams**, which is not Poisson —
  yet `t_ul` is documented as taking a *Poisson* arrival rate. So `fj` is precisely where the
  analytic approximation is unjustified and where simulation is expected to diverge.

An expected-divergence-location prediction like that is worth far more as a demonstration than a
tandem chain, where every station is equally suspect.

Note that `qsim-service`'s `qopt-3station.json` fixture wires the same three stations as a
*tandem* chain at λ = 1.0. That is simply **a different network**, not a conflict: it is a
translation-layer test on the qsim side (§8.1), not a description of this example.

### 4.2 Structural validation at construction

| Check | Rationale |
|---|---|
| Station names non-empty and unique | names are the routing keys |
| Station names contain no `__`, and are not `src`/`snk` | avoids collision with qsim's internal `<node>__b0`/`<node>__join` names and our emitted source/sink node names |
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

`Station.sim_node(S)` returns that station's qsim node dict. This follows the existing design
philosophy — a station already owns `alloc_cost`, `default_zeta`, and its queueing math — so
adding a station type stays a one-file change instead of growing an `isinstance` ladder in
`spec.py`. The accepted cost is that `station.py` knows the qsim schema shape. (The
alternative, a type-keyed registry in `spec.py`, keeps `station.py` schema-free at the price
of splitting each station type across two files.)

Alongside `sim_node`, a station carries two **class-level** qsim facts:

```python
class Station(ABC):
    SIM_MEASURE_TYPE = "response-time"     # which measure supplies E[T] for eq 22
    sim_conservation_checked = True        # is simulated throughput a valid witness on γ?

class ForkJoinStation(Station):
    sim_conservation_checked = False       # qsim-service#8; delete the line when it lands
```

`SIM_MEASURE_TYPE` is deliberately **not** an abstract property. An earlier revision made it one
because fork-join stations were expected to need a distinct `fork-join-response-time` type; §5.3
records why that turned out not to be so. A hook every subclass implements identically is dead
abstraction, so it is a base-class constant until a station type genuinely varies it.

`sim_conservation_checked` is where the real per-station variation now lives (§6.8).

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

| Station type | qsim measure | Status |
|---|---|---|
| single-server queue | per-station `response-time` | available |
| fork-join | per-station `response-time` | available — qsim resolves it to the fork-to-join sojourn (§5.3) |

A fork-join station's sojourn time is the **fork-to-join** interval: the time from a job being
forked into the branches until the join completes. It is *not* the response time of any single
branch, nor of the fork node. `qsim-service` honors exactly that reading — `response-time` on a
`fork-join` node is *defined* as the fork-to-join sojourn (§5.3) — so both station types request the
same domain measure type and `SIM_MEASURE_TYPE` is uniformly `"response-time"`.

### 5.3 Fork-join measurement, as built

The as-built contract, verified against `qsim-service` at `51a99c7`:

| | |
|---|---|
| Domain type qopt requests for a fork-join station | `response-time` |
| Literal `fork-join-response-time` | **not a type** — `MeasureMapper.SUPPORTED` rejects it with **400**. Never request it |
| JMT measure qsim emits underneath | `"Fork Join Response Time"` (`SimConstants.FORK_JOIN_RESPONSE_TIME`), anchored on the **fork** station, because the fork's input section is what maintains the fork-to-join job list |
| `station` name in the response | the **domain** name (e.g. `fj`) — `SolutionsParser.REVERSE` maps `"Fork Join Response Time"` → `response-time`, so the no-internal-names contract holds |

`MeasureMapper.FORK_JOIN_STATION` overrides the station-level mapping for `ForkJoinNode`s only, so
`response-time` means "station response time" at a queue and "fork-to-join sojourn" at a fork-join
node. That is the semantics §5.2 needs, obtained without a distinct domain type — which is why
`SIM_MEASURE_TYPE` is a single constant for every station type this spec ships.

**History.** An earlier revision of this section specified a `fork-join-response-time` domain type
and declared the work blocked on
[qsim-service#5](https://github.com/atantawi/qsim-service/issues/5). Both were superseded by
[qsim-service#7](https://github.com/atantawi/qsim-service/pull/7), which closed #5 and its duplicate
#6. That PR and issue hold the forensics — what the pre-fix encoding measured, why the join is not a
valid anchor at any measure type, and the measured numbers. None of it is load-bearing here, and no
qopt result was ever affected: at the time `SimulationAnalyzer` did not yet exist, and the
analytic path uses `t_ul` rather than qsim.

**Two as-built gotchas qopt must respect.**

1. **Only `response-time` has fork-join-region semantics.** `residence-time`, `queue-time`,
   `queue-length`, `utilization`, `throughput`, and `drop-rate` on a fork-join node are still
   *join-station* numbers ([qsim-service#8](https://github.com/atantawi/qsim-service/issues/8)).
   `residence-time` deserves particular care: it is the closest-looking alternative to
   `response-time` and just as wrong. §5.4 fixes the requested list; §6.8 is why `throughput` is on
   it anyway; §10 records the upstream ceiling.
2. **System-level measures come back with `station: ""`** — **verified 2026-07-30**; this was an
   inference when the spec was written, and it held.
   `MeasureMapper` emits `referenceNode=""` for system measures and `SolutionsParser.domainStation`
   passes an empty name through unchanged, so `""` is what the response should carry. No
   `qsim-service` fixture pinned it (that repo has no system-measure solutions fixture) and the
   `qsim-service` spec §5.2 example comment says `station: "system"` instead, so it was flagged as
   an inference deliberately.

   **Settled by a live run on 2026-07-30.** A single-station M/M/1 network returned
   `{"station": "", "class": "jobs", "type": "system-response-time", "mean": 0.994325}`, with no
   measure keyed on `"system"`. `qopt`'s `SYSTEM_STATION = ""` is therefore correct as written, and
   `tests/test_integration_qsim.py::test_system_measure_key_inference_holds` is now a regression
   guard rather than a discovery.

**Consequence for delivery.** Nothing here is gated. `ForkJoinStation.sim_node` was never blocked
(emitting the node, its two heterogeneous branches, and the join is entirely qopt-side), and reading
the measure back works too. Simulated fork-join support is still sequenced **last** in the
implementation plan — it is the one path whose verification needs a live service and a non-trivial
oracle — but as an ordering choice, not an external dependency.

`system-response-time` is recorded as a diagnostic on `Result`, **not** optimized — the
objective stays `Σ ωᵢ E[Tᵢ]` for continuity with the analytic path. It is also the oracle for §8's
fork-join identity test.

### 5.4 The requested measure list is explicit and closed

`build_request` always sends exactly, and never omits or empties:

```python
MEASURES = ("response-time",         # E[T] for eq 22, every station type (§5.2)
            "system-response-time",   # Result diagnostic; §8 fork-join oracle
            "throughput")             # γ-conservation witness (§6.8)
```

**Why explicit matters.** `MeasureMapper` falls back to `DEFAULTS = ["response-time",
"utilization", "throughput", "queue-length"]` when the request's `measures` is null or empty. Two of
those four — `utilization` and `queue-length` — are join-station numbers on a fork-join node, and
they come back with `success: true` and no warning (§5.3). So omitting the list is not a harmless
default; it is a silent-wrong-answer path. A unit test asserts the exact tuple against the fake
transport.

**Why closed.** The list is not a user-facing knob. Nothing outside these three enters eq 21, eq 22,
the objective, or the fixed point, so there is no request qopt could honor by widening it — only
measures it would have to reason about the trustworthiness of. Widening waits on qsim-service#8
(§10).

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
    degraded: list           # audit strings: weak measures, γ-conservation misses (§6.8)
    extras: dict             # system-response-time, throughput, wallClockSeconds, seed
```

`extras["throughput"]` maps station name → `(mean, (lower, upper))`. It feeds §6.8 and is otherwise
inert; nothing in the loop consumes it.

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
3. Stop when `‖ΔS‖∞ / θ < max(tol, κ·noise_floor)` — that is, **test convergence in
   target space**, on the step `allocate` still wants to take, not on the damped step the
   iterate actually took.

   > **Amended 2026-07-30, superseded 2026-07-31 (both post-implementation).** This step
   > originally read `‖ΔS‖∞ < max(tol, κ·noise_floor)`, comparing a **damped** step against
   > two **target-space** quantities. `‖ΔS‖∞` is `θ·|S_target − S|`, while `noise_floor`
   > (§6.4) is the spread in `allocate`'s *output* and `tol` is a tolerance on how far
   > `allocate` still wants to move. So *both* terms were off by `1/θ`: at the §6.6 defaults
   > (`θ = 0.5`, `κ = 1.0`) the loop stopped at **2** noise widths while reporting `κ = 1`,
   > and `tol = 1e-9` behaved as `2e-9`.
   >
   > The first amendment scaled only the noise term (`κ·θ·noise_floor`), which fixed κ and
   > left `tol` wrong. The current form normalizes the step **once** instead, which repairs
   > both terms together and — the reason it matters — keeps `tol` meaning the same thing on
   > the analytic and simulated paths. Cross-path comparability is the premise §1.1 rests on.
   >
   > Both errors failed safe: they stopped early, returning a less-converged `S*`, and never
   > looped forever. That is why they survived design review and two implementation passes.
   > Division by `1.0` is exact in IEEE 754, so the normalization is bit-for-bit inert at
   > `θ = 1.0` and the analytic path is untouched. Note the κ arm's *behavior* is identical
   > under either amendment — `θd < κθf ⟺ d < κf` — so only the `tol` arm changed on
   > 2026-07-31.
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

Note that what this yields is a spread in `allocate`'s **output** — the movement noise can induce
in `S_target`, not in the damped iterate. §6.3's stopping rule must scale it by θ to compare
against a damped `‖ΔS‖∞`.

**The perturbation direction matters.** Eq 21 is **invariant under uniform positive scaling
of ζ**: scaling every `ζ` by `k` multiplies both `numᵢ = √(ωᵢζᵢ/(cᵢµᵢ))` and
`denom = Σ√(ω_jζ_jc_j/µ_j)` by `√k`, which cancels in the ratio. So perturbing all stations
upward together is nearly a no-op rather than a worst case. The worst case is
**anti-correlated**. For each station `i`, evaluate `allocate` with component `i` at
`ζᵢ+δζᵢ` and all others at `ζ_j−δζ_j`, plus the mirror:

```
noise_floor = maxᵢ |Sᵢ(ζ⁺) − Sᵢ(ζ⁻)| / 2
```

That is `2n` closed-form evaluations — negligible against one simulation run. (The spec first said `2n+1`, counting an unperturbed baseline the implementation does not need, since each station's pair is compared against itself.) (The same
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
| `degraded` | per-iteration audit of weak measures and γ-conservation misses (§6.8) |
| `system_response_time` | qsim diagnostic; not optimized |
| `sim_calls` | POSTs issued — the real cost meter |

The existing `converged` field is retained for backward compatibility;
`stop_reason` is the finer-grained signal.

### 6.8 The γ-conservation check

§4 makes `γ` **derived** — one write point in `Network.__init__`, then read by `allocate`,
`min_feasible_budget`, and `zeta_from`. If the traffic solve and the emitted routing ever disagree,
every number the optimizer produces is correct for a network that is not the one being simulated. §8's
closed-form topology tests cannot catch that: they check `solve_traffic` against analytic expectations,
not against what `to_model_dict` actually serialized.

Simulated throughput is an independent witness on exactly that, and it arrives on the same POST for
free. So `SimulationAnalyzer.evaluate` checks it every iteration:

```python
for st in stations:
    if not st.sim_conservation_checked:      # fork-join: qsim-service#8
        continue
    mean, (lo, hi) = throughput[st.name]
    if not (lo <= st.gamma <= hi):
        degraded.append(
            f"{st.name}: simulated throughput {mean:.6f} CI ({lo:.6f}, {hi:.6f}) "
            f"excludes derived γ={st.gamma:.6f}")
```

Policy is §7.2's, verbatim and unextended: a miss emits a `RuntimeWarning`, is recorded in
`Result.degraded`, and the run proceeds; `strict=True` raises `SimulationQualityError`. It is not a
hard failure, because a watchdog-truncated run (`completed: false`) can widen or bias throughput
enough to miss legitimately, and halting an otherwise usable optimization for that would be worse
than reporting it.

Fork-join stations are skipped because their throughput is the internal join station's number
(§5.3). Under `join: "all"` exactly one job leaves the join per job forked, so it *ought* to equal λ,
and one probe measured `0.985` against λ = 1.0 — but that is a single measurement rather than a
pinned upstream guarantee, and it is unverified for more than two branches. Asserting on it would
mean trusting a number qsim-service does not yet claim is correct.

## 7. Error handling

### 7.1 Exception hierarchy

Extends the existing `QOptError` root:

```
QOptError
├── TopologyError                   structural Network failures (§4.2)
└── SimulationError
    ├── SimulationTransportError    refused / timeout / DNS, a /health that was not
    │                               200, or a /simulate status in neither family below
    ├── SimulationRequestError      400 / 405 / 413 / 422 — our request was wrong, in
    │                               its body (a spec.py bug or an invalid network), its
    │                               method, or its size. Never the engine's fault
    ├── SimulationEngineError       500
    ├── SimulationQualityError      degraded result; raised only under strict=True
    └── MeasureMissingError         response lacked a measure a station needs
```

`MeasureMissingError` is a hard error rather than a warning: there is no number to proceed
with, so warn-and-proceed does not apply.

Its scope is narrow — a **station `E[T]`** that eq 22 needs, i.e. a missing `(station, class,
"response-time")` triple. The other two requested measures are diagnostics, and their absence must
not abort a run that has everything the mathematics requires:

| Missing | Consequence |
|---|---|
| `response-time` for any station | `MeasureMissingError`. Eq 22 has no input |
| `system-response-time` | `Result.system_response_time = None` + `RuntimeWarning`. Also the signal that §5.3's `station: ""` inference was wrong |
| `throughput` for a conservation-checked station | `RuntimeWarning` + a `degraded` entry saying the check could not run — distinct from the check running and failing (§6.8) |

A **present mean whose confidence interval is absent** is a separate axis, and a separate table:
qsim can return `mean` without `lower`/`upper`, and a mean is enough for everything the mathematics
requires. So no measure escalates to an error here — but none of them may pass the `None`s through
silently either, because the caller that formats a bound is the one that discovers it.

| Mean present, CI absent | Consequence |
|---|---|
| `response-time` for a station | `RuntimeWarning` + `degraded`. The mean still feeds eq 22; `Result.sojourn_ci[i]` is `None` and that station drops out of the noise-floor estimate (§6.4) |
| `system-response-time` | `RuntimeWarning` + `degraded`. Reported as `(mean, (None, None))` — the tuple shape is kept, matching `throughput`, so consumers that format bounds must handle the `None`s |
| `throughput` for a conservation-checked station | `RuntimeWarning` + `degraded`. The γ-conservation check cannot run — the same consequence as the measure being absent |

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
| Golden request fixture | a **qopt-authored** `tests/fixtures/qopt_mixed_network_request.json`, built from the §4.1.1 topology; `Network.to_model_dict(S)` at a known `S` must reproduce it byte-for-byte. Locks our emission against accidental drift (see §8.1 on why qsim's own fixture is not the golden file) |
| Traffic equations | tandem (λ equal throughout), branch (λ splits by probability), feedback loop (λ = λ₀/(1−p)) — all closed-form expected values |
| Mixed-network γ derivation | the §4.1.1 topology derives exactly `γ = (0.6, 0.4, 0.5)` |
| Mixed-network regression | the §4.1.1 `Network` on the analytic path reproduces the legacy standalone result bit-for-bit — same budget 15.6, same `S*`, `E[T]`, `ζ`, and objective. This is the strongest single check that deriving γ changed nothing |
| ζ-scaling invariance | asserts `allocate` really is invariant to uniform ζ scaling — the property §6.4 depends on |
| Noise floor | synthetic analyzer with dialed CI widths; `stop_reason` flips `"tol"` → `"noise-floor"` as widths grow; κ=0 restores naive behavior |
| Naive-equivalence | all knobs off plus a deterministic fake analyzer mirroring `sojourn_time` ⇒ bit-identical to `Optimizer(stations, budget)` |
| Topology validation | one test per §4.2 row |
| Explicit measure list | `build_request` emits exactly `("response-time", "system-response-time", "throughput")` — never null, never empty. Pins §5.4, whose failure mode is qsim's `DEFAULTS` silently substituting two join-station measures |
| γ-conservation, unit | fake transport returning throughput that excludes a station's `γ` ⇒ `RuntimeWarning` + a `degraded` entry; `strict=True` ⇒ `SimulationQualityError`; a fork-join station is never flagged whatever its throughput (§6.8) |
| γ-conservation *(integration)* | the §4.1.1 branching network against the live service: simulated throughput at `mm1` and `md1` must bracket the derived `0.6` and `0.4`. This is the end-to-end check that `solve_traffic` and `to_model_dict` agree about the same network — the branching topology is what makes it meaningful, since a tandem chain would pass even with the source split wrong |
| M/M/1 bracket *(integration)* | single-station network against the live service: the simulated CI must bracket the analytic `1/(Sµ − γ)`. Gated on `QOPT_QSIM_URL`, skipped by default |
| Fork-join identity *(integration)* | `src → fj → snk`, where the fork-join is the only service: `response-time` at `fj` must equal `system-response-time` to `1e-9`. Both come from the same sample path, so this is an **identity, not a statistical bracket** — it holds at any precision target and needs no analytic model. The sharpest available guard against a re-regression to join-anchoring, which read `0.0987` where the identity gives `0.2885` |
| Fork-join vs `t_ul` *(integration)* | a **symmetric** two-branch fork-join (`r = 1`, where `t_ul` is exact): the simulated `response-time` CI must bracket `t_ul`. Symmetry is what makes bracketing the right assertion shape — see §8.2. Both fork-join tests also assert `sim ≥ the slower branch's own mean`, a rigorous bound the old join-anchored value violated |
| Unsupported measure type | requesting the literal `fork-join-response-time` maps qsim's 400 to `SimulationRequestError`; pins that qopt never emits it (§5.3) |

The bracket, identity, and conservation tests are the actual validation of the idea; everything
above them is plumbing correctness.

### 8.1 Why qsim's `qopt-3station.json` is not the golden file

That fixture is a hand-authored *translation-layer* test on the qsim side, and it
deliberately exercises input forms qopt does not emit. Its fork-join branches use moment form
(`{"mean": 0.2, "scv": 1.5}` and `{"mean": 0.1, "scv": 0.5}`), but `ForkJoinStation` has no
per-branch `cov_s` and its `t_ul` approximation is built from `1/(µ − λ)` terms — i.e. it
assumes **exponential** servers. qopt can therefore only emit exponential fork-join branches,
and that fixture is not reproducible from any `Network`.

Its single-server nodes *are* consistent with §5.2 (`exponential rate 3.0` ⇒ `S·µ = 3`;
`deterministic 0.25` ⇒ `S·µ = 4`), and its tandem-at-λ=1.0 wiring is simply a different network
from §4.1.1's. So it remains a useful cross-repo reference for the shared vocabulary — just not
a byte-comparison target. Per-branch `cov_s` on `ForkJoinStation` would require a fork-join
approximation that admits non-exponential servers, which is a modeling change, not plumbing
(§10).

### 8.2 Why the `t_ul` cross-check is symmetric, and what shape each oracle takes

Simulation and `t_ul` now measure the same quantity, so they can check each other — but the
assertion shape has to match the oracle, and bracketing is only right for one of them.

| Oracle | Assertion | Why that shape |
|---|---|---|
| `system-response-time`, fork-join-only network | **equality**, `1e-9` | Same sample path, so it is an identity. CI width is irrelevant; tightening or loosening precision cannot change the verdict |
| `t_ul`, symmetric `r = 1` | **CI brackets** `t_ul` | `t_ul` is exact for equal branch rates, so the only discrepancy is sampling noise. Tighter precision strictly strengthens the test |
| `t_ul`, heterogeneous | *not asserted* — see below | `t_ul` carries a real bias here, so bracketing and precision fight each other |
| max branch `E[T]` | **`sim ≥ bound`** | Rigorous for any branch configuration; a cheap always-true assertion, folded into both fork-join tests |

**The heterogeneous case is deliberately not an acceptance criterion.** It is tempting — a probe at
λ = 1.0 with branch rates `µ = (5, 10)` gives `t_ul = 0.282906` against a simulated `0.288451`, a
1.9% gap — but a bracket test there is self-defeating. `t_ul` is exact only for equal rates, so the
gap is genuine approximation bias, not noise. Tighten the precision target below 1.9% to make the
comparison discriminating and `t_ul` falls *outside* the CI, failing a correct run; leave it loose
enough to bracket and the CI half-width exceeds the effect being measured, so the test passes
regardless. The honest formulation would assert a bounded *relative* gap with a documented tolerance
— which validates `t_ul`'s accuracy rather than qopt's plumbing, and costs a long tightly-toleranced
run. It belongs in `examples/simulated_mixed_network.py`'s printed comparison (§9), not in the gate.

The symmetric case has no such tension, and the identity has no tension at all, so those two carry
the fork-join validation.

## 9. Documentation deliverables

- `examples/mixed_network.py` — **converted** to the §4.1.1 topology, with γ derived rather
  than hand-supplied. Its analytic output must remain byte-identical to today's (budget 15.6,
  same `S*` / `E[T]` / `ζ` table), so the conversion is verifiable rather than a rewrite.
- `examples/simulated_tandem.py` — **ships first**, because it isolates variability propagation
  with the fewest moving parts. A Poisson source feeding `M/D/1 → M/M/1` in series:
  the M/D/1's departure process is not Poisson, so the downstream station's true `cov_a ≠ 1`
  while the analytic path assumes whatever `cov_a` it was given. That divergence *is*
  variability propagation, demonstrated with no fork-join involved.
- `examples/simulated_mixed_network.py` — sequenced last (§5.3), no longer gated. The §4.1.1 network solved
  analytically and by simulation side by side. Because γ is identical on both paths, the printed
  difference is attributable to variability propagation alone. Expected to show close agreement
  at `mm1` and `md1` (Bernoulli-split Poisson arrivals, so `cov_a = 1` is exact) and visible
  divergence at `fj` (non-Poisson superposition, where `t_ul`'s Poisson assumption does not
  hold) — a prediction the example should state up front and then demonstrate.
  This example is also where the **heterogeneous** `t_ul`-vs-simulation comparison is printed and
  discussed rather than asserted (§8.2), including the simulated CI alongside each number so a
  reader can see whether a gap is approximation bias or noise.
- README: replace the dashed `future: simulation analyzer` box in the architecture diagram
  with the real path, and update **Scope & limitations** — network coupling moves from
  "future work" to "supported via simulation", with closed/multi-class remaining the honest
  open limitations.

## 10. Out of scope

| Deferred | Why |
|---|---|
| **Closed chains** | λ would depend on `S` via throughput, so `γ` shifts each iteration and eq 21's budget floor moves under the optimizer (§1.2). Needs a math extension, not plumbing |
| **Multi-class networks** | Eq 21 has no per-class notion — `γᵢ`, `µᵢ`, and `ωᵢ` are scalar per station. Needs aggregate λ and a class-mix-weighted service rate, i.e. a math change |
| **Delay (infinite-server) nodes** | Cheap on the qsim side, but a delay node has no capacity to allocate, so it sits outside the optimization as pass-through latency. Additive later |
| **Non-exponential fork-join branches** | `t_ul` is built from `1/(µ − λ)` terms and assumes exponential servers (§8.1). Per-branch `cov_s` would need a fork-join approximation admitting general service — a modeling change |
| **Fork-join measures other than `response-time`** | Gated on [qsim-service#8](https://github.com/atantawi/qsim-service/issues/8). JMT defines only **two** fork-join region measures — `FORK_JOIN_RESPONSE_TIME` and `FORK_JOIN_NUMBER_OF_JOBS` — so at most `queue-length` is reachable by a type swap; the other five would need invented semantics, not a fix. That bounds how much this can ever widen, and is why `throughput` is requested but not asserted at fork-join nodes (§6.8). Nothing outside `response-time` enters eq 22 regardless, so this constrains diagnostics only |
| **Independent replications per iteration** | qsim already targets a per-measure CI internally, so single-run CIs suffice for §6.4. `qsim-service` spec §9 documents the aggregation method when this is wanted |
| **Optimizing the topology** | Routing is a fixed input by assumption (§1.2) |
| **Parallel/concurrent simulation calls** | The loop is inherently sequential — iteration `k+1` needs `S` from `k` |

## 11. Acceptance criteria

1. `Optimizer(stations, budget)` behaves bit-identically to the current implementation; the
   existing test suite passes unmodified.
2. `Network` derives `γ` for tandem, branching, and feedback topologies, matching
   closed-form expected values.
2a. The §4.1.1 topology derives `γ = (0.6, 0.4, 0.5)` and, on the analytic path, reproduces the
   current `examples/mixed_network.py` output bit-for-bit at budget 15.6.
3. `Network.to_model_dict(S)` reproduces the committed
   `tests/fixtures/qopt_mixed_network_request.json` golden fixture byte-for-byte.
4. The naive-equivalence test (§8) passes: knobs off ⇒ today's loop with a substituted
   `E[T]`.
5. Against a live `qsim-service`, a single-station M/M/1 network's simulated CI brackets the
   analytic `1/(Sµ − γ)`.
5a. `ForkJoinStation.sim_node` emits the fork-join node with both heterogeneous branches
   (`S·µ`, `S·r·µ`) and `join: "all"`, covered by the golden fixture, and inherits
   `SIM_MEASURE_TYPE == "response-time"` — the same constant single-server stations use (§5.3).
   Against a live service: (i) in a fork-join-only network, `response-time` at the fork-join node
   equals `system-response-time` to `1e-9`; (ii) a **symmetric** two-branch fork-join's simulated
   `response-time` CI brackets `t_ul`; (iii) both exceed the slower branch's own mean (§8.2).
5b. `build_request` emits exactly `("response-time", "system-response-time", "throughput")`, never
   null or empty (§5.4), and simulated throughput brackets the derived `γ` at every
   conservation-checked station of the §4.1.1 network. A miss warns and records; `strict=True`
   raises. Fork-join stations are exempt (§6.8).
6. `qopt` still declares zero runtime dependencies.
7. Every §4.2 validation row and every §7.1 exception branch has a test.
