# Design Spec: Measured arrival `cov_a` on the simulated path

Date: 2026-09-25
Status: **Implemented on feat/measured-arrival-cov.**

Companion specs:

- `docs/superpowers/specs/2026-09-22-slope-calibrated-zeta-design.md` — §8 of that spec made
  `cov_a` a live input and shipped an assumption plus a detector in its place; §8.6 named the
  measurement this spec consumes as the follow-up. This spec replaces the assumption on the
  simulated path and re-scopes the detector (§6 below).
- `docs/superpowers/specs/2026-07-29-simulation-support-design.md` — the request envelope
  (§5.4) and response extraction (§5.3) this design extends.

Upstream: qsim-service issue #15 (closed), implemented by qsim-service PR #16, merged at
`8e7358b`. Its README documents the `interarrival-time` measure this spec reads.

---

## 1. Goal

Under slope calibration, `phi` is taken from the station's analytic model even when `E[T]` is
measured. For a `GG1Station` that model depends on the arrival variability through
`k = (cov_a² + cov_s²)/2`. `cov_s` is known from the station's specification; `cov_a` is known
only for the exogenous stream. The stream a station actually sees merges the exogenous stream
with internal traffic whose variability upstream service has reshaped — exactly what a
per-station closed form cannot know, and the main reason to simulate the network at all.

On the simulated path, measure it: ask qsim-service for each station's interarrival-time
moments, compute `cov_a = sqrt(variance / mean²)`, and let `phi` use that value for the
evaluation that measured it.

### 1.1 Decisions (settled in conversation)

| # | Question | Decision |
|---|---|---|
| D1 | Lifetime of a measured `cov_a` | **Per evaluation.** Used only at that evaluation's `S`; the constructor `cov_a` is never written. A downstream station's true `cov_a` moves with the upstream capacities, so it is a function of `S`, not a constant to pin once. |
| D2 | Analytic path | **Unchanged.** The constructor `cov_a` remains the analytic model's value and the user's responsibility. Deriving a merged `cov_a` from the routing (superposition / QNA) is out of scope. |
| D3 | When to request the measure | **Only when needed**, with an opt-out: requested iff `SimulationAnalyzer(measure_cov_a=True)` (the default) **and** some station has `uses_measured_cov_a` (§4: it reads `cov_a` *and* is slope-mode). |
| D4 | The `zeta_shape_tol` cross-check | **Compares against the measured `cov_a`** when one was used, so it tests the G/G/1 approximation's shape, not the constructor argument. |
| D5 | Plumbing | **A keyword threaded through the calls** (`cov_a=None`), carried on `Evaluation`. No station state, no copies. |

### 1.2 What does not change

- The level path, bit for bit: the level arm of `zeta_from` never reads `cov_a`.
- The analytic path, bit for bit: `AnalyticAnalyzer` never measures, so every override is `None`.
- A simulated run with no slope-mode `GG1Station`, or with `measure_cov_a=False`: the request is
  byte-identical to today's and the numbers are identical.
- `cov_a` is still never *sent* to the simulator. `Network.to_model_dict` is untouched, and the
  `_refused_by_analyzer` / `evaluate` preflight reasoning about the emitted arrival process stays
  true as written.
- Policy snapshot/restore and warm start. The measurement is not station state, so the
  roll-backs need nothing new; the analytic warm-start phase has no measurement and the first
  simulated iteration picks one up.

---

## 2. Data flow

```
SimulationAnalyzer.evaluate(stations, S)
  build_request(..., measures=MEASURES + (COV_A_MEASURE,))   # only when D3's trigger holds
  POST /simulate
  extract(...)                      -> sojourn_times, ci, degraded, extras   (unchanged)
  extract_arrival_cov(...)          -> arrival_cov, degraded'                (new)
  Evaluation(..., arrival_cov=arrival_cov)

Optimizer.run
  zeta_i = st.zeta_from(T_i, S_i, cov_a=arrival_cov_i)       # eq 22, loop
  _noise_floor(stations, S, zeta, ci, arrival_cov)           # same override
  shape check: st.sojourn_time(S_i, cov_a=arrival_cov_i)     # D4
  final: zeta, zeta_phi from the FINAL evaluation's arrival_cov
  Result(..., arrival_cov=final evaluation's arrival_cov)
```

The override is passed to a station **only when it is not `None`**. That one rule gives
three properties at once: a user subclass whose `phi(self, S)` or `sojourn_time(self, S)`
override predates this keyword keeps working on every path that never measures; `ForkJoinStation`
is never called with it (its entry is always `None`); and the analytic path executes the same
calls it does today.

---

## 3. The qsim layer

### 3.1 Request — `qopt/qsim/spec.py`

- `MEASURES` is unchanged.
- New constant `COV_A_MEASURE = "interarrival-time"`, documented with the qsim-service README
  facts it relies on: the measure turns on JMT's per-sample logging for itself (≈25–30% wall
  clock and temp disk per measure, in qsim-service's README), reports `mean`/`variance`/`stdDev`,
  reports **no CI** (JMT's interval is on the rate), and on a fork-join node is measured at the
  fork. The measure list is network-wide, so the measure is taken at every station, not only at
  the ones that use it, and its cost grows with network size: qopt's own runs at a fixed seed
  measured +35% wall clock on a 3-station network and about +50% on a 14-station one.
- `build_request` keeps its signature; the analyzer passes the extended tuple.
- `secondMoments: true` is **never** sent. `interarrival-time` enables sample logging for itself
  alone; the flag would put every other measure on the logging cost too.

### 3.2 Response — `qopt/qsim/measures.py`

New function `extract_arrival_cov(response, stations, job_class) -> (arrival_cov, degraded)`.
`extract` is unchanged.

For each station with `uses_measured_cov_a`, find `(st.name, job_class, "interarrival-time")`:

| Condition | Entry | Audit |
|---|---|---|
| measure absent | `None` | warn + `degraded` |
| `mean` missing, non-finite, or `<= 0` | `None` | warn + `degraded` |
| `variance` missing, non-finite, or `< 0` | `None` | warn + `degraded` |
| `sqrt(variance / mean²)` non-finite | `None` | warn + `degraded` |
| `success == false` | the value | `_flag_weak` (used, flagged) |
| otherwise | the value | none |

- A measured SCV of exactly 0 is valid (deterministic arrivals) and is returned as `0.0`.
- The missing CI is the documented contract, so it is **not** flagged.
- A station without `uses_measured_cov_a` — a level-mode `GG1Station` included — gets `None`
  silently, even though the response carries the measure for it (the measure list is
  network-wide). Reading it would record a measurement that enters nothing, and flag a missing
  one that costs nothing.
- `mean**2` is never formed: the SCV is `variance / mean / mean`, so a tiny `mean` overflows to
  a rejected `inf` instead of underflowing `mean**2` to a `ZeroDivisionError`.
- Every warning names the station and the reason, and says the constructor `cov_a` is used for
  that station in that evaluation.

### 3.3 Analyzer — `qopt/qsim/analyzer.py`

- `SimulationAnalyzer.__init__(..., measure_cov_a=True)`.
- `evaluate` computes the D3 trigger from the stations it is given, extends the measure tuple
  when it holds, and when it does calls `extract_arrival_cov` and appends its `degraded` to the
  evaluation's. When it does not, `arrival_cov` stays `None`.
- `strict=True` raises on a degraded measurement like on any other degradation.

### 3.4 `Evaluation` — `qopt/analyzer.py`

New field `arrival_cov: list | None = None`: per-station measured `cov_a`, aligned to the station
order, each entry a float or `None`; the whole field `None` when nothing was measured. One home —
not duplicated into `extras`. `AnalyticAnalyzer` leaves it `None`.

### 3.5 Version floor

Measuring needs qsim-service `8e7358b` or later. An older build answers the POST with
`unsupported measure type: 'interarrival-time'`, which the client already raises. There is **no**
retry without the measure — that would silently fall back to the constructor `cov_a`, the
assumption this feature exists to replace. The raised error is wrapped (or its message extended)
to name `measure_cov_a=False` as the way to run against an older service. The README's version
floor is updated.

---

## 4. The station layer — `qopt/station.py`

- New class attribute `Station.reads_arrival_cov = False`; `GG1Station.reads_arrival_cov = True`.
- New property `Station.uses_measured_cov_a = reads_arrival_cov and zeta_mode == ZETA_SLOPE`, and
  the station's bound `phi`, `sojourn_time` and `dT_dS` each accept a `cov_a` keyword (checked
  with `inspect.signature`; see §7.3): the single predicate both the request trigger (§3.3) and
  the extraction (§3.2) read.
- `Station.phi(S, *, cov_a=None)`: the base implementation ignores `cov_a` (no base-class model
  reads one).
- `Station.zeta_from(T, S, *, cov_a=None)`: the slope arm calls `self.phi(S, cov_a=cov_a)` when
  `cov_a` is not `None` and `self.phi(S)` otherwise; the level arm never reads it.
- `GG1Station`:
  - `sojourn_time(S, *, cov_a=None)` and `dT_dS(S, *, cov_a=None)` — widened, not bypassed, so
    the optimizer's shape check calls a public method. Both take `k` from one private
    `_k(cov_a)`, which for `None` evaluates today's expression on the constructor `cov_a`, so the
    default arithmetic is unchanged and the existing tests pin it. An override is validated like
    the constructor argument (finite, `>= 0`).
  - `phi(S, *, cov_a=None)` overrides the base: without an override it defers to the base
    arithmetic; with one it evaluates the same expression on the widened pair.
  - `admits_full_utilization` stays on the constructor `k`. The `phi == 0` boundary branch in
    `zeta_from` cannot be reached on the simulated path, because `S*mu == gamma` is refused by
    the analyzer first; a comment says so rather than a guard.
- `ForkJoinStation` inherits `reads_arrival_cov = False` and the ignoring base `phi`. `t_ul` has no
  `cov_a`.

---

## 5. The optimizer — `qopt/optimizer.py`

A helper returns `evaluation.arrival_cov[i]`, or `None` when the field is `None`. It is used at all
four places that read `phi`, each time with the evaluation whose `E[T]` is being inverted:

1. the eq-22 `zeta` list in the loop;
2. `_noise_floor`, which gains an `arrival_cov` argument. It passes a CI half-width in the `T`
   position and relies on `zeta_from` being linear in `T`; that holds only if the half-width goes
   through the **same** map, i.e. the same override;
3. the shape check (§6);
4. the final `zeta` and `zeta_phi` recomputation.

New `Result.arrival_cov`: per-station measured `cov_a` that produced the reported `zeta`, `None`
where the constructor value was used, and an empty list when nothing was measured (matching how
the other defaulted diagnostic lists are constructed). Like `Result.zeta` it comes from the
**final** evaluation, which on a stochastic path is a different sample from the loop iterate that
set the capacities — the round-6 scoping rule, stated in the field's docstring.

---

## 6. The shape cross-check (D4)

Today `T_model = st.sojourn_time(Si)` with the constructor `cov_a`, and the message sends the user
to check that argument. With a measurement:

- `T_model = st.sojourn_time(Si, cov_a=c)` for the measured `c`, so the check asks whether the
  G/G/1 (Allen–Cunneen) curve, fed the variability the station actually sees, reproduces the
  measured `E[T]`. That is the assumption `phi` still rests on.
- The message names the measured value and blames the approximation's shape, not `cov_a`.
- Without a measurement for that station (fallback, opt-out, or a station type that does not read
  `cov_a`) the check and its message are exactly today's.
- Warn-once-per-station is unchanged. The flag remains advisory and out of `degraded`.

`ZETA_SHAPE_TOL`'s docstring is updated: on the simulated path with measurement on, it now bounds
the approximation's error, and its "no G/G/1 with cov != 1 in the evidence" caveat still applies.

---

## 7. Stated limitations

1. **No CI on the SCV.** qsim reports none, so `_noise_floor` does not account for the sampling
   error in `cov_a`. The floor is therefore a lower bound on the step noise of a slope G/G/1
   station on the simulated path.
2. **Seed policy.** Under the default `seed_policy="fixed"` (common random numbers) the measured
   SCV is a deterministic function of `S` and adds no iteration-to-iteration noise. Under
   `"vary"` it does, and that noise is not in the floor either.
3. **Only `GG1Station` consumes it.** `ForkJoinStation`'s model has no arrival variability. The
   opt-in `reads_arrival_cov = True` is set on `GG1Station`, so every `GG1Station` subclass
   inherits it, including one written before the keyword existed (main's README advertised
   overriding `phi(S)` on a subclass). What protects such a subclass is a signature check:
   `uses_measured_cov_a` also requires the bound `phi`, `sojourn_time` and `dT_dS` to accept a
   `cov_a` keyword (a parameter of that name, or `**kwargs`). A subclass that overrides any of
   the three as `(self, S)` is therefore not measured: it is priced at its constructor `cov_a`
   and its `arrival_cov` entry is `None`, as on main. A subclass of `Station` outside the
   `GG1Station` tree opts in by setting `reads_arrival_cov = True` and accepting `cov_a` in all
   three methods.

---

## 8. Error handling summary

| Situation | Behaviour |
|---|---|
| One station's measurement missing or degenerate | warn, `degraded`, entry `None` → constructor `cov_a` for that station in that evaluation only |
| `success=false` on the measure | used, flagged via `_flag_weak` |
| qsim older than `8e7358b` | POST fails loudly; message names `measure_cov_a=False`; no silent retry |
| `strict=True` | a degraded measurement raises, like any other degradation |
| Old-signature `phi` / `sojourn_time` / `dT_dS` overrides | the station is not measured (the signature check in `uses_measured_cov_a`, §7.3), so it is never called with the keyword; priced at its constructor `cov_a`, `arrival_cov` entry `None` |

---

## 9. Testing

TDD. Every new test is mutation-checked by reverting the code it covers and watching it fail
(clear `__pycache__` between a mutation and its restore, or a same-size edit reuses stale bytecode).

- **Request.** Byte-identical to today's when there is no slope `GG1Station` and when
  `measure_cov_a=False`; carries `interarrival-time` and no `secondMoments` when the trigger holds.
  A level-mode `GG1Station` alone does not trigger it.
- **Parsing.** Table-driven over §3.2's rows. SCV 0 accepted; `success=false` used and flagged;
  a non-reading station gets `None` with no message. The response fixture follows the shape of
  qsim-service's own `src/test/resources/results/interarrival.solutions.xml` output as parsed by
  its `SolutionsParser` (null `lower`/`upper`, populated `variance`), not an invented shape.
- **Station.** `GG1Station(cov_a=a).phi(S, cov_a=c) == GG1Station(cov_a=c).phi(S)` exactly;
  likewise `sojourn_time`. With `cov_a=None`, `phi`, `sojourn_time`, `dT_dS` are bit-identical to
  `main`. The level arm of `zeta_from` ignores the override. `ForkJoinStation` ignores it.
  An old-signature `phi(self, S)` subclass still runs on the analytic path.
- **Optimizer**, with a fake analyzer returning `arrival_cov`:
  - a measured value moves `zeta`, and removing the plumbing at each of §5's four points alone
    makes some test fail;
  - the noise floor equals the one computed by hand through the same override;
  - `Result.zeta`, `Result.zeta_phi` and `Result.arrival_cov` come from the **final**
    evaluation — the fake returns a different value there than in the loop, so a mix-up shows;
  - the shape check uses the measured value and emits the new message; without a value it emits
    today's;
  - an all-`None` `arrival_cov` reproduces today's `Result` exactly.
- **Analytic path.** Existing suite green; one test asserts `Evaluation.arrival_cov is None` and
  an unchanged `Result` on the reference network.
- **Live** (skips without `QOPT_QSIM_URL`; run with `run_in_background`):
  - *Burke:* an M/M/1 feeding a downstream station; the downstream measured `cov_a` is 1 within a
    tolerance derived from the run's `samplesAnalyzed` and stated in the test.
  - *Reshaping:* Poisson arrivals through a `cov_s = 0` station at `rho = 0.9` into a slope
    `GG1Station`. For an M/G/1 the *marginal* interdeparture SCV is exactly
    `1 − rho²(1 − cs²)` (condition on the queue being empty after a departure), here `0.19`.
    Successive interdeparture times are correlated, so the iid standard error understates the
    estimate's; the assertion is a stated tolerance around 0.19 plus the direction (`< 0.5`).

---

## 10. Documentation ripple

- README: `measure_cov_a`, the new version floor, `Result.arrival_cov`, the measurement's cost.
- `qopt/zeta.py` module note and `ZETA_SHAPE_TOL`: `cov_a` is measured on the simulated path;
  the check's meaning under D4.
- `Station.phi` docstring: the "override `phi` if you cannot express `cov_a`" advice now applies
  to the analytic path, and to the simulated path only with `measure_cov_a=False`.
- `optimizer.py`'s shape-check comment and message.
- The slope-calibrated-ζ spec §8.6: mark the follow-up as consumed, pointing here.

---

## 11. Out of scope

- Deriving a merged `cov_a` from the routing on the analytic path (D2).
- A measured `cov_s` as a cross-check of the configured value.
- Fork-join diagnostic measures (qsim-service#8).
- A CI for the SCV (would need a batch-means estimate upstream).
