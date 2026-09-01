# Choosing S₂ — what was implemented, and what it measured

Companion to [`findings.md`](findings.md), whose §10 ("If this were to be implemented") was
explicitly undecided. This records the decisions taken, the numbers the implementation
produces, and the seven places measurement contradicted the plan. Issue #10; PRs #12 and #13.

The figures in the results table and the corrections are **pinned in the committed suite**,
so they are executable rather than transcribed: `tests/test_example_qcsc.py`
(`FINDINGS_SECTION_7`, `FINDINGS_BEST_RAY`, `TUNED`), `tests/test_forkjoin_policy.py`
(`FINDINGS_SECTION_4`, `FINDINGS_Q7`), `tests/test_optimizer_loop.py` (`FJ_SWEEP` and the
descending-sweep tests) and `tests/test_allocator.py` (the floor's bit-exactness).

The **Validation section is different** and should be read as a record, not as a test: its
sweep counts, the `float.hex()` comparison against pre-change code, and the iteration figures
come from ad-hoc runs that are not committed here. Where a Validation bullet *is* pinned, it
says so.

## The API

`ForkJoinStation(..., r_star=...)` takes either a positive float — some fixed ray of the
family — or one of three named policies:

| `r_star` | ray | `alloc_cost` | |
|---|---|---|---|
| `R_STAR_INVARIANT_R` | `r* = r` | `c₁ + c₂` | the default; qopt's incumbent |
| `R_STAR_EQUAL_RATE` | `r* = 1` | `c₁ + c₂/r` | the paper's rule |
| `R_STAR_TUNED` | solved each iteration | `c₁ + c₂·r*/r` | §8's local optimality condition |

`qopt/forkjoin_policy.py` solves `|∂T/∂m₁|/|∂T/∂m₂| = β₁/β₂` by bisection on the station's
spend line. `Station.retune(S) -> S` is a new base-class hook, a no-op on every station but a
tuned fork-join, which the `Optimizer` calls once per iteration; it reprices the station and
returns the capacity that buys the *same spend*, so the budget stays exhausted and `S` keeps
meaning "server 1's capacity". Its counterpart `Station.reset_policy()` restores the
constructed ray, and `Station.min_spend` reports the floor at that ray rather than at the
current one — see the fifth correction below for why both are correctness requirements
rather than hygiene, and why neither works without the other, and the sixth for what still
had to be checked inside `allocate` itself.

**The nesting is one-sided.** At a fixed spend the inner optimum is determined — one scalar
solve, no inner loop. The fixed point closes through the outer loop, because the spend comes
from eq 21, whose prices depend on the `r_star` chosen here.

## Results — the 14-station QCSC network at C = 41.040000

Same budget, same cost vector, and the same topology findings §7 used.

| workload | invariant-r | equal-rate | **tuned** | §7 grid best | tuned r\* | §7 best ray | it |
|---|---|---|---|---|---|---|---|
| balanced | 6.401440 | 6.401440 | **6.249429** | 6.249439 | 1.447382 | 1.440 | 6 |
| quantum_dominant | 4.528844 | 4.776428 | **4.431691** | 4.431693 | 2.316118 | 2.320 | 6 |
| classical_dominant | 3.463677 | 2.613335 | **2.613335** | 2.613335 | 1.000000 | 1.000 | 4 |

- Tuning **reaches or marginally beats** the objective a 0.02-grid sweep of `r_star` found, in
  all three workloads. It is free to land between grid points, so "no worse than" is the
  assertion; it may not land worse.
- `classical_dominant` recovers `r* = 1` **exactly**, which is `t_bot`'s kink — the one
  workload where the paper's rule is the true optimum, and where the incumbent loses 24.55%.
- The two incumbent columns reproduce findings §7 digit-for-digit, which is what makes the
  tuned column same-money comparable to it.
- The default policy is unchanged, so no previously recorded number moves. Verified by
  comparing `float.hex()` of every capacity, ζ and objective against the pre-change code
  across three workloads × four rays.

A note on §7's other triple, 1.4457 / 2.3195 / 1.0000, which that section calls "the local
optimality condition evaluated at the converged spend": those were computed at the spend the
**inner-split embedding** converged to, and §10 item 4 rejects that embedding — it converged
at a different spend (7.96 against 7.49). They agree with the tuned rays to 1.2e-3 and 1.5e-3.
That is corroboration, not a specification, so the tests compare against the sweep grid.

## Seven corrections to the plan

**§10 item 3 named the right rule, but the probe's method cannot deliver it.** Minimizing
`t_ul` along the spend line pins `r*` only to `√ε` (~1e-8 relative), because a quadratic
minimum is flat and locating it by comparing function values cannot do better. That noise
reaches `alloc_cost` and stalls the outer fixed point, which jitters above `tol = 1e-9` for
several iterations after the answer has been reached: **9 iterations against the incumbent's
5**. Bisecting the stationarity condition is exact — a sign change is not flat — and costs 6.
The 9 is a *coincidence* with §7's 9 under the inner split, which came from a mispriced ζ on a
different network.

**§10 item 2's anchoring makes a post-retune feasibility guard unnecessary — but only once the
returned ray is guarded against float cancellation.** A spend-preserving retune lands the
station strictly inside its own stability region, so `Σfloor < Σspend = C` and the budget
cannot become infeasible. That argument fails on the arithmetic: `m₂` is recovered as
`(spend − β₁m₁)/β₂`, carrying `~ε·m₁·(β₁/β₂)` of absolute error. At `β₁/β₂ = 1e5` and
`γ = 0.45`, one ulp of the spend is 1e-11 against a true `m₂ − γ` of 1e-12, and `m₂` comes back
*below* γ — an unstable ray on a spend that admits stable ones, escaping as an
`InstabilityError` raised from inside the optimizer loop. Every such case has `r* = 1`, and
not by luck: it only triggers at the stability boundary, where `r* → 1` (§4). So the ray is
reconstructed directly there, which needs no cancelling subtraction.

**A tuned station must start at `r* = 1`, not on the incumbent ray.** The station's floor over
the family, `γ(c₁ + c₂r*/r)/(μ̂₁·min(1,r*))`, is minimized at exactly `r* = 1`, where it equals
the spend line's own floor `γ(β₁+β₂)` — which is findings §4's observation, under
"The paper's ray has the optimal stability floor", that qopt's is up to 2.5× worse.
`min_feasible_budget` is evaluated **once**, before any retune, so a
tuned station starting at `r` advertises the incumbent's floor and refuses budgets it can in
fact serve: on the `γ=.45, μ=1, r=4, c₁=4, c₂=1` station that is every `C` in
`(1.9125, 2.2500]` — from the equal-rate floor up to the incumbent's — all of which
`equal-rate` completes. Converged answers are bit-identical either way.

**`r_star` is not free of simulation noise.** It is a function of the station's spend, and on
the simulated path that spend descends from a measured `E[T]`: injecting ±2% noise into `E[T]`
moves the converged `r_star` by ~6e-4 relative. It still needs no damping of its own, but for
a different reason than "it is deterministic" — the `S` it reads has already been damped, so
damping the retune too would attenuate one perturbation twice.

**A mutating policy parameter needs an explicit reset — and the floor it moves has to be
reported for the policy, not for the ray.** The starting-ray correction above only holds for a
station *as constructed*. `retune` mutates, so a tuned station that has already run sits on that
run's ray, which — being anything other than `r* = 1` — needs strictly more budget to stay stable
than the policy does. Measured on `γ=.45, μ=1, r=4, c₁=4, c₂=1`, a run at `20×` the floor leaves
`r* = 2.69255` and lifts that ray's floor from `1.91250` to `2.10291`.

Two failures came out of that one fact, and they need **two** guards. Neither alone is enough,
which is why this correction is stated as a pair:

- With `min_feasible_budget` reading the current ray, a station reused at a lower budget was
  rejected against the **previous** run's floor: `C = 2.008125` raised `InfeasibleBudgetError`
  while a freshly constructed equivalent converged, so a descending budget sweep broke partway
  down and feasibility depended on run history. The exported helper also *disagreed with the
  optimizer* once the reset existed — it reported `2.10291` while `run()` served every budget
  above `1.9125` — and the README derives budgets from that helper. Fixed by
  `Station.min_spend`, a hook the
  allocator sums: it prices the floor at the ray a run **starts from**, which under `tuned` is
  also the family's minimizer, and is the plain `alloc_cost·γ/μ` for every other station.
- Repricing the floor is not a substitute for restoring the ray, and this was checked rather
  than assumed: eq 21 prices each station at its *current* ray, so at budget `2.00771` — which
  the policy-aware floor now clears — a stale ray gives slack `−0.09521` and the run dies on an
  `InstabilityError` (`S·μ = 0.42963 ≤ γ = 0.45`) from inside the loop instead of running.

So `reset_policy()` sits **after every preflight guard** and **before the first `allocate`**.
After, because validation must not mutate: a budget that never passes preflight used to reset
the stations anyway, discarding a converged ray that is a *reported output* (`r_star`, and
per-unit capacity attribution). Before, for the slack reason above. Each of the four positions —
floor policy-aware or not, reset before the guards, after `allocate`, or absent — is failed by a
different subset of the suite.

It also removes the caveat the `retune` docstring used to carry: a run is now a pure function of
(stations-as-constructed, budget), and reusing tuned station objects reproduces a fresh run
**bit-for-bit**, iteration count included.

**Splitting the floor in two put a hole in the composition of two exported functions.**
`min_feasible_budget` answers for the policy; `allocate` prices the current ray. They are both
root-exported, and the helper's documented guarantee is that any budget above it allocates
stably — so a tuned station left on a finished run's ray broke it: `allocate([st], 2.0, ...)`
returned `S = 0.42798` against `γ = 0.45`, an unstable capacity, silently, because eq 21 has no
stability test of its own and a non-positive slack term *subtracts* from the base `γ/μ`.

`allocate` now checks its own precondition and raises `InfeasibleBudgetError` naming the floor
it actually prices. Documenting the case as out-of-contract was not enough: the guarantee is
public and the two functions have always composed.

Making that check exact surfaced a **pre-existing** ulp bug, latent since before this PR.
Eq 21 needs `base = γ/μ` for the capacity formula and prices the floor as `Σ c·base`, i.e.
`c·(γ/μ)`; the helper summed `c·γ/μ`, i.e. `(c·γ)/μ`. Those are adjacent floats whenever `μ` is
not a power of two — `(0.7·0.1)/0.3` against `0.7·(0.1/0.3)` — and the helper came out *lower*,
so a budget in the one-ulp gap passed the Optimizer's guard and then met a non-positive slack.
Before the check that returned capacities below the stability boundary in silence. `min_spend`
now carries eq 21's grouping deliberately, and a test asserts the two agree bit-for-bit by
allocating at `nextafter(floor, inf)`.

**Also hardened, though not a correction to the plan:** `optimal_ray` is root-exported and had
no input validation, so `spend = inf` returned `nan`, a zero rate or cost raised a raw
`ZeroDivisionError` from inside the bisection, and `γ = nan` surfaced as an `InstabilityError`
quoting a `nan` floor — an arithmetic accident reported as a modelling result. Its arguments now
get the same treatment `ForkJoinStation` gives the constructor arguments they mirror.

**A pre-merge review found the composition still had two holes, one of them silent.** Four
independent reviewers went over the whole change; 32 distinct findings, of which these are the
ones that were not merely prose:

- **`allocate` validated its budget but not its `zeta_vec`.** A short vector was the bad one:
  `zip` truncates, so it returned *fewer capacities than there were stations* and renormalized
  the budget across the survivors, silently. A zero left its station at exactly `S·μ = γ` with
  the budget far above the floor — the same silently-unstable outcome the ray guards were added
  to prevent, reached through the other argument. Now validated, as is `noise_floor`'s `dzeta`.
- **The bisection had no NaN guard.** `_dt_dm1` squares rate differences, which `t_ul` never
  does, so the tuned policy runs out of exponent range long before the rest of the library: it
  overflows above ~1.3e154, underflows below ~1.5e-162, and *between those it evaluates to NaN
  with no exception*. Bisection tests `g(mid) < 0.0`, which is False for NaN, so the bracket
  narrowed the wrong way and returned a confidently wrong ray — measured 32.9% off at
  `γ = 1e-154`, against the same case scaled up by 1e120, under which `t_ul` is invariant. All
  three modes are now one `ValueError` that names the scale and says a fixed ray still runs.
- **Two claims had to be withdrawn rather than fixed.** `allocate`'s promise that any budget
  above the floor yields stable capacities is false in float arithmetic and cannot be made
  true by a check: the slack term is an *aggregate*, and within a few ulps of the floor a
  station's share rounds away entirely, leaving `S·μ = γ` exactly — measured on plain M/M/1
  stations, 206 of 12052 probes. And the cancellation rescue at the end of `_min_on_spend_line`
  is a *stability* rescue, not the optimum it was described as: at `β₁/β₂ = 1e16` it returns
  `r* = 1` at `t_ul = 13750` where the best point on the same spend line is 10002, 37% better.
  Both are now documented as what they are.

Three of the guards were also **green under mutations that reverted real behaviour**, which is
its own lesson: the floor's bit-exactness test compared the helper against a *hand-copy* of eq
21's expression rather than against `allocate`, so regrouping `allocate`'s own floor went
unnoticed; the reset loop passed as `stations[0].reset_policy()` because every reuse fixture put
the tuned station first; and the slack guard's stated NaN rationale had no test. All three now
fail when reverted.

## Validation

- Suite 406 passed / 10 skipped.
- 348 runs over budget multiples from `1.000001×` the floor, eight hardware configurations and
  three damping values, warnings promoted to errors: all converged, budget exact at every
  answer, every station stable, and **45 runs in the `r* < 1` regime**.
- A further 432 runs re-walk that grid over **reused** station objects, ascending and
  descending, each against a freshly constructed control: every one converged, reuse
  reproduced the control **bit-for-bit**, and the exported floor never moved (162 of them in
  the `r* < 1` regime). Interleaved with them, 1296 deliberately rejected runs — below the
  floor, at it, and non-finite — none of which disturbed the answer the preceding run had
  reached, and 864 sub-floor `allocate` calls, every one refused rather than served. The
  loop's own allocations kept a slack margin of at least `9.3e-7` of `C` at every converged
  point, so the new precondition never fires on a legitimate path. The sweep fails on the
  pre-reset code at the second budget it tries, and on the round-1 code at the first
  rejected run.
- Every guard this took is mutation-checked independently: the float-cancellation rescue in
  `_min_on_spend_line`, the policy-aware floor, the reset's placement (above `allocate`,
  below the preflight guards), `allocate`'s slack precondition, and eq 21's grouping in
  `min_spend`. Two of those needed a *specific* fixture to be pinned at all: the grouping
  needed a single-server station, because the fork-join cases route through the override and
  reverting the base-class expression alone left the suite green; and the reset loop needed a
  fixture with the tuned station *last*, because every other one puts it at index 0 where
  `stations[0].reset_policy()` passes.
- Iteration counts stay the same order as the incumbent's at every damping. On the
  `_tuned_pair` fixture at 4× the floor: **183 against 187** at `θ = 0.1`, **30 against 30**
  at `θ = 0.5`, **6 against 5** at `θ = 1.0`. Only the last of those is asserted
  (`test_tuning_does_not_degrade_convergence`, `tuned <= incumbent + 2` at the default
  damping), and the bound is not universal — a search over hardware and budget found tuning
  costing up to +32 iterations at `θ = 0.1` (203 against 171). What the suite does pin is
  that the *inner solve's precision* is what drives the count: injecting `√ε` noise into
  `optimal_ray` takes the same fixture from 6 iterations to 9, and `1e-6` noise to
  non-convergence.
- `r*` over a budget sweep (125 multiples of each workload's own floor, `1.000001×` to 40×)
  independently reproduces §4's two limits: it is **1.0000 at the stability boundary** in every
  workload, and rises to 1.6369 in `balanced` against §4's asymptote of 1.633 and to 2.7027 in
  `quantum_dominant` against 2.693. `classical_dominant` is 1.0000 throughout, as β₁/β₂ = 1
  requires.
- The floor a tuned station advertises comes out 6.8400 / 5.8275 / 2.7900, which are exactly
  findings §7's *paper* floors (6.840 / 5.828 / 2.790) — the ones §4 identified as optimal.
  That is the starting-ray correction above, visible from the outside. It now holds for a
  reused station too, and that needed `min_spend`: with the helper reading the current ray,
  a station that had already run advertised its converged ray's floor instead, so this
  sentence was true only of freshly constructed stations.

## Simulated cross-check — the gains hold, and the closed form gets their size right too

Run 2026-09-01. `docs/forkjoin-s2-policy/simcheck.py`, output committed as
`simcheck-output.txt`. **No `qopt/` change** — `sim_node` emits whatever ray the station is
on, so the whole cross-check is a grid of existing runs.

Conditions: `qsim-service` built from source at `b022c53` (docs-only on top of `df45cd1`, so
it clears the version floor), verified by measurement rather than by tag — `samplesAnalyzed`
scales with `minSamples` (87,040 / 174,080 / 696,320 on an M/M/1 probe) and `alpha` returns
`0.05` rather than its complement. The probe measures this itself, as section 0 of its output. Stopping rule `precision 0.02`, `alpha 0.05`, `minSamples 1e5`,
`maxSamples 4e6`, `maxWallClockSeconds 300` — the same rule as
`docs/qcsc-example/live-run.log`, deliberately, so the two are comparable. One shared budget
`C = 41.040000` for every cell. 3 workloads × 3 policies × 5 base seeds = 45 passes in ~12 minutes;
all 45 converged, all on `stop_reason = noise-floor` at `sim_calls = 2`.

**Two witnesses before any of it.** `python -m examples.qcsc_network` reproduces the committed
`live-run.log` bit-for-bit (the single diff is the caption that directory's README already
records as stale by wording), and the probe's own `invariant-r` cells at seed 20260729 come
back at **6.373131 / 4.518446 / 3.448158** — digit-for-digit the objectives spec §7 records.
The analytic column is gated against PR #13's recorded numbers before a single comparison is
printed, and it reproduced all nine cells exactly.

### The result

Gains are paired by seed: within a seed the three policies share the base seed, so
`seed_policy="fixed"` gives common random numbers and the *policy difference* is the
low-variance quantity. The interval is two-sided 95% Student-t on those paired differences —
it measures the variability instead of assuming a correlation structure.

| workload | policy | analytic gain | measured gain | 95% CI (paired) | |
|---|---|---|---|---|---|
| balanced | equal-rate | 0.00% | 0.00% | exactly 0 | identical ray |
| balanced | **tuned** | **2.37%** | **2.54%** | (+2.163, +2.909) | **confirmed** |
| quantum_dominant | equal-rate | −5.47% | −5.55% | (−5.699, −5.393) | confirmed loss |
| quantum_dominant | **tuned** | **2.15%** | **2.15%** | (+1.944, +2.357) | **confirmed** |
| classical_dominant | equal-rate | 24.55% | 24.47% | (+24.329, +24.607) | confirmed |
| classical_dominant | **tuned** | **24.55%** | **24.47%** | (+24.329, +24.607) | **confirmed** |

**Every confirmed row is confirmed, and every analytic gain lands inside its own measured
interval.** Counted honestly there are **four distinct comparisons**, not five: in
`classical_dominant` the tuned ray solves to exactly `r* = 1`, so its `equal-rate` and `tuned`
rows are one measurement printed twice — the same collapse the `balanced` row is annotated for.
The output now lists both bit-identical pairs explicitly. So this is stronger than findings §9
asked for: it
establishes not just that the policy *ranking* survives measurement, but that the closed form
gets the *magnitude* of each gain right to within the interval. The two ~2% gains that findings
§9 called "suggestive, not established" are now established. Nothing came out inconclusive, so
the planned tightening pass to `precision 0.01` was never needed.

**Pairing is what did that, and the propagated interval shows why it was necessary.** The
conservative half-width on a single measured objective — `Σᵢ hᵢ`, correlated-errors — runs
0.743% to 1.151% across the 45 cells. Against that, a 2.15% gain is barely two half-widths and
would have stayed marginal. The paired CRN interval on the same 2.15% is ±0.21 pp. Reporting
only the propagated width would have left the exact two gains this run existed to settle
unresolved.

### The pre-registered risk directions are not resolved, and that is the honest reading

Both directions were stated in advance, and the measured means do line up with them —
`classical_dominant` shrank (24.55% → 24.47%) as predicted for a workload whose baseline sits
at `r = 4` where `t_ul` is least validated, and `balanced` grew (2.37% → 2.54%), which was the
cell that could go either way. **But neither shift is resolved at this stopping rule, and
saying otherwise would be reading noise.** The per-seed gains, which the output now prints
under each row, are why:

| workload | analytic | per-seed measured gains | mean | paired 95% CI |
|---|---|---|---|---|
| balanced | 2.37% | 2.158, 2.452, 2.463, 2.624, 2.982 | 2.54% | (+2.163, +2.909) |
| quantum_dominant | 2.15% | 1.880, 2.114, 2.200, 2.263, 2.295 | 2.15% | (+1.944, +2.357) |
| classical_dominant | 24.55% | 24.361, 24.388, 24.425, 24.538, 24.628 | 24.47% | (+24.329, +24.607) |

The analytic gain lies **inside both the per-seed range and the paired interval in all three
workloads**. The offsets — +0.16, +0.005 and −0.08 pp — are each well under their own interval
half-width (0.37, 0.21, 0.14 pp). So what this run resolves is that the gains are real and
roughly the predicted size; the second-order question of whether measurement shifts them up or
down is below its own resolution.

**One error worth recording, because it is the trap this network invites.** The first reading of
this run attributed `balanced`'s risen mean to its fork-join rows: at the committed seed the
tuned station's `fj` gaps are −1.180% / −0.020% against the incumbent's −1.042% / +0.047%, so
correcting `T` takes more off the tuned side and the gain should widen. The sign logic is right
and the conclusion was wrong, because **the committed seed is the one seed where `balanced`'s
gain narrows** — 2.158%, the minimum of the five — while the other four widen it. Two of
fourteen stations do not determine the network objective, and a mechanism read off one seed's
two rows was contradicted by that same seed's own objective. The per-seed rows are printed
precisely so this cannot be done again.

### `r_star` carries the sample path, as predicted, and it is small

| workload | analytic `r*` | measured mean | min | max | relative spread |
|---|---|---|---|---|---|
| balanced | 1.447382 | 1.447268 | 1.447016 | 1.447419 | 2.8e-4 |
| quantum_dominant | 2.316118 | 2.315949 | 2.315493 | 2.316506 | 4.4e-4 |
| classical_dominant | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2.2e-16 |

n = 5 seeds × 2 fork-joins, pooled: the two stations are analytically identical to the last
bit, so their divergence on the simulated path is the same sample-path noise this measures.
The 2.8e-4 / 4.4e-4 sits right where the ±2% injected-noise experiment above put it (~6e-4),
which is corroboration from a different route. `classical_dominant` stays pinned at exactly
1.0 because it is at the family's boundary, where the solver reconstructs the ray directly.

### Model bias at the tuned operating point is the same size as at the incumbent

This is the reassurance findings §9 wanted, since the whole doubt was that `t_ul` might be
worse where the tuned policy operates. Pooled over 210 station rows per policy (14 × 3
workloads × 5 seeds):

| policy | rows | negative | mean gap | worst row | over own CI |
|---|---|---|---|---|---|
| invariant-r | 210 | 136 | −0.126% | 1.165% | 9 |
| equal-rate | 210 | 132 | −0.110% | 1.165% | 6 |
| tuned | 210 | 134 | −0.132% | −1.522% | 7 |

The tuned column is **−0.132% against the incumbent's −0.126%** — indistinguishable — and its
worst row is −1.522% against 1.165%, modestly worse and in `balanced`, exactly where moving
off `m₁ = m₂` predicts it. That contrast is the load-bearing result of this table, and it is
sound: the three policies ran at the *same five seeds*, so it is a paired comparison, which is
the only thing the shared sample paths permit.

### A claim withdrawn: this does NOT replicate spec §7's bias measurement

The first write-up of this run said spec §7's negative lean "replicates a fifth, sixth and
seventh time" on the strength of 136 / 132 / 134 of 210 rows being negative. **That is wrong,
and the arithmetic says so.** `SEEDS` contains spec §7's own four replication seeds, the
`invariant-r` cells run its ray at its budget under its stopping rule, and the service is
deterministic given a seed — so those cells reproduce its rows *bit-for-bit*. The probe now
prints the split:

| bucket | rows | negative | mean gap | sign-test p |
|---|---|---|---|---|
| `invariant-r` at spec §7's seeds — **reproduces it** | 168 | 115 | −0.149% | 0.0000 |
| `invariant-r` at a seed spec §7 never ran — **new** | 42 | 21 | −0.032% | 1.0000 |
| the two new rays, all seeds — **new** | 420 | 266 | −0.121% | 0.0000 |

115 / 168 at −0.149% is **exactly** spec §7's published pooled figure. That makes those rows a
*pipeline witness* — the same role the objectives 6.373131 / 4.518446 / 3.448158 already play
above — and not a second sample. This script's own docstring says "a repeat confirms the
pipeline and is NOT a second sample"; the first write-up broke its own rule.

What is genuinely new, and what it says:

- **One new seed, and it shows no lean at all**: 21 of 42 rows negative, mean −0.032%,
  sign-test p = 1.0000. That *weakens* the cross-seed generality of the lean rather than
  strengthening it — consistent with spec §7's own note that one of its four seeds was not
  significant alone and that the effect sits at the edge of what `precision 0.02` resolves.
- **420 rows on the two rays spec §7 could not measure**, which do lean: 266 negative, mean
  −0.121%. Real, and at operating points that had never been measured — but at the *same five
  seeds*, so these are new rays rather than independent draws, and their errors are correlated
  with the reproduced rows through the shared sample paths.

So the defensible statement is narrow and is the one that matters: **the bias at the tuned ray
is the same size as at the default ray**, measured pairwise at identical seeds. Whether the
lean itself generalizes across seeds is not advanced by this run.

γ conservation: 9 misses over 1080 checks (12 checked stations × 90 evaluations — two per
cell, since `sim_calls = 2`), i.e. 0.83% against the 5% a 95% interval implies. Every one of
the 9 is a conservation miss rather than some other quality flag, and no station in the whole
grid came back without a CI. `fj_pp` and `fj_sp` carry no throughput witness at all
(qsim-service#8), so 2 of 14 stations are never checked.

### What this still does not license

- **Three data points, not six.** Both fork-joins share γ and rates, so they land on the same
  ray in every workload (findings §9, unchanged).
- **One operating point.** λ = 0.9, p₁₁ = p₀ = 0.5, one budget multiple, one cost vector. The
  `r*` mechanism was swept over price ratio and budget analytically (findings §4, §5); the
  measured confirmation is at this point only.
- **A re-run at these seeds is not a replication.** The service is deterministic given a seed,
  so repeating the command reproduces every digit. Vary `--seeds` for an independent sample.

## Still open

**Nothing on issue #10.** Items 1, 2 and 3 are implemented, item 4 was a prohibition and was
honoured, and the simulated cross-check above is done and confirms all of it.

**The default is unchanged and remains `invariant-r`.** Flipping it is now a decision with
measured evidence behind it rather than a `t_ul` extrapolation — the tuned policy is confirmed
better in all three workloads and never worse — but it is still a separate decision about
advertised behaviour, and #14 (a tuned station stays mutated after a run, so a stored `Result`
misreports) is the thing to settle first if the default is to move.
