# Choosing S₂ — what was implemented, and what it measured

Companion to [`findings.md`](findings.md), whose §10 ("If this were to be implemented") was
explicitly undecided. This records the decisions taken, the numbers the implementation
produces, and the six places measurement contradicted the plan. Issue #10; PRs #12 and #13.

Every figure here is reproducible from the committed test suite — the reference constants
live in `tests/test_example_qcsc.py` (`FINDINGS_SECTION_7`, `FINDINGS_BEST_RAY`, `TUNED`) and
`tests/test_forkjoin_policy.py` (`FINDINGS_SECTION_4`, `FINDINGS_Q7`), so they are executable
rather than transcribed.

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
current one — see the fifth and sixth corrections below for why both are correctness
requirements rather than hygiene, and why neither works without the other.

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

## Six corrections to the plan

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
the spend line's own floor `γ(β₁+β₂)` — which is §4b's observation that the paper's ray has the
optimal stability floor. `min_feasible_budget` is evaluated **once**, before any retune, so a
tuned station starting at `r` advertises the incumbent's floor and refuses budgets it can in
fact serve: every `C` in `(1.80, 3.15]` on a test station, all of which `equal-rate` completes.
Converged answers are bit-identical either way.

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
  optimizer* once the reset existed — it reported `2.10291` while `run()` served `1.91442` —
  and the README derives budgets from that helper. Fixed by `Station.min_spend`, a hook the
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

## Validation

- Suite 391 passed / 10 skipped.
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
- Each of the five guards this took is mutation-checked independently: policy-aware floor,
  reset placement (above `allocate`, below the preflight guards), `allocate`'s precondition,
  and eq 21's grouping in `min_spend`. The grouping needed a *single-server* fixture to be
  pinned at all — the fork-join cases go through the override, so reverting the base-class
  expression alone left the suite green.
- Iteration counts track the incumbent's at every damping — 120 against 119 at `θ = 0.1`, 20
  against 19 at `θ = 0.5`.
- `r*` over a budget sweep (125 multiples of each workload's own floor, `1.000001×` to 40×)
  independently reproduces §4's two limits: it is **1.0000 at the stability boundary** in every
  workload, and rises to 1.6369 in `balanced` against §4's asymptote of 1.633 and to 2.7027 in
  `quantum_dominant` against 2.693. `classical_dominant` is 1.0000 throughout, as β₁/β₂ = 1
  requires.
- The floor a tuned station advertises comes out 6.8400 / 5.8275 / 2.7900, which are exactly
  findings §7's *paper* floors (6.840 / 5.828 / 2.790) — the ones §4b identified as optimal.
  That is the starting-ray correction above, visible from the outside. It now holds for a
  reused station too, and that needed `min_spend`: with the helper reading the current ray,
  a station that had already run advertised its converged ray's floor instead, so this
  sentence was true only of freshly constructed stations.

## Still open

**The simulated cross-check**, and only that. Unchanged in urgency from findings §9: the 24.6%
`classical_dominant` gain is far outside known model error (~0.15% mean, ~1.1% worst row) and
is structural; a simulated pass sharpens only the marginal ~2% gains. Two things now bear on it:

- The tuned rays (1.447 / 2.316 / 1.000) are all **closer to homogeneity** than the incumbent
  `r = 4`, and `t_ul` is exact at `m₁ = m₂` and least validated at `r = 4`. So the tuned
  operating point sits where the approximation is *more* trustworthy than the baseline it is
  measured against — meaning a simulated pass that corrects `T` at `r = 4` downward would
  **shrink** the measured gain, not grow it.
- A simulated run's reported ray carries that run's sample path (see the noise correction
  above), so the cross-check should report `r_star` with its own interval rather than as a
  point.

It needs no library change: `sim_node` emits the effective ray at whatever `r_star` the station
is on.

**The default is unchanged and remains `invariant-r`.** Flipping it is a separate decision,
and findings §9's caveats on the two ~2% gains are the reason not to take it on this evidence.
