# Capacity Allocation Optimizer — Brainstorm Summary

_A running record of the problem statement, the design choices we made, and how our
understanding shifted during the discussion. This is a recall/context document, not the
formal spec (that comes next)._

Date: 2026-07-10

---

## 1. Problem statement

Build a Python optimizer that allocates **resource capacities** to a **network of queues**,
using the fixed-point iteration of **Section 5** of the analysis paper
(`docs/analysis.pdf`, "Optimization and Performance Analysis of Resource Allocation in
Quantum-Centric Supercomputing Environments").

The network is a collection of **stations** (a.k.a. nodes). Two station types:

- **Single-server queue** — a G/G/1 queue analyzed with the Kingman / Allen–Cunneen
  mean-value approximation (M/M/1 and M/D/1 are presets). See §3.5.
- **Fork-join (FJ) queue** — a pair of parallel servers; analyzed with a closed-form
  approximation lifted from the local `~/Projects/fork-join` repo.

We solve the paper's problem `(OPT_RCA:FF)`: minimize the sum of weighted expected
sojourn times subject to a budget on cost-weighted capacities and per-station stability:

```
min_S  Σ_i ω_i · ζ_i / (S_i·µ̂_i − γ_i)
s.t.   Σ_i c_i·S_i ≤ C ,   S_i·µ̂_i > γ_i   for all i ∈ [N̂]
```

where the closed-form optimum for a fixed ζ vector is (Theorem 5 / eq 20–21):

```
S_i = γ_i/µ̂_i + ( C − Σ_j c_j·γ_j/µ̂_j ) · √(ω_i·ζ_i/(c_i·µ̂_i)) / Σ_j √(ω_j·ζ_j·c_j/µ̂_j)
```

and ζ is recovered by inverting the functional form (eq 22):

```
ζ_i = E[T_i] · (S_i·µ̂_i − γ_i)
```

### The fixed-point loop (paper Steps 0–5)

0. Initialize `ζ⁽⁰⁾` (one entry per station), `k = 0`.
1. Compute `S⁽ᵏ⁺¹⁾` from `ζ⁽ᵏ⁾` via eq 21.
2. Obtain `E[T_i⁽ᵏ⁺¹⁾]` for each station under `S⁽ᵏ⁺¹⁾` (paper simulates; **we use the
   per-station analytic formula instead**).
3. Recover `ζ_i⁽ᵏ⁺¹⁾` via eq 22.
4. Converge if `‖S⁽ᵏ⁺¹⁾ − S⁽ᵏ⁾‖∞ < ε`; else `k++` and repeat from step 1.
5. Output `S*` and `E[T_i]`.

---

## 2. Scope decisions

| Decision | Choice | Rationale |
|---|---|---|
| Paper coverage | Section 5 only (self-contained optimization) | Rest of paper not needed for the optimizer. |
| Arrival rates `γ` | **Fixed per-station constants**; no network routing modeled | Analyzers are independent per station. Only `S` changes across iterations. Paper's full routing model (eqs 23–27) is out of scope. |
| Step 2 evaluation | **Analytic sojourn time per station**, replacing the paper's network simulation | M/M/1 for single-server, UL approximation for FJ. |
| FJ approximation | **UL** (upper–lower bound interpolation), `mean_response_time` in the fork-join repo | See §3. Lifted (copied) into this project with attribution — no runtime dependency on the fork-join repo. |

---

## 3. Changes in understanding along the way

These are the points where our initial reading shifted:

### 3.1 FJ approximation: UB → UL
Initially we planned to lift the **UB** (independent upper bound). The user corrected this
to the **UL** approximation — the convex blend of the independent upper bound and the
bottleneck lower bound:

```
α    = (λ/µ₁ + λ/µ₂) / 8
T_UB = 1/(µ₁−λ) + 1/(µ₂−λ) − 1/(µ₁+µ₂−2λ)
T_bot= max( 1/(µ₁−λ), 1/(µ₂−λ) )
T_UL = (1−α)·T_UB + α·T_bot
```

**Why it matters:** with the pure UB (and the corrected equal-capacity FJ model), ζ turns
out constant, so the loop would converge in one iteration. Under **UL**, ζ genuinely varies
with load (e.g. `ζ = 3/2 − ρ/8` at balanced rates), so the fixed-point iteration does real
work.

### 3.2 The S₂ interpretation (paper vs. intent) — one ray of a family
The paper ties the non-bottleneck server's capacity as `S₂ = S₁/r`, which forces
`S₂·µ̂₂ = S₁·µ̂₁` — it **equalizes the two effective service rates** and destroys the ratio
`r`. That was flagged as a mistake.

**Correction adopted:** _both servers of a FJ station receive the same capacity `S`._ Then
effective rates are `m₁ = S·µ̂₁` and `m₂ = S·r·µ̂₁`, and their ratio stays `r` for any `S`,
independent of the allocation. Knock-on effects:

- Functional form stays `E[T_FJ] = ζ/(S·mu − γ)`, where the station's `mu` field is the
  **slower** server's rate (the smaller of the two); the faster server's rate is `r·mu`.
  (The paper calls the slower one the "bottleneck µ̂₁"; we just call it `mu`.)
- FJ budget spend is `S·c₁ + S·c₂`, so the allocator's cost coefficient is
  **`c_FJ = c₁ + c₂`** (not the paper's `c₁ + c₂/r`).
- Stability condition: `S·mu > γ` (slower server binds). Since `m₁ < m₂`, this implies
  `m₂ > γ` too.

**Update (2026-08-31, issue #10).** "Wrong" was too strong, and the correction is not free.
Both rules are members of one family — the ray `m₂ = r*·m₁`, priced `c₁ + c₂·r*/r` — with the
paper at `r* = 1` and the correction above at `r* = r`. So the paper's `c₁ + c₂/r` is the
*exact* cost of its own S₂ rule rather than a fudge, and neither ray dominates: on the QCSC
network at a shared budget the paper's ray is optimal in `classical_dominant` (worth 24.55%)
and 5.47% worse in `quantum_dominant`. `ForkJoinStation` now takes `r_star`, defaulting to
`r`, so the correction adopted here remains the behaviour and both policies are reachable.
It is also **solvable**: `r_star = R_STAR_TUNED` re-solves the local optimality condition
at the station's own spend on every optimizer iteration — a fixed point nested inside
eqs 21/22 — and reaches what a grid sweep of `r_star` finds in all three workloads,
recovering the paper's `r* = 1` exactly in `classical_dominant`, where it is optimal.
Stability still reads `S·mu > γ`, but `mu` is the *effectively* slower rate `µ̂₁·min(1, r*)`,
which is what carries the implication above through `r* < 1`. Evidence and the local
optimality condition: [`docs/forkjoin-s2-policy/findings.md`](forkjoin-s2-policy/findings.md).

### 3.3 FJ cost representation
Decision: a FJ station stores **`c₁` and `c₂` separately**; the allocator forms
`c_FJ` internally. (The alternative — a single combined `c` — was rejected in favor of
faithfully representing the two physical servers.) Since issue #10 that combination is
`c₁ + c₂·r*/r`, the exact cost of the two capacities the station's ray buys, which is
`c₁ + c₂` at the default `r* = r`; storing the two costs separately is what makes any
other ray priceable at all.

### 3.4 `ζ = 0` instability guard
An initial `ζ_i⁽⁰⁾ = 0` collapses eq 21 to `S_i = γ_i/µ̂_i` (the stability boundary),
giving `E[T_i] = 1/0 → ∞`; and all-zero ζ makes eq 21 a `0/0`. Therefore:

- **Require `ζ⁽⁰⁾ > 0` strictly** for every station (defaults: M/M/1 → 1, FJ → 3/2).
- With a **feasible budget** `C > Σ_j c_j·γ_j/µ̂_j`, eq 21 keeps `S_i·µ̂_i > γ_i`, so
  `ζ_i⁽ᵏ⁺¹⁾ = E[T_i]·(S_i·µ̂_i − γ_i) > 0` throughout — no station is ever driven to the
  boundary during iteration.

### 3.5 Single-server generalized to G/G/1
Rather than hard-coding M/M/1, the single-server type is an abstract `SingleServerStation`
with a concrete **`GG1Station`** using the Kingman / Allen–Cunneen mean-value approximation,
parameterized by the coefficients of variation of interarrival (`cov_a`) and service
(`cov_s`) times:

```
ζ = 1 − ρ·(1 − (cov_a² + cov_s²)/2),   ρ = γ/(S·mu)
```

M/M/1 (`cov_a=1, cov_s=1` → ζ ≡ 1) and M/D/1 (`cov_a=1, cov_s=0` → ζ = 1 − ρ/2) are presets;
the approximation is exact for any M/G/1 (Poisson arrivals). This confirmed that ζ is **not**
generally 1 — M/D/1 gives a load-dependent ζ — so the loop treats ζ generically (the earlier
`ζ = 1` was only an M/M/1 sanity check, never an assumption).

**Scope note — variability propagation (deferred).** `cov_a` is taken as a fixed per-station
input. In a real network, an upstream station's *departure* variability drives its
downstream neighbors' *arrival* variability; analyzing stations independently does not
capture that coupling. Doing so faithfully requires **simulation** of the whole network, not
independent per-station closed forms. This is beyond the current scope — treated as a current
approximation and a subject for future incorporation (same seam as the future
`SimulationAnalyzer`). Recorded in the spec §8 and the README.

---

## 4. Agreed architecture

Three roles; per-station analysis is a **method on the station** (each station is its own
analyzer — the seam where a future `SimulationAnalyzer` could be swapped in).

```
Station (ABC)                      # queueing entity + its math
├─ fields: gamma, mu, S, weight
├─ sojourn_time(S) -> float        # THE analysis ("Analyzer" role)
├─ alloc_cost    -> float          # cost coefficient in budget / eq 21
└─ zeta(S)       -> float          # E[T]·(S·mu − γ)         (eq 22)

SingleServerStation(Station)       # ABC for one-server queues; extra field: c; alloc_cost=c
   └─ GG1Station                   # G/G/1 (Kingman/Allen-Cunneen); extra: cov_a, cov_s
        sojourn_time(S) via E[T] = (1/µ)[1 + ((cov_a²+cov_s²)/2)·ρ/(1−ρ)], µ=S·mu, ρ=γ/µ
        ζ = 1 − ρ·(1 − (cov_a²+cov_s²)/2)
        presets: mm1() → (1,1), ζ≡1 ;  md1() → (1,0), ζ = 1 − ρ/2   (both exact)

ForkJoinStation(Station)           # extra fields: r, c1, c2
   sojourn_time(S) = T_UL(gamma, S·mu, S·r·mu)     # mu = slower server
   alloc_cost = c1 + c2

Allocator
   allocate(stations, C, zeta_vec) -> S_vec        # eq 21

Optimizer
   run() -> Result   # init ζ → allocate → E[T] per station → new ζ
                     #        → converge test (‖ΔS‖∞ < ε) → repeat
   Result: S*, E[T] per station, objective Σ ω_i·E[T_i], iterations, converged
```

Sanity checks embedded in the model:
- M/M/1 has `ζ ≡ 1` exactly (functional form is exact).
- A single-station network spends the whole budget: `S = C/c`.
- eq 21 always spends the full budget: `Σ c_i·S_i = C` at every step.

---

## 5. Error handling & edge cases

- **Budget feasibility:** require `C > Σ_j c_j·γ_j/µ̂_j`; clear error otherwise.
- **Stability guard** in `sojourn_time`/`zeta`: error if `S·µ̂ ≤ γ`.
- **Positive initial ζ** required (see §3.4).
- **Non-convergence:** return `Result` with `converged=False` after `max_iter`.
- **Construction validation:** `r ≥ 1`, positive rates/costs/weights.

---

## 6. Layout & dependencies

```
optimizer/
  station.py      # Station ABC + SingleServerStation + ForkJoinStation (+ T_UL)
  allocator.py    # eq 21
  optimizer.py    # loop + Result
  examples/…      # a sample mixed network
  tests/…
```

Core: pure Python + stdlib. Tests: `pytest`. `numpy` optional (likely unnecessary).
`T_UL` is copied from the fork-join repo's `forkjoin/analytical.py::mean_response_time`
with attribution — no runtime coupling.

---

## 7. Open / next steps

- Write the formal design spec to `docs/superpowers/specs/YYYY-MM-DD-optimizer-design.md`.
- User review of the spec.
- Produce the implementation plan (writing-plans skill).
