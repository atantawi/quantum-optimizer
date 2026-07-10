# Design Spec: Capacity Allocation Optimizer for a Network of Queues

Date: 2026-07-10
Status: Approved design — pending final review before implementation planning.

Companion recall doc: `docs/optimizer-brainstorm-summary.md` (problem statement and the
evolution of our understanding). This spec is the authoritative, implementation-facing
version.

---

## 1. Goal

Implement a Python optimizer that allocates resource capacities `S` to a network of queues
so as to solve the Section-5 problem `(OPT_RCA:FF)` of `docs/analysis.pdf`, using the
paper's fixed-point iteration (eqs 20–22, Steps 0–5).

The network is a collection of **stations**. Two station types:

- **Single-server queue** — M/M/1 mean response time.
- **Fork-join (FJ) queue** — two parallel servers, analyzed with the UL (upper–lower bound
  interpolation) approximation.

Arrival rates `γ` are fixed per-station constants. Only the capacity vector `S` changes
across iterations, until the optimal `S*` is reached. Per-station analyzers are independent
(no network routing is modeled).

## 2. Mathematical formulation

Optimization problem (over `N̂` stations; each FJ queue is one station):

```
min_S  Σ_i ω_i · ζ_i / (S_i·µ̂_i − γ_i)
s.t.   Σ_i c_i·S_i ≤ C ,   S_i·µ̂_i > γ_i    for all i ∈ [N̂]
```

Closed-form allocation for a fixed ζ vector (Theorem 5, eq 21):

```
S_i = γ_i/µ̂_i
    + ( C − Σ_j c_j·γ_j/µ̂_j ) · √(ω_i·ζ_i/(c_i·µ̂_i)) / Σ_j √(ω_j·ζ_j·c_j/µ̂_j)
```

ζ recovery by inverting the functional form (eq 22):

```
ζ_i = E[T_i] · (S_i·µ̂_i − γ_i)
```

Fixed-point loop (Steps 0–5): initialize `ζ⁽⁰⁾` → compute `S⁽ᵏ⁺¹⁾` (eq 21) → obtain
`E[T_i⁽ᵏ⁺¹⁾]` from the per-station analytic formula → recover `ζ⁽ᵏ⁺¹⁾` (eq 22) → stop when
`‖S⁽ᵏ⁺¹⁾ − S⁽ᵏ⁾‖∞ < ε`, else repeat. Cap iterations at `max_iter`.

## 3. Station model

### 3.1 Single-server station
- Fields: `gamma (γ)`, `mu (µ̂)`, `S`, `weight (ω)`, `c`.
- `sojourn_time(S) = 1 / (S·mu − gamma)`   (M/M/1; `ζ ≡ 1` exactly).
- `mu_bottleneck = mu`;  `alloc_cost = c`.

### 3.2 Fork-join station
- Fields: `gamma (γ)`, `mu (µ̂₁, the bottleneck/base rate)`, `S`, `weight (ω)`,
  `r ≥ 1`, `c1`, `c2`.
- Both servers receive the **same** capacity `S`. Effective rates:
  `m₁ = S·mu` (bottleneck), `m₂ = S·r·mu`. Their ratio is `r` for all `S`.
- `sojourn_time(S) = T_UL(gamma, m₁, m₂)` where
  ```
  α    = (γ/m₁ + γ/m₂) / 8
  T_UB = 1/(m₁−γ) + 1/(m₂−γ) − 1/(m₁+m₂−2γ)
  T_bot= max( 1/(m₁−γ), 1/(m₂−γ) )
  T_UL = (1−α)·T_UB + α·T_bot
  ```
  (`T_UL` copied from `~/Projects/fork-join` `forkjoin/analytical.py::mean_response_time`,
  with attribution; no runtime dependency on that repo.)
- `mu_bottleneck = mu (= µ̂₁)`;  `alloc_cost = c1 + c2`.

### 3.3 Shared interface (Station ABC)
- `sojourn_time(S) -> float`   — the analysis ("Analyzer" role, a method on the station).
- `mu_bottleneck -> float`     — `µ̂` used in eqs 20–22.
- `alloc_cost -> float`        — cost coefficient in the budget and eq 21.
- `zeta(S) -> float`           — `sojourn_time(S) · (S·mu_bottleneck − gamma)` (eq 22).

Per-station analysis is deliberately a method on the station (each station is its own
analyzer). This is the seam where a future `SimulationAnalyzer` could be substituted without
changing station data or the allocator/optimizer.

## 4. Components

- **Station (ABC)** + `SingleServerStation` + `ForkJoinStation` — as in §3.
- **Allocator** — `allocate(stations, C, zeta_vec) -> S_vec` implementing eq 21 using each
  station's `mu_bottleneck` and `alloc_cost`.
- **Optimizer** — owns the stations, budget `C`, tolerance `ε`, `max_iter`, and optional
  initial ζ. `run() -> Result` executes the loop. `Result` carries: `S*`, per-station
  `E[T_i]`, objective `Σ ω_i·E[T_i]`, iteration count, and `converged: bool`.

Default initial ζ: single-server → `1`, fork-join → `3/2` (both strictly positive; see §5).

## 5. Error handling & invariants

- **Budget feasibility:** require `C > Σ_j alloc_cost_j·γ_j/µ̂_j`; raise a clear error
  otherwise. This keeps eq 21's second term positive, guaranteeing `S_i·µ̂_i > γ_i`.
- **Positive initial ζ:** require `ζ_i⁽⁰⁾ > 0` for every station. A zero collapses eq 21 to
  the stability boundary (`S_i = γ_i/µ̂_i ⟹ E[T_i] = ∞`); all-zero makes eq 21 a `0/0`.
  With positive ζ⁽⁰⁾ and a feasible budget, `ζ_i⁽ᵏ⁺¹⁾ = E[T_i]·(S_i·µ̂_i − γ_i) > 0` holds
  throughout, so no station is driven to instability during iteration.
- **Stability guard:** `sojourn_time`/`zeta` raise if `S·µ̂ ≤ γ`.
- **Construction validation:** positive `gamma`, `mu`, costs, `weight`; `r ≥ 1`.
- **Non-convergence:** after `max_iter`, return `Result` with `converged=False` (no hang).

## 6. Package layout & dependencies

```
optimizer/
  station.py      # Station ABC + SingleServerStation + ForkJoinStation (+ T_UL)
  allocator.py    # eq 21
  optimizer.py    # loop + Result
  examples/…      # a sample mixed network
  tests/…
```

Core: pure Python + stdlib. Tests: `pytest`. `numpy` optional (likely unnecessary).

## 7. Acceptance criteria

Functional:
1. A network of only single-server stations converges, and every station reports `ζ = 1`.
2. A single-station network allocates the full budget: `S = C / c` (within tolerance).
3. eq 21 spends the full budget every iteration: `Σ_i alloc_cost_i·S_i ≈ C`.
4. A homogeneous FJ station (`r = 1`) matches the Nelson–Tantawi exact result, and `T_UL`
   equals the fork-join repo's `mean_response_time` for shared test inputs.
5. A mixed network (single-server + FJ, feasible budget) converges within `max_iter` to a
   stable `S*` (`S_i·µ̂_i > γ_i` for all i), and the reported objective equals
   `Σ ω_i·E[T_i]` at `S*`.

Guards / error paths:
6. Infeasible budget (`C ≤ Σ_j alloc_cost_j·γ_j/µ̂_j`) raises before iterating.
7. Non-positive initial ζ raises.
8. A starved / non-convergent run returns `converged=False` rather than hanging.
9. `sojourn_time` on an unstable capacity (`S·µ̂ ≤ γ`) raises.

## 8. Out of scope

- Network routing / arrival-rate propagation (γ fixed per station).
- Alternative FJ approximations beyond UL (the `sojourn_time` method is the seam if added).
- Simulation-based analysis (future `SimulationAnalyzer` at the same seam).
