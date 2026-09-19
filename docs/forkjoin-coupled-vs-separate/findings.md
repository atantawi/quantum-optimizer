# Coupled vs separate fork-join `r*`

**Question.** qopt solves each fork-join station's `r*` on its own, at the spend eq 21 handed it.
Should the fork-join stations instead be solved as one coupled problem — a vector of `r*`
minimizing a weighted sum of their sojourn times under a combined budget `C_FJ`, obtained by
subtracting the single-server stations' spend from the total?

**Answer: no, and the reason is worth more than the answer.** The two formulations have *identical*
ray conditions; they can differ only in how spend is split. On every instance tested the coupled
fork-join block is worth at most **0.027%** of the objective, and it structurally cannot reach the
larger effect — **0.043%** — because that one is fork-join against the *single-server* stations,
which is precisely what fixing `C_FJ` freezes.

Equation numbers follow [`../paper-map.md`](../paper-map.md). Every number below is from
[`probe-output.txt`](probe-output.txt), reproducible with
`python docs/forkjoin-coupled-vs-separate/probe.py` (deterministic, no simulation service).

**Status: analysis only.** Nothing here proposes a code change. The follow-on proposal it
motivates is [`../slope-calibrated-zeta/`](../slope-calibrated-zeta/findings.md).

## 1. The three formulations

| | what it is |
|---|---|
| **(1)** | what qopt does today: eq 21 splits spend across *all* stations, then each fork-join station solves its own ray at the spend it received |
| **(2)** | the proposal: `C_FJ = C −` (single-server spend chosen by (1)), then all fork-join stations solved as one coupled problem under `C_FJ` |
| **(3)** | reference: one coupled problem over *every* station |

All three minimize the same objective under the same total budget.

## 2. `r*` barely depends on the budget, and not at all on `B` directly

The starting observation was that `r*` seems insensitive to `C_FJ`. Refined: `r*` *does* depend on
its station's own spend, materially so near the stability boundary — but `B`, the weights, and the
other stations never enter the condition that determines it. They reach `r*` only by setting one
scalar, that station's spend.

Even that dependence is structurally thin. In *slack* space `x_k = m_k − γ`, both `t_ub` and
`t_bot` are homogeneous of degree −1, and `α = (γ/m₁ + γ/m₂)/8` is the only term in `t_ul` that is
not. With `α` frozen the stationarity condition is homogeneous of degree 0 — it pins a
**direction**, not a magnitude — and `u = x₂/x₁` solves a closed form in the price ratio
`p = β₁/β₂` alone:

```
p/u² − 1 = (p−1)/(1+u)²
```

```
 spend/floor    r* = m2/m1     u = x2/x1      alpha     rho1
       1.001    1.00137876    2.50195279   0.249599   0.9991
         1.5    1.50767039    2.61975238   0.142745   0.6866
          10    2.58494574    2.77948472   0.018952   0.1093
       1e+08    2.80223707    2.80223709   0.000000   0.0000
```

`u*(16) = 2.8022370912` from the closed form, against `2.80223707` as the measured spend→∞ limit:
the same to 8 digits. So `r* = (γ+x₂)/(γ+x₁)` runs from 1 at the boundary up to `u*` and saturates
— `u` moves only 2.502 → 2.802 across eight orders of magnitude of spend, all of it `α` drift.

## 3. Why the coupled KKT collapses onto the per-station condition

The coupled problem `min Σ wᵢTᵢ(m_{i1}, m_{i2})` s.t. `Σ(β_{i1}m_{i1} + β_{i2}m_{i2}) = B` has
stationarity `wᵢ ∂Tᵢ/∂m_{ik} = −ν β_{ik}`, two equations per station. Take the ratio *within* a
station:

```
(∂Tᵢ/∂m_{i1}) / (∂Tᵢ/∂m_{i2}) = β_{i1}/β_{i2}
```

**ν cancels, and with it `B`, `wᵢ`, and every other station.** That is verbatim
`_min_on_spend_line`'s condition. The only surviving global information is the scalar `ν`, which
equalizes `wᵢ·|dTᵢ/d(spend)|` across stations.

Checked without assuming it — a projected-gradient solve in the full 2n-dimensional space, no
per-station reduction used, so the check cannot beg the question:

```
 stn      dT/dm1 / dT/dm2    beta1/beta2    rel err    w*(-dT/dm1)/beta1
   0      15.999999901455    16.00000000   6.16e-09       0.082732559752
   1       0.666666666041     0.66666667   9.39e-10       0.082732560186
   2       0.006666705163     0.00666667   5.77e-06       0.082733039311
```

Three stations with weights 1 / 2 / 0.3 and price ratios 16 / 0.667 / 0.00667: each ray obeys its
own price ratio, under one common `ν`.

**Consequence.** A station's weight and the total budget decide how much *money* it gets, never the
*shape* of its hardware. Formulation (2) cannot change any ray except by changing a spend — so (1)
is already exact in the ray dimension, and approximate only in the split.

(1) is therefore not "uncoupled" at all: it is a block-coordinate method on (2)'s KKT, with
coupling running through eq 21's shared slack and through `alloc_cost = c₁ + c₂r*/r`, which feeds
each ray back into the next iteration's prices.

## 4. What the split is actually worth

Negative means better than (1).

```
--- 2 fork-join + 2 M/M/1 ---
 C/floor         obj (1)     (2)-(1)     (3)-(1)     (3)-(2)  max rho  r* (1)vs(3)
   1.001    8452.8951140   -0.02701%   -0.03954%   -0.01253%   0.9992      0.0027%
     1.5      17.1478173   -0.00484%   -0.02044%   -0.01560%   0.7301      0.3594%
     100       0.0883721   -0.00000%   -0.00001%   -0.00001%   0.0139      0.0003%

--- 3 fork-join + 1 M/M/1, weights 0.02..50, price ratios 16 / 1 / 0.022 ---
   1.001    6625.5319387   -0.02455%   -0.04338%   -0.01884%   0.9993      0.0037%
     1.5      13.4860549   -0.01114%   -0.02590%   -0.01476%   0.7387      0.3501%

--- 2 near-identical fork-join, no single-server station ---
   1.001    5193.6808915   -0.00000%   -0.00000%    0.00000%   0.9991      0.0000%
     1.5      10.4992860   -0.00000%   -0.00000%    0.00000%   0.6889      0.0023%

--- ONE fork-join: (2) is vacuous by construction ---
   1.001   10894.1139483    0.00000%   -0.04289%   -0.04289%   0.9994      0.0021%
     1.5      21.8558761    0.00000%   -0.03274%   -0.03274%   0.7791      0.4677%
       2      10.9435478    0.00000%   -0.02407%   -0.02407%   0.6392      0.4823%
```

Four things fall out:

1. **(2) is worth at most 0.027%**, and only near the stability boundary; it decays to nothing as
   the budget loosens.
2. **(2) captures only part of what is available, and how much is unreliable** — between `8.7%` and
   `68.3%` of the `(3)−(1)` gap across the two multi-fork-join instances, falling steeply with the
   budget in the baseline case (68% → 9%) while holding near half in the adversarial one. The
   remainder is the fork-join-vs-single-server split, which (2) froze.
3. **With symmetric fork-join stations and nothing else to share with, all three coincide** — to
   5e-8. There is no fork-join-to-fork-join misallocation to recover when the stations are alike.
4. **The decisive case is a single fork-join station.** (2) is then empty by construction, yet
   `(3)−(1)` is at its *largest*, `−0.04289%`. So the coupling that matters is not fork-join to
   fork-join at all.

The rays themselves differ between (1) and (3) by at most **0.48%**, which is the same point from
the other side: even a mispriced spend gives very nearly the right ray.

### The structural objection to (2)

"Subtract the single-server budget, then optimize the fork-join block" is *itself* a decomposition.
It is correct only if that pre-split already equalized marginal value per dollar across the two
blocks — which is exactly what (2) was introduced to fix. Fixing `C_FJ` exogenously imposes a wrong
`ν` on the fork-join block, so (2) relocates the approximation rather than removing it.

If the gain is wanted, the move is not to couple the fork-join stations but to **couple
everything**: one `ν` water-filled over all stations against the true `dTᵢ/d(spend)`. That is (3),
it is still a scalar bisection, and each per-station ray solve is unchanged.

## 5. Why (1) and (3) differ at all

eq 21 splits spend using the surrogate `T ≈ ζ/(Sμ−γ)`, whose implied marginal per dollar is
`w·T·μ/(c·(Sμ−γ))`. Only *relative* marginals decide a split, so eq 21 is exact iff

```
φᵢ = [true wᵢ·|dTᵢ/d(spend)|] / [surrogate marginal]
```

is the same for every station. Measured at (1)'s converged point:

```
 C/floor       FJ-A phi       FJ-B phi       SS-1 phi       SS-2 phi    spread    (3)-(1)
   1.001    0.918854256    0.993930512    1.000000000    1.000000000   8.831%  -0.03954%
     1.5    0.931922812    0.963462472    1.000000000    1.000000000   7.305%  -0.02044%
     100    0.998470118    0.998950075    1.000000000    1.000000000   0.153%  -0.00001%
```

`φ ≡ 1` **exactly** for M/M/1 — `ζ = 1` identically there, and eq 21 *is* the exact KKT solution
for an all-M/M/1 network. `φ_FJ < 1` means eq 21 overprices fork-join money and overpays those
stations, and (3) moves it back toward the single-server stations.

Note the second-order washout: an **8.8% error in the split prices costs 0.04% of the objective**,
because the objective is flat near its optimum. That is why (1) is nearly optimal despite
mispricing — and why the lever that matters is fixing `φ`, not restructuring which stations are
solved together.

## 6. Conclusion

Do not couple the fork-join stations. (1) is already exact in the ray dimension; the residual loss
is a mispriced split, it is second order, and a combined `C_FJ` addresses the smaller half of it
while re-introducing the same class of error at the block boundary.

The mispricing itself is worth fixing, and `φ` is the handle. That is
[`../slope-calibrated-zeta/`](../slope-calibrated-zeta/findings.md).

## Caveats

- Everything is measured against the **analytic** model. `t_ul` is an approximation, so "the global
  optimum" means the optimum of the model qopt solves.
- The reference optimum comes from water-filling plus a pairwise local-descent audit. Both are
  local methods; they agree with each other at every row, but neither establishes global
  uniqueness.
- The gaps here are far below the noise floor on the simulated path, so none of this is measurable
  end-to-end today.
- `(2)−(1)` is reported as exactly `0.00000%` for the single-fork-join instance because (2) is
  *defined* to collapse onto (1) there — that column is not a measurement in that block.
