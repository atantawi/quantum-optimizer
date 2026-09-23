# Slope-calibrated ζ

**Status: implemented** (2026-09-22) as a per-station opt-in; the default is still eq 22's level
calibration, and the paper is unchanged. What follows is the argument as it was made, in its own
tense: it argues for *replacing* the default, which is not what was built. Read its "would" as
conditional on a decision that landed narrower -- an opt-in, not a swap.

Equation numbers follow [`../paper-map.md`](../paper-map.md): `eq 21` (the allocation rule) and
`eq 22` (the ζ inversion) mean `docs/analysis.pdf`, the snapshot committed in `37a3a11`. Every
number below is from
[`probe-output.txt`](probe-output.txt), reproducible with `python docs/slope-calibrated-zeta/probe.py`
(30 s, deterministic, no simulation service). That claim is enforced, not just stated:
[`../quotes-selfcheck.py`](../quotes-selfcheck.py) requires every fenced quote below to match that
file exactly.

## 1. Claim

eq 21's optimality condition is a statement about the **slope** of each station's sojourn-time
curve. eq 22 calibrates ζ to match that curve's **level**. In the form `T = ζ/x` the two are
rigidly tied — `slope = −level·μ/x` — so matching one necessarily misses the other.

Calibrate the slope instead:

```
ζ_slope = |dT/dS| · x²/μ  =  φ · (T·x),        φ = |dT/dS| · x / (μ·T),   x = S·μ − γ
```

and the loop's fixed point stops being an approximation of the coupled optimum and becomes the
coupled optimum, exactly. Measured: the slope-calibrated run matches an independently computed
global optimum **to machine precision at all 27 rows tested**, with iteration counts essentially
unchanged.

`φ ≡ 1` for M/M/1 — which is why this went unnoticed. It is invisible on the station type most of
the suite uses.

**This is not a fork-join issue.** Every station whose ζ depends on S has it, and the fork-join is
the mildest case (§5).

## 2. Background: where the loss comes from

This came out of asking whether the fork-join stations' `r*` should be solved as one coupled
problem rather than per-station. The answer was no, and the reason points here. That analysis, with
its evidence, is [`../forkjoin-coupled-vs-separate/`](../forkjoin-coupled-vs-separate/findings.md);
it is not repeated here. What this document needs from it is one result:

> The coupled problem's within-station stationarity ratio is `(∂Tᵢ/∂m_{i1})/(∂Tᵢ/∂m_{i2}) =
> β_{i1}/β_{i2}`, in which the multiplier `ν`, the budget, and the weights all cancel. So coupling
> stations cannot change any ray — only a spend.

Which means the loop is **already exact in the ray dimension, and approximate only in the spend
split**. The split is eq 21's job, and eq 22 feeds it with the wrong slope. That is this document.

That analysis also measures the resulting loss: at most `0.043%`, with the dominant part being the
fork-join-vs-single-server split rather than anything between fork-join stations. `φ` — the same
factor defined in §1 here — is what prices it, and its spread across stations reaches `8.8%`.

## 3. Derivation

eq 21 solves `min Σ wᵢ ζᵢ/(Sᵢμᵢ − γᵢ)` s.t. `Σ cᵢSᵢ = C`, whose stationarity is

```
wᵢ ζᵢ μᵢ / xᵢ² = ν cᵢ
```

Substitute `ζᵢ = φᵢ Tᵢ xᵢ = |dTᵢ/dSᵢ| xᵢ²/μᵢ`. The `xᵢ²/μᵢ` cancels against `μᵢ/xᵢ²`:

```
wᵢ |dTᵢ/dSᵢ| = ν cᵢ        ⟺        wᵢ |dTᵢ/d(spendᵢ)| = ν
```

which is exactly the condition that defines the coupled optimum. Note `Tᵢ` has cancelled: the
slope-calibrated ζ does not contain the sojourn time at all. §6 is about why that matters.

For contrast, eq 22's `ζ = T·x` gives `wᵢ Tᵢ μᵢ/xᵢ = ν cᵢ`, which agrees only where
`|dTᵢ/dSᵢ| = Tᵢ μᵢ/xᵢ`, i.e. where `φᵢ = 1`.

### Where that form comes from

`ζ_slope` is a **definition**, but a forced one. Both calibrations pick one member of the same
one-parameter family `T̂ = ζ/x`, and the question is only which property of the true curve to
preserve with the single degree of freedom available.

What settles it is that **eq 21 reads that family only through its derivative.** The stationarity
condition above can be rewritten `wᵢ·|dT̂ᵢ/dSᵢ| = ν cᵢ`, because `ζᵢμᵢ/xᵢ² = |dT̂ᵢ/dSᵢ|` — and
nothing in the loop consumes the surrogate's *value*: the objective qopt reports is assembled from
the analyzer's `E[T]`, never from `ζ/x`. So ask the surrogate's slope to be the true slope:

```
ζμ/x² = |dT/dS|        ⟹        ζ_slope = |dT/dS| · x²/μ
```

That is the entire derivation. Level calibration spends the same free parameter on the one
quantity the allocator never looks at.

Since `x = Sμ − γ` makes `dT/dS = μ·dT/dx`, the `μ` cancels and the form simplifies:

```
ζ_slope = x² · |dT/dx|
```

which is the version worth remembering — `μ` was an artifact of writing things in `S` rather than in
spare capacity.

### What it means

`x = Sμ − γ` is **spare capacity**, an excess service rate. The surrogate `T = ζ/x` therefore reads
a sojourn time as `ζ` units of reciprocal spare capacity, making **ζ dimensionless** — and an M/M/1
station has `T = 1/x` exactly, so `ζ ≡ 1`. ζ measures how far a station departs from the M/M/1
shape:

| | asks | answers |
|---|---|---|
| `ζ_level = T·x` | what is `E[T]` *now*? | current congestion, in M/M/1 units |
| `ζ_slope = x²·\|dT/dx\|` | how fast does `E[T]` *fall* as spare capacity is bought? | marginal return on capacity, in M/M/1 units |

And their ratio is not a fudge factor: **φ is the elasticity of `E[T]` with respect to spare
capacity**, equivalently the local power-law exponent in `T ∼ x^(−φ)`.

```
φ = ζ_slope/ζ_level = |dT/dx|·x/T = −d log T / d log x
```

Probe §1 checks all three spellings against each other, worst disagreement `6.80e-09`:

```
station          rho  zeta_lvl=Tx  zeta_slope  x^2|dT/dx|       phi  |dT/dx|x/T  -dlogT/dlogx
M/M/1            0.6     1.000000    1.000000    1.000000  1.000000    1.000000      1.000000
M/D/1            0.6     0.700000    0.580000    0.580000  0.828571    0.828571      0.828571
G/G/1 cov=2      0.6     2.800000    3.520000    3.520000  1.257143    1.257143      1.257143
G/G/1 cov=5      0.3     8.200000   13.240000   13.240000  1.614634    1.614634      1.614634
D/D/1 cov=0      0.9     0.100000    0.010000    0.010000  0.100000    0.100000      0.100000
fork-join p=16   0.6     1.011223    0.999501    0.999501  0.988408    0.988408      0.988408
```

So **eq 21 is a water-filling rule built on the premise that every station's delay curve is a
rectangular hyperbola in spare capacity, `T ∝ 1/x`.** Where that premise fails, φ is the true local
exponent, and `ζ ← φ·ζ` hands eq 21 the real marginal return instead of the assumed one. `D/D/1` is
the clearest case: `φ = 1 − ρ` **exactly**, because `E[T] = 1/m` has no queueing term, so at high
load spare capacity barely moves it. §7 notes that `φ → 0` there and why; the elasticity reading is
what pins it to a closed form.

### Why matching the slope makes the fixed point exact

Geometrically there is a family of hyperbolas `ζ/x` and one point at which to spend the parameter.
`ζ_level` picks the hyperbola **through** `(x, T)`; `ζ_slope` picks the one **tangent in slope** at
`x`. They are different hyperbolas unless `φ = 1` — `ζ_slope` does not reproduce `E[T]` at all, it
gives `T̂ = φT`.

Giving that up costs nothing, because at convergence the calibration point and the argmin are the
same point:

- calibration ⟹ the surrogate's slope equals the true slope **at `S*`**
- argmin ⟹ the surrogate's slope satisfies eq 21's KKT **at `S*`**
- compose ⟹ the *true* slope satisfies the KKT at `S*`, which is global optimality

Level calibration gets the right value at `S*` and a wrong slope there, so the KKT is enforced on
the wrong quantity. That is the whole difference between the two rows of §4.

## 4. Payoff

`(1)` as shipped, `(1s)` slope-calibrated, `(3)` the coupled optimum over all stations —
water-filled on true marginals, then audited by pairwise local descent on the exact objective so a
bug in one cannot hide in the other. The audit runs in **both** directions: descent from the
water-filled answer must find nothing better, *and* descent from a 5% perturbation must return to
the same objective. The second direction is what makes the first meaningful — a descent that did
nothing would satisfy the first on its own.

That those two assertions can actually fail is not taken on trust:
[`../audit-selfcheck.py`](../audit-selfcheck.py) breaks each premise in turn and requires the
matching assertion to fire with the matching message. The first version of this audit was vacuous,
so the check exists.

```
--- 2 fork-join + 2 M/M/1  (phi moves only on the FJ pair) ---
 C/floor       (1) level      (1s) slope       (3) exact     (1s)-(3)    (1)-(3)   it  it_s
    1.01   845.653557502   845.326602738   845.326602738    0.000000%    0.0387%    6     7
     1.5    17.147817304    17.144312750    17.144312750    0.000000%    0.0204%    8     8
      20     0.459984473     0.459983404     0.459983404    0.000000%    0.0002%    6     6

--- 1 fork-join + M/D/1 + G/G/1(cov 2) + M/M/1 ---
    1.01   672.781044226   672.409070110   672.409070151   -0.000000%    0.0553%    6     7
     1.2    32.456697879    32.343754976    32.343754976    0.000000%    0.3492%    9     7
     1.5    12.821589655    12.755896143    12.755896143   -0.000000%    0.5150%   11    10
       2     6.396010744     6.364185455     6.364185455   -0.000000%    0.5001%   11    11
```

The gain tracks how far `φ` is from 1, not the station type: **0.0002–0.039%** where only
fork-join stations deviate, **up to 0.515%** once the single-server stations are not M/M/1, and
**1.59%** on the §7 stress network. It is largest at moderate load and vanishes as the budget
loosens.

## 5. φ, and where the existing derivative helper cannot be used

Both closed forms are checked against central differences (§1 of the probe, relative error ≤
`6.7e-09` everywhere):

```
φ_GG1  = (x² + kγ(2m−γ)) / (m(x + kγ)),   k = (cov_a² + cov_s²)/2,   ≡ 1 at k=1,  = 1−ρ at k=0
dT/dS_FJ = (α/S)(t_ub − t_bot) + (1−α)·t_ub' + α·t_bot'
```

| station | φ at ρ=0.95 / 0.67 / 0.25 |
|---|---|
| M/M/1 | 1.000000 / 1.000000 / 1.000000 |
| M/D/1 | 0.954762 / 0.833759 / 0.892857 |
| G/G/1 cov=2 | 1.037013 / 1.220365 / 1.321429 |
| G/G/1 cov=5 | 1.047899 / 1.310679 / **1.642857** |
| fork-join p=16 | 0.999581 / 0.990545 / 0.987673 |
| fork-join r*=1 | 0.995701 / 0.980485 / 0.984043 |

The fork-join barely moves; `M/D/1` and high-cov `G/G/1` are where the money is.

The fork-join derivative is the **radial** one, taken along the station's current ray, and it is
deliberately *not* built from `_dt_dm1`. Along a ray both rates scale together, so `min(m1, m2)`
never switches — `_anchor` pairs `mu` with the slower server and keeps `r ≥ 1`, so `t_bot = 1/x1`
with no `max()` to cross, and the derivative is smooth. `_dt_dm1`, by contrast, takes the "m1 is
the non-bottleneck" branch at `m1 == m2` and drops the `α·t_bot` term entirely. That is harmless
inside `_min_on_spend_line` — its docstring argues correctly that the kink is measure-zero there
and that its jump runs with the bisection — but it is wrong for pricing `dT/d(spend)`, and
`r* = 1` is exactly where tight budgets and every `β₁ = β₂` station sit:

```
station             spend/floor   radial (correct)    via _dt_dm1       rel
FJ beta1==beta2            1.05          67.915232      56.157442     17.3%
FJ beta1==beta2             1.5         0.68587106     0.60356653     12.0%
FJ beta1==beta2               4         0.01982596    0.018968621      4.3%
```

This is not hypothetical: building the reference optimum with `_dt_dm1` made the *exact* optimum
appear 0.1% worse than the shipped loop, which is how the defect was found.

**A load-bearing subtlety.** eq 21 works in S-space; the coupled condition is in spend-space. They
agree because at an optimal ray `∇T ∥ β`, so every direction with the same spend increment gives
the same first-order change in `T` — including the radial one. Off the optimal ray it fails, so
this is a property of the fixed point, not an identity (§2 of the probe: agreement to `3.3e-08` or
better on the optimal ray; `20.4%` and `12.3%` disagreement at `r* = 1.0` and `4.0` where the
optimum is `2.274413`). Consequence: eq 21 needs no new variable, but `φ` is only meaningful once
the ray has settled.

## 6. The simulated path — the real design constraint

`T` cancels out of `ζ_slope` (§3). So a purely slope-calibrated ζ built from an analytic
derivative is *entirely analytic*, and **the simulator would stop influencing the allocation at
all** — which is what eq 22 exists to prevent. The proposal is therefore the hybrid:

```
ζ = φ_analytic · T_measured · x        # measured level, analytic shape
```

The obvious alternative, a secant slope from consecutive loop iterates, does **not** work: the
iterates converge, so it degenerates to 0/0 over a shrinking interval while differencing two
CRN-noisy numbers.

So the question is whether an analytic `φ` is safe when the analytic model's shape is imperfect.
Blending `ζ = [1 + f(φ−1)]·T·x`, as % above the true optimum:

```
 C/floor       f=0.0      f=0.25       f=0.5      f=0.75       f=1.0       f=1.5       f=2.0
     1.2    1.18036%    0.61908%    0.25771%    0.06060%    0.00000%    0.20629%    0.75316%
     1.5    1.58358%    0.80277%    0.32516%    0.07479%   -0.00000%    0.24359%    0.87422%
       2    1.59086%    0.79546%    0.31961%    0.07324%    0.00000%    0.24038%    0.87788%
```

The loss is quadratic in the `φ` error: a half-right `φ` already recovers ~80% of the gain, and
even a 100% overcorrection (`f=2`) beats doing nothing. An approximate `φ` is safe.

## 7. Cost and risk

**Safe by construction**

- `φ > 0` always, so eq 21's positivity requirement holds. Measured range over a wide grid:
  `[0.000001, 1.817431]`. `φ → 0` at the boundary for `cov=0` is correct — a D/D/1 station has
  `E[T] = 1/m` with no queueing term, so extra capacity buys it little.
- `zeta_from` stays **linear in T**, so `Optimizer._noise_floor` keeps working unchanged: it
  propagates a CI half-width through the same hook, and `φ` is noise-free.
- Convergence is unaffected: 13 budgets from `1.0001×` to `1e4×` the floor, iterations 5–18 vs
  6–18, **0 failures and 0 off-optimum rows**.

**What it costs**

- A `dT/dS` hook per station type. Two closed forms cover everything shipped today.
- `Result.zeta` stops satisfying eq 22 — a reported output changes meaning, and anything reading
  it as "the eq 22 value" must be re-read.
- Hardcoded ζ values in tests move.
- It is a deliberate **divergence from the paper's eq 22**. Worth a decision, not a silent commit.
- `tuned-ray-contraction-bound`'s convergence analysis is for the current map. The map changes,
  so the *proof* needs redoing even though the measurements above show no practical change.

**Honest limits**

- On the simulated path the gain is still below the noise floor, so the case for this is
  correctness and the G/G/1 stations — not the fork-join, and not measurable end-to-end today.
- Everything here is measured against the *analytic* model. `t_ul` is itself an approximation, so
  "exactly the global optimum" means exactly the optimum of the model qopt solves.
- The reference optimum is computed by my own water-filling plus a local-descent audit. Both are
  local methods; they agree with each other and with the slope-calibrated loop at every row, but
  none of the three establishes global uniqueness.

### What it does not touch: the tuned-`r*` retune

The first question an implementer asks is whether this removes the need for
`ForkJoinStation.retune` — which solves `r*` at the spend eq 21 granted, then re-expresses that
spend as a capacity at the station's *new* `alloc_cost`. **It does not, and the two are
independent.**

The reason is structural. ζ prices one half of the KKT — the scalar ν equalizing marginals across
stations — while `r*` answers the other half, the per-station ray condition, which is ν-free,
budget-free, weight-free and **ζ-free** (see
[`../forkjoin-coupled-vs-separate/`](../forkjoin-coupled-vs-separate/findings.md) §3). ζ is a
scalar per station and cannot encode a ray, so it could not absorb `r*` even in principle. And the
re-expression is neither half: it is a change of variables forced by eq 21 allocating `S` at a
price that **moves with `r*`**.

Every slope-calibrated number above was produced with the retune active — the probe builds its
fork-join stations with `r_star="tuned"`. Probe §7 varies the two independently, so the absence of
an interaction is measured rather than asserted:

```
 C/floor    zeta  rescale    final spend/C - 1  iters  converged
     1.5   level     True           +2.220e-16     11       True
     1.5   level    False           -6.318e-13     11       True
     1.5   slope     True           +0.000e+00     10       True
     1.5   slope    False           +1.776e-12     10       True
```

Two things that table shows, the second of which corrected my own expectation:

1. **ζ's calibration does not appear in the answer** — level and slope behave the same way with
   the re-expression on and with it off.
2. **Dropping it does not break the final budget.** `allocate` re-derives `S` from scratch each
   iteration, and once `r*` settles the price stops moving, so *at the fixed point the
   re-expression is a no-op.*

What it buys is per-**iterate** budget exactness, which is not a small effect while `r*` is still
moving — per station per iteration, `|spend after the ray solve / spend before − 1|`:

```
 C/floor    max drift        first         last
    1.05    3.388e-03    3.388e-03    5.878e-12
     1.5    2.690e-02    2.690e-02    3.288e-12
       5    7.869e-02    7.869e-02    9.276e-13
```

Up to **7.9% on the first iteration**, decaying to ~1e-12 on the last: the shape of a
transient-only correction. So it protects iterate feasibility and the stability guards, not the
final answer. If the re-expression were ever to go away, the lever is eq 21's decision *variable*
— allocating spend rather than `S` needs no rescale, because spend is what is held fixed — and
that is a change to the paper's equation, not to ζ.

**One real interaction, in the other direction.** Slope ζ needs `dT/dS`, taken along the station's
*current* ray, where level ζ needs only `E[T]`. §5 shows the radial derivative equals the true
marginal only *on* the optimal ray. That works out because the `Optimizer` calls `retune` **last**
in each iteration, so the next iteration's `zeta_from` sees a ray already optimal for the spend the
station holds. The retune's existing placement is therefore load-bearing for this proposal, not
merely compatible with it — worth knowing before anyone reorders that loop.

## 8. Suggested sequencing

1. Land this directory as analysis only — no behaviour change. (This PR.)
2. Decide the paper question: is eq 22 amended, or does qopt document a deliberate divergence?
3. ~~If it proceeds: implement behind a per-station opt-in so the default path stays bit-for-bit
   identical, with the two closed forms and a test that pins `φ ≡ 1` for M/M/1.~~ **Done** —
   `qopt/zeta.py`, `Station.dT_dS`/`phi`, `tests/test_zeta.py`.
4. Redo the contraction argument for the new map.
