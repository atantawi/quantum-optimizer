# Slope-calibrated ζ

**Status: proposal. Not implemented.** `main` ships eq 22's level calibration; this argues for
replacing it and records what that would cost.

Equation numbers follow [`../paper-map.md`](../paper-map.md): `eq 21` (the allocation rule) and
`eq 22` (the ζ inversion) mean `docs/analysis.pdf`, the snapshot committed in `37a3a11`. Every
number below is from
[`probe-output.txt`](probe-output.txt), reproducible with `python docs/slope-calibrated-zeta/probe.py`
(29 s, deterministic, no simulation service).

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

## 4. Payoff

`(1)` as shipped, `(1s)` slope-calibrated, `(3)` the coupled optimum over all stations —
water-filled on true marginals, then audited by pairwise local descent on the exact objective so a
bug in one cannot hide in the other. The audit runs in **both** directions: descent from the
water-filled answer must find nothing better, *and* descent from a 5% perturbation must return to
the same objective. The second direction is what makes the first meaningful — a descent that did
nothing would satisfy the first on its own.

```
--- 2 fork-join + 2 M/M/1 (phi moves only on the FJ pair) ---
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

## 8. Suggested sequencing

1. Land this directory as analysis only — no behaviour change. (This PR.)
2. Decide the paper question: is eq 22 amended, or does qopt document a deliberate divergence?
3. If it proceeds: implement behind a per-station opt-in so the default path stays bit-for-bit
   identical, with the two closed forms and a test that pins `φ ≡ 1` for M/M/1.
4. Redo the contraction argument for the new map.
