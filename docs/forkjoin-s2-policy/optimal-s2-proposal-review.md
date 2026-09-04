---
documentclass: article
fontsize: 10pt
papersize: letter
geometry:
  - margin=0.8in
mainfont: STIXGeneral
mathfont: STIX Two Math
monofont: Menlo
colorlinks: true
linkcolor: blue
urlcolor: blue
toccolor: black
---

# Reviewing the proposed S₂ procedure in `Optimal-S2.docx`

**Date:** 2026-09-04
**Status:** Review of an alternative proposal. Nothing in `qopt/` is changed, and nothing in
it is recommended for change.
**Evidence:** [`optimal-s2-check.py`](optimal-s2-check.py) $\rightarrow$
[`optimal-s2-check-output.txt`](optimal-s2-check-output.txt), six gates, all passing.

## Verdict

The proposal is the **dual** of the condition `qopt` already solves, not an alternative to
it. Its algebra is correct and its $dT/dB$ is *literally the same expression* as the
derivative in [`optimality-condition-derivation.md`](optimality-condition-derivation.md) §5.
The difference is bookkeeping: `qopt` eliminates the Lagrange multiplier by taking a ratio,
the proposal keeps it and requires it as an input.

That single choice is what breaks it:

1. As posed the problem has **zero degrees of freedom**, so there is nothing to minimize.
2. The budget $C=c_{FJ}$ it takes from the optimization loop **cancels**; it cannot reach
   the answer. The input that does the work is the multiplier $\chi$.
3. Given the correct $\chi$ it reproduces `qopt`'s ray to $7\times10^{-16}$ relative — but
   the correct $\chi$ is exactly what `qopt`'s other equation computes, and the only
   multiplier the loop actually carries is a different number.
4. It silently assumes server 1 is the bottleneck, and it cannot represent the kink
   $m_1=m_2$ — which is `classical_dominant`'s answer.

## 1. What the proposal is

Writing $A=S_1\mu_1$ and $B=S_2\mu_2$ for the two effective rates and $\gamma$ for the
station arrival rate, it forms

$$
\mathcal L = T + \chi\left(S_1c_1+S_2c_2-C\right),
\qquad C=c_{FJ},
$$

takes $S_1$ from step 1, and solves

$$
F(B)=\mu_2\frac{dT}{dB}+\chi c_2=0
$$

for $B$ by bisection, reporting $S_2^{*}=B^{*}/\mu_2$.

## 2. The algebra is correct

Gates **G1** and **G2** check it against `qopt`'s own code rather than against a
re-derivation:

| claim | measured |
|---|---|
| $T = \dfrac{1}{A-\gamma}+P(B)Q(B)$ | exact |
| $dP/dB=\gamma/(8B^2)$, $dQ/dB=-\dfrac{1}{(B-\gamma)^2}+\dfrac{1}{(A+B-2\gamma)^2}$ | exact |
| its $dT/dB$ vs `qopt`'s `_dt_dm1(g, m2, m1)` | $2.4\times10^{-16}$ relative |
| both vs a 5-point difference of `t_ul` | $3.2\times10^{-9}$ relative |

Its $P(B)=1-\gamma/(8A)-\gamma/(8B)$ is exactly $1-\alpha$ for `t_ul`'s
$\alpha=\frac{\gamma}{8}\left(\frac1A+\frac1B\right)$, so the two response-time models
coincide (with the caveat in §6).

## 3. It is the same condition, with the multiplier left in

$F(B)=0$ rearranges to

$$
\frac{\partial T}{\partial m_2}=-\chi\beta_2,
\qquad
\beta_2=\frac{c_2}{\mu_2},
$$

which is the **second** of the two stationarity equations in
[`optimality-condition-derivation.md`](optimality-condition-derivation.md) §3. `qopt` solves
the ratio of that equation and its $m_1$ twin,

$$
\frac{\left|\partial T/\partial m_1\right|}{\left|\partial T/\partial m_2\right|}
=\frac{\beta_1}{\beta_2},
$$

which eliminates $\chi$. **This is why `qopt` needs neither the station budget nor a price
handed in from the loop**, and it is the same cancellation noted in the derivation: the spend
$B$ fixes the *line*, and the ratio fixes the *point on it*, with the multiplier appearing in
neither.

## 4. As posed, there is nothing to optimize

With $S_1$ taken from step 1 **and** the equality constraint imposed, $S_2$ is already
determined:

$$
S_2=\frac{C-c_1S_1}{c_2}.
$$

Gate **G5** measures this at all three converged QCSC fixed points: the constraint
reproduces `qopt`'s $S_2=(r^*/r)S_1$ to at worst **5 ulp** — round-off in a
subtract-then-divide chain, i.e. the same number. So $\min_{S_2}$ ranges over a single
feasible point, and $d\mathcal L/dS_2=0$ determines $\chi$, not $S_2$.

The proposal escapes this only by dropping the constraint. Gate **G4** records the mechanism:
$C$ enters $\mathcal L$ additively, so it cancels from $d\mathcal L/dS_2$ and **appears
nowhere in $F$**. The concern that motivated this review is therefore justified, and in a
stronger form than "it takes $c_{FJ}$ as an input": $c_{FJ}$ is an input that cannot affect
the output.

Closing the loop the other way — using the constraint to pin $\chi$ — is vacuous. Since

$$
c_1S_1+c_2S_2=S_1\left(c_1+c_2\frac{r^*}{r}\right)=\text{spend}
$$

holds **identically for every** $r^*$, the constraint is satisfied whatever ray the station
is on, and such a fixed point would never move off its initial ray.

## 5. Equivalence, and the price of the multiplier

Gate **G3**: with $\chi$ set to the station's own shadow price
$\nu=\left|\partial T/\partial m_1\right|/\beta_1$, the proposal's root reproduces `qopt`'s
ray to $6.8\times10^{-16}$ relative, and $F$ evaluates to $\sim10^{-14}$ at `qopt`'s answer.
**So the two are equivalent — conditional on $\chi$.** But $\nu$ is precisely what `qopt`'s
first equation computes, so the proposal needs `qopt`'s answer as its input.

$\chi$ is the *only* input that reaches the answer, which makes its sensitivity the whole
error budget of the procedure (gate **G6**, at $\gamma=0.45$, $\mu=1$, $r=4$, $c_1=4$,
$c_2=1$, spend $=3$):

| $\chi/\nu$ | $r^*$ | budget error | $T$ error |
|---:|---:|---:|---:|
| 0.50 | 1.830937 | $+1.56\%$ | $-3.05\%$ |
| 0.80 | 1.637884 | $+0.46\%$ | $-1.14\%$ |
| 1.00 | 1.556681 | $+0.00\%$ | $+0.00\%$ |
| 1.25 | 1.481567 | $-0.43\%$ | $+1.31\%$ |
| 2.00 | 1.341413 | $-1.23\%$ | $+4.78\%$ |

A wrong $\chi$ does not merely mis-tune the ray, it **breaks the budget**: the station stops
costing the share eq 21 gave it. (The negative $T$ errors at $\chi<\nu$ are not
improvements — those rays overspend.)

### The loop's multiplier is not this multiplier

Eq 21 carries no explicit $\chi$, but implies one. Its stationarity gives
$S_i=\gamma_i/\mu_i+\sqrt{w_i\zeta_i/(\nu c_i\mu_i)}$, hence

$$
\nu_{\text{net}}=\frac{w\zeta\mu}{c_{\text{alloc}}\left(S\mu-\gamma\right)^2},
$$

which gate **G7** confirms is equal across all stations at eq 21's solution (spread
$\le5\times10^{-11}$) — the check that it has been recovered correctly. It is **not** $\nu$,
because it prices the $\zeta$-linearized surrogate $w\zeta/(S\mu-\gamma)$, which matches $T$
in *value* at the fixed point but not in *derivative*:

| workload | $\nu_{\text{net}}$ | $\nu_{\text{station}}$ | ratio | $r^*$ error | spend error |
|---|---:|---:|---:|---:|---:|
| `balanced` | 0.184909 | 0.177331 | 0.959019 | $-1.36\%$ | $-0.36\%$ |
| `quantum_dominant` | 0.126923 | 0.123612 | 0.973910 | $-0.90\%$ | $-0.11\%$ |
| `classical_dominant` | 0.068322 | 0.069096 | 1.011325 | $-1.12\%$ | $-0.56\%$ |

So substituting the one multiplier the loop can supply mis-tunes every ray by about a
percent and violates every station's budget share.

## 6. Two undeclared domain restrictions

**Server 1 must be the bottleneck.** The proposal sets $T_{\mathrm{bot}}=1/(A-\gamma)$, which
is `t_ul`'s $1/(\min(m_1,m_2)-\gamma)$ only when $A\le B$ — equivalently, at the optimum,
only when $\beta_1\ge\beta_2$. Where server 2 binds it is a different function, by

$$
T_{\mathrm{docx}}-T_{\mathrm{UL}}
=\frac{\gamma\left(B^2-A^2\right)}{8AB\left(A-\gamma\right)\left(B-\gamma\right)},
$$

confirmed in closed form by gate **G1** (up to $10.9\%$ on the probed points). On a
$\beta_2>\beta_1$ station, $F$ evaluates to $+10.3$ at the true optimum against a price term
of $47$, and its root is $2.5\%$ off in $r^*$ and $1.9\%$ off in spend. All three QCSC
workloads happen to satisfy $\beta_1\ge\beta_2$, so this would not surface there.

**It cannot represent the kink.** $T_{\mathrm{bot}}$ kinks at $m_1=m_2$, where the correct
condition is one-sided ([derivation §6](optimality-condition-derivation.md)). A smooth
$F(B)=0$ pairs the two rates' one-sided derivatives from *opposite* branches, so its root
steps past the kink into the region where its own $T$ is wrong. `classical_dominant`'s answer
is that kink (gate **G8**, $\beta_1=\beta_2=1$):

| ray | $r^*$ | $T$ at the same spend |
|---|---:|---:|
| `qopt` | 1.0000000 | 0.450461892 |
| proposal | 0.9845819 | 0.450588643 |
| 200000-point scan | 1.0000088 | 0.450461937 |

`qopt` returns the kink to within 1 ulp ($1.0000000000000002$; exactly $1.0$ at the station
level, where the prices are exactly symmetric, but here $m_1$ and $m_2$ are recovered from a
network spend by subtraction). The proposal's root is worse by $0.028\%$ at equal spend —
small, but it is a systematic failure to represent one of the three answers, not noise.

## 7. Minor points

- The **"direct minimization of $\mathcal L$"** alternative inherits the same $\chi$
  dependence ($C$ is an additive constant, so the argmin is unchanged) and additionally loses
  precision: comparing function values near a flat minimum pins the argument only to
  $\sqrt{\varepsilon}$. [`implementation.md`](implementation.md) records that as what stalls
  the outer fixed point at 9 iterations against bisection's 6, which is why `qopt` bisects
  the stationarity condition rather than minimizing `t_ul`.
- The sample code does not run as written: `(g/(8B**2))(...)` and `m2(term1 + term2)` are
  missing multiplication operators, `1/(A+B-2g)2` and `8B**2` are malformed, and the
  bracket `[g/m2 + 1e-6, 1000]` hard-codes an upper bound that need not contain the root.
- The Newton step is garbled in transcription, though the intent is unambiguous.

## 8. Conclusion

Nothing to change in `qopt`. The proposal is a valid restatement of the condition already
implemented, restricted to $m_1\le m_2$ and to smooth optima, and usable only when fed the
shadow price that `qopt`'s eliminated equation supplies. The multiplier is worth eliminating
precisely because the loop has no correct value for it.

## 9. Reproducing

```sh
python docs/forkjoin-s2-policy/optimal-s2-check.py
```

Exits non-zero if any gate fails. Stdout is deterministic — no wall clock, no seeds — so a
re-run is checkable with `diff` against the committed output. The proposal's $F(B)$ and $T$
are transcribed at the top of the script and are the only hand-written formulas in it;
everything they are compared against is imported from `qopt`, and the networks come from
`examples/qcsc_network.py`, so rates, costs and budget cannot drift from the example.
