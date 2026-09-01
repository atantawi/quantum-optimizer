# Choosing S₂ for a fork-join station — findings

**Date:** 2026-08-31
**Status:** Spike. Exploratory; nothing in `qopt/` was changed.
**Reproduce:** `python docs/forkjoin-s2-policy/probe.py`, output committed alongside as
[`probe-output.txt`](probe-output.txt). Every number below comes from that file.
**Superseded in part, 2026-09-01:** this is kept as the original spike record and is not
rewritten, so two things in it are out of date. §9's "no simulated cross-check was run" was
true then and is not now — it ran, and confirmed all three gains with each analytic gain
inside its own measured interval. And §9's first bullet infers from "every best ray is closer
to homogeneity than r = 4" that the new policy moves *toward* where `t_ul` is most
trustworthy, making qopt's `r = 4` the least-validated operating point. **That inference does
not hold for `balanced`,** whose fork-join hardware is `r = 1`: its incumbent is *already* at
m₁ = m₂ where `t_ul` is exact, so the best ray 1.44 moves *away* from there and the "than
r = 4" comparison does not apply to it at all. Both are settled in
[`implementation.md`](implementation.md), which is the maintained document.

## 1. The question

The recursive allocator (eq 21) sets one capacity per station. For a fork-join station
that fixes S₁, the speedup of the slower server, and leaves **S₂ a free variable**. Two
policies already exist, and each has a stated objection:

- **The paper** (`docs/analysis.pdf`, p. 18) sets
  S₂ = S₁/r, equalizing the effective rates and destroying the ratio r, and prices the
  station at `c_FJ = c₁ + c₂/r`. Objection: forcing two heterogeneous queues to become
  homogeneous is counter-intuitive, and the cost factor looks unreal.
- **qopt** gives both servers the same S, holding the effective ratio at r for every
  allocation, and prices the station at `c₁ + c₂`
  ([spec §10](../superpowers/specs/2026-07-31-qcsc-example-network-design.md)).
  Objection: an expensive second server makes the whole station expensive to speed up,
  so the slow server stays slow.

Is there a locally optimal S₂ — an r\* that is neither r nor 1?

## 2. The frame: both incumbents are rays of one line

Work in **effective rates** m₁, m₂ rather than in speedups. Raising m_k by one unit means
raising S_k by 1/μ̂_k, so the price of a unit of effective rate is

    β₁ = c₁/μ̂₁,    β₂ = c₂/μ̂₂ = c₂/(r·μ̂₁)

and the station's spend is exactly `β₁m₁ + β₂m₂`. Both policies are **rays through the
origin of that spend line**:

| policy | ray | spend | = |
|---|---|---|---|
| qopt | m₂ = r·m₁ | c₁S + c₂S | `(c₁+c₂)·S` |
| the paper | m₂ = m₁ | c₁S₁ + c₂S₁/r | `(c₁ + c₂/r)·S₁` |

**The paper's cost factor is therefore not unreal — it is the exact cost of its own S₂
rule.** Neither cost expression is wrong; they are the spend along two different rays.
That reframes the original objection: the paper is not mispricing the station, it is
committing to a ray. So is qopt.

Generalizing the ray to a free ratio **r\*** with m₂ = r\*·m₁ gives

    spend = S·(c₁ + c₂·r*/r)        →   alloc_cost(r*) = c₁ + c₂·r*/r

a **one-parameter family** containing qopt at r\* = r and the paper at r\* = 1. The probe
asserts this identity holds bit-for-bit on all three workloads (Q9, "family identity
check: PASS"), so the comparison below is between members of one family, not between
different models.

Keeping the ray is what keeps spend **linear in S**, which is what eq 21's budget column
requires. That constraint turns out to be the whole story — see §7.

## 3. The gate: is the question even well posed?

`t_ul` blends an upper bound with a bottleneck term, `T = (1−α)T_UB + α·T_bot`, with
α = (ρ₁+ρ₂)/8. Because α depends on m₂ through ρ₂ = λ/m₂, raising m₂ *lowers* α and
shifts weight onto the strictly larger `T_UB`. Expanding:

    ∂T/∂m₂ = α′·(T_bot − T_UB) + (1−α)·T_UB′ + α·T_bot′

and the first term is **positive** — more m₂ looks harmful. If it ever won, r\* would be
an artifact of the approximation rather than a property of fork-join queues, and it would
be a silent one: a perfectly plausible r\* with no symptom.

**It never wins.** Over 30 configurations (3 workloads × 2 stations × 5 budget multiples
from 1.05× to 20× the floor), `∂T/∂m₂ < 0` at every probed point; the largest value seen
is −8.705e-09. The analytic derivative is verified against a 5-point finite-difference
stencil to 4.6e-04 relative. And `t_ul` restricted to the spend line is **unimodal** —
at most one interior local minimum on a 4001-point grid in all 30 cases — so a 1-D solve
is safe.

Both gates pass. Everything below is therefore about fork-join queues as `t_ul` models
them, not about an artifact of the blend. (§9 says what `t_ul` itself does and does not
license.)

## 4. What r\* is

r\* is a function of two things only: the **price ratio** β₁/β₂ = (c₁/c₂)·r, and how far
the station's spend sits above its stability floor. It is *not* a function of r alone,
and it is *not* a constant across allocations.

    workload            stn       r   b1/b2  B/floor      r*     T(r*)   T(qopt)  T(paper)   gain%
    balanced            fj_pp   1.0   4.000     1.20   1.079  14.81628  15.50926  15.50926    4.47
    balanced            fj_pp   1.0   4.000     4.00   1.461   1.01745   1.08796   1.08796    6.48
    balanced            fj_pp   1.0   4.000    20.00   1.633   0.16204   0.17471   0.17471    7.25
    quantum_dominant    fj_pp   4.0  16.000     1.20   1.242  13.20789 111.11505  15.50926   88.11
    quantum_dominant    fj_pp   4.0  16.000     4.00   2.274   0.89239   0.95285   1.08796    6.35
    quantum_dominant    fj_pp   4.0  16.000    20.00   2.693   0.14138   0.14522   0.17471    2.64
    classical_dominant  fj_pp   4.0   1.000     1.20   1.000  15.50926   UNSTABLE  15.50926       --
    classical_dominant  fj_pp   4.0   1.000     4.00   1.000   1.08796   3.74084   1.08796   70.92
    classical_dominant  fj_pp   4.0   1.000    20.00   1.000   0.17471   0.33041   0.17471   47.12

Four things fall out, all of them measured:

**r\* → 1 at the stability boundary, and → a price-determined asymptote at large spend.**
The two limits explain both incumbents. The paper is exactly right in the tight-budget
limit; qopt is right in neither limit, because the asymptote depends on β₁/β₂ (1.633 at
ratio 4, 2.693 at ratio 16) and not on r.

**The paper's ray has the optimal stability floor; qopt's is up to 2.5× worse.** This is
not a coincidence: the cheapest stable point has both servers just above γ, which *is*
homogeneous, so the paper's ray passes through the true floor.

    workload               r   optimal      qopt     paper  qopt/opt  paper/opt
    balanced             1.0    2.2500    2.2500    2.2500     1.000      1.000
    quantum_dominant     4.0    1.9125    2.2500    1.9125     1.176      1.000
    classical_dominant   4.0    0.9000    2.2500    0.9000     2.500      1.000

At B/floor = 1.20 in `classical_dominant`, qopt's ray is outright **unstable** where the
line still has feasible points — a real consequence, not a rounding effect.

**"balanced" should not run homogeneous.** It has r = 1, so qopt and the paper coincide
exactly (identical objectives, 6.401440), yet r\* runs 1.08 → 1.63 because
c_QPU/c_GPU = 4 makes β₁/β₂ = 4. r = 1 is a statement about *rates*; homogeneity is
optimal only when *prices* match. This is the cleanest available demonstration that the
two are different claims.

**r\* is driven by the price ratio, and the same β₁/β₂ gives the same r\* regardless of
which device is slower.** Sweeping c_QPU/c_GPU with the topology fixed (Q7): at
β₁/β₂ = 1 both dominant workloads give r\* = 1.0000; at β₁/β₂ = 4 both give 1.5298; at 16
both give 2.4447. Structure enters only through β.

## 5. Where r\* < 1 lives

r\* < 1 — buying the nominally *faster* server down below the slower one — needs
c₂/c₁ > r. Measured crossings (Q7, at 6× the floor):

- `quantum_dominant` (QPU slow): r\* crosses 1 between c_QPU/c_GPU = 0.25 and 0.5, i.e.
  only if the QPU were **cheaper** than the GPU. Unreachable in practice.
- `classical_dominant` (QPU fast): crosses between 3.0 and 4.0. At the QCSC cost vector
  c_QPU/c_GPU = **4.0 exactly**, r\* = 1.0000 — the knife edge. Past it, r\* = 0.818 at
  ratio 8 and 0.519 at ratio 32.

So r\* < 1 is real and reachable, but only when the quantum device is both intrinsically
faster *and* priced above its speed advantage. At the QCSC cost vector we sit exactly on
the boundary.

**What r\* < 1 does not mean.** It is not "a GPU would do instead of a QPU". The two
fork-join branches are **not substitutes — they are both mandatory**: one job forks into a
quantum task *and* a classical task and joins on both. There is no decision variable
anywhere in this model for routing work to a different device, so the model is
structurally incapable of expressing "we do not need quantum". r\* < 1 is a *provisioning*
statement about two required resources: do not run the expensive branch faster than the
cheap one. The math is also symmetric in quantum/classical — it sees (β₁, β₂) and nothing
else, and swapping the price tags holds the GPU back instead. Any conclusion of the form
"hybrid loses" would come from the cost vector, which is our modelling choice
([spec §5.1](../superpowers/specs/2026-07-31-qcsc-example-network-design.md) flags it as
load-bearing), not from the fork-join structure. The question *is quantum worth its price*
needs a model with an offload/routing decision; that is a different model.

## 6. Fork-join vs series — the intuition is backwards

There *is* a structural asymmetry. `T_bot = max(1/(m₁−λ), 1/(m₂−λ))` has identically zero
derivative in the non-bottleneck server, and in `T_UB` the `−1/(m₁+m₂−2λ)` term partially
cancels the gain from m₂. A fraction α of the response time is blind to the non-bottleneck
server. Series has neither: `T = T₁(m₁) + T₂(m₂)` is separable.

**But that asymmetry runs the opposite way to the intuition.** Matched control — same γ,
same spend line, same (β₁,β₂), two M/M/1 queues in *tandem* instead of forked — comparing
the slack ratio Q = (m₂−γ)/(m₁−γ):

        b1/b2  Q_series  Q_forkjoin  exp_series   exp_fj  local d/dlog
          2.0    1.4142      1.2707      0.5000   0.3456        0.3767
          4.0    2.0000      1.6495      0.5000   0.3610        0.3757
         16.0    4.0000      2.7636      0.5000   0.3666        0.3683
        256.0   16.0000      7.4869      0.5000   0.3630        0.3513

Series sits at exactly 1/2 — the closed form, which also validates the fit. Fork-join sits
at ≈0.36, **shallower**, and remarkably flat across two decades of price ratio.

This **corrects a claim made earlier in the discussion**, which predicted fork-join would
be *steeper* than 1/2. It is shallower, and the sign of that difference inverts the
conclusion:

> The fork-join **resists** shifting budget to the cheap server. It compresses toward
> homogeneity relative to series, because you cannot profitably over-buy the cheap server
> (it goes non-bottleneck and blind) and you cannot starve the expensive one (it becomes
> the bottleneck and dominates the max).

So "hybrid loses on fork-join stations but not in series" is backwards. Fork-join
*protects* the expensive branch's provisioning. Where an expensive resource gets starved
hardest is in the **series** stations, where the −1/2 elasticity applies with no
saturation to check it. This also retroactively partially vindicates the paper's r → 1:
fork-join pulls toward balance anyway, so homogenizing is a better approximation than it
looks — just not for the reason the paper gives.

Caveat on scope: this compares *slack ratios at a fixed station spend*. It is not a
statement about total quantum spend across the network; see §10.

## 7. The decisive result: the ray matters, the split does not

Two embeddings were run on the full 14-station QCSC network at the **same** absolute
budget C = 41.040000 (`BUDGET_MULTIPLE × the balanced floor`), so all comparisons are
same-money:

- **`split`** — the station internally minimizes `t_ul` on its own spend line at every S,
  with `alloc_cost = c₁+c₂` left unchanged. This is the "obviously right" embedding: the
  current policy is a feasible point of that minimization, so it can only weakly improve
  the station.
- **`r*` family** — a fixed ray with `alloc_cost = c₁ + c₂·r*/r`, swept over
  r\* ∈ [0.20, 6.00].

    workload            policy       floor        obj      d%  it  cnv  fj_pp r*  fj_sp r*
    balanced            qopt         6.840   6.401440    0.00   5 True    1.0000    1.0000
    balanced            paper        6.840   6.401440    0.00   5 True    1.0000    1.0000
    balanced            optimal      6.840   6.248828    2.38   4 True    1.4457    1.4457
    quantum_dominant    qopt         6.503   4.528844    0.00   5 True    4.0000    4.0000
    quantum_dominant    paper        5.828   4.776428   -5.47   4 True    1.0000    1.0000
    quantum_dominant    optimal      6.503   4.432330    2.13   6 True    2.3195    2.3195
    classical_dominant  qopt         5.490   3.463677    0.00   5 True    4.0000    4.0000
    classical_dominant  paper        2.790   2.613335   24.55   4 True    1.0000    1.0000
    classical_dominant  optimal      5.490   2.620524   24.34   9 True    1.0000    1.0000

**Neither incumbent dominates.** The paper beats qopt by 24.55% in `classical_dominant`
and loses to it by 5.47% in `quantum_dominant`. Both original objections are validated:
each policy is wrong, in opposite regimes.

**The per-station dominance argument does not survive eq 21.** In `classical_dominant` the
paper (2.613335) *beats* the split policy (2.620524) by 0.27%, which the dominance
argument said was impossible. The reason is exact and measurable. At the converged point
the split policy chooses r\* = 1.0000 — the *identical* split to the paper's — so the only
remaining difference is `alloc_cost`, hence the share of C the station receives:

    policy    alloc_cost  S(fj_pp)    spend       m1       m2    T(fj)        obj
    qopt           5.000    1.9062   9.5308   1.9062   7.6247  0.70938   3.463677
    paper          2.000    3.7466   7.4932   3.7466   3.7466  0.45046   2.613335
    split          5.000    1.5920   7.9601   3.9801   3.9801  0.42092   2.620524

The split policy reports `S = spend/(c₁+c₂)` while its true m₁ comes from the inner solve,
so **"S" no longer means "server 1 runs at S·μ̂₁"** and eq 22's surrogate
ζ = T·(S·μ̂₁ − γ) is anchored to a rate the station does not have. eq 21 then mis-prices
the station: here it **over**-funds it (spend 7.96 against the paper's 7.49) and buys a
better `T(fj)` that is not worth what the other 13 stations gave up. The direction is
incidental; consistency is the point. It also costs convergence — 9 iterations against 4
and 5, the surrogate fighting the true curve.

**The family fixes it, and the local condition predicts the answer.**

    workload               r  r*_best  obj(best)  obj(qopt)  obj(paper)  obj(split)  best vs qopt
    balanced             1.0    1.440   6.249439   6.401440    6.401440    6.248828         2.37%
    quantum_dominant     4.0    2.320   4.431693   4.528844    4.776428    4.432330         2.15%
    classical_dominant   4.0    1.000   2.613335   3.463677    2.613335    2.620524        24.55%

The best ray (1.440 / 2.320 / 1.000) agrees with the local optimality condition evaluated
at the converged spend (1.4457 / 2.3195 / 1.0000) to within the 0.02 sweep grid, in all
three workloads. **So the local condition is the right rule — it just has to be applied
with consistent pricing.** And the objective-vs-r\* curve is unimodal with an interior
minimum in every workload (in `classical_dominant`, r\* = 0.75 → 2.645368 and
r\* = 1.25 → 2.633480 both sit above r\* = 1.00 → 2.613335).

The headline, then: **the best ray and the sophisticated inner split land within ±0.3% of
each other, while the choice of ray is worth 2.2% to 24.6%.** What matters is which ray,
not how cleverly the split is computed at a fixed spend. In `classical_dominant`
r\*_best = 1.000, so the paper's rule is *exactly optimal there* — which is why it wins by
24.55%. It is not optimal anywhere else, including in `balanced` where it coincides with
qopt and both are 2.4% off.

## 8. Answers

**Is there a locally optimal way of selecting S₂?** Yes, and it is a one-line condition:

    |∂T/∂m₁| / |∂T/∂m₂| = β₁/β₂ = (c₁/c₂)·r

a tug-of-war between a price ratio and a marginal-value ratio. It defines an r\* that is
neither r nor 1, agrees with the network-level optimum to within the sweep grid, and is
worth 2.2%–24.6% on the QCSC objective at fixed budget. β₁ = β₂ recovers the paper's r → 1
exactly by symmetry, and the stability boundary recovers it as a limit.

**Does hybrid lose on fork-join stations but not in series?** No — the reverse. Fork-join
is *less* price-elastic than series (slope ≈0.36 against exactly 0.50), so it resists
starving the expensive branch. And in any case this model cannot answer the underlying
question, because its fork-join branches are complementary and mandatory, with no routing
decision to express substitution.

## 9. What this does and does not license

- **It rests on `t_ul`**, which is documented exact only at m₁ = m₂, and whose
  heterogeneous-server bias was measured
  ([spec §7](../superpowers/specs/2026-07-31-qcsc-example-network-design.md)) **only at
  r = 4**. Reassuringly, every best ray (1.44 / 2.32 / 1.00) is *closer* to homogeneity
  than r = 4, so the new policy moves toward where the approximation is most trustworthy,
  not away — and qopt's current r = 4 operating point is the least-validated of the lot.
- **Effect size against known model error.** The measured analytic-vs-simulated bias is
  ~0.15% mean with a worst per-row gap of ~1.1% (spec §7). The 24.6%
  `classical_dominant` gain is far outside that. The 2.2%/2.4% gains are ~10× the mean
  bias but only ~2× the worst per-row gap — **suggestive, not established**, and they are
  the ones a simulated cross-check should target.
- **No simulated cross-check was run.** `QOPT_QSIM_URL` is unset in this session, so the
  planned confirmation at the r\* operating point did not happen. It is less urgent than
  predicted (previous bullet) but it is the outstanding piece of evidence for the two
  small gains.
- **Both fork-joins move together** in every result (`fj_pp` and `fj_sp` share γ = 0.45
  and identical rates), so the three workloads give three independent data points, not
  six.
- **Single operating point.** λ = 0.9, p₁₁ = p₀ = 0.5, one budget multiple, one cost
  vector for the network-level results (§7). The r\* mechanism was swept over price ratio
  and budget (§4, §5), but the network-level gains were not.
- **Nothing was changed in `qopt/`.** Test suite: 250 passed, 10 skipped (the gated qsim
  tests). The probe patches `examples.qcsc_network.ForkJoinStation` inside a
  `try/finally` so the topology, rates, costs and budget come from the real example module
  and cannot drift from it.

## 10. If this were to be implemented

Not decided, and out of scope for this spike. Recorded so the reasoning is not lost:

1. **Parameterize `ForkJoinStation` by r\***, with `alloc_cost = c₁ + c₂·r*/r` and
   `sojourn_time(S) = t_ul(γ, S·μ̂₁, r*·S·μ̂₁)`. Both incumbents become members of the
   family (r\* = r and r\* = 1), which makes the change strictly a generalization and lets
   existing behaviour be pinned as a default.
2. **Anchor `mu` on the effectively-slower server**, μ̂₁·min(1, r\*). Without this,
   r\* < 1 lets eq 21's base term γ/μ hand back an S that starves server 2 —
   `_check_stable(S·mu)` currently guards only server 1. The probe does this and the
   stability check becomes `min(m₁, m₂)`.
3. **Solve for r\* as a nested fixed point**, not at a fixed station spend: r\* satisfies
   the local condition at the spend eq 21 gives it *when priced at* `c₁ + c₂·r*/r`. The
   measured agreement in §7 is the evidence this converges to the network optimum.
4. **Do not use the inner-split embedding** (§7). It breaks the meaning of S, mis-prices
   the station through ζ, degrades convergence, and buys ±0.3% against a 2–25% effect.
5. **Open question not probed:** total QPU vs GPU capacity purchased across the whole
   network under each policy. `capacity_by_unit` in the example cannot see it, because it
   adds a fork-join's S to *both* unit totals — which is only correct while both servers
   share one S. Any r\* work must fix that reporting contract too.
