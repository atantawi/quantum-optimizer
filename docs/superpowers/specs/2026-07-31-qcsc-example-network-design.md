# QCSC Example Network — Design

**Date:** 2026-07-31
**Status:** Approved, not yet implemented
**Source:** `docs/analysis.pdf` §2 (mathematical models), §5 p. 20 (topology
characterization), Figure 5 p. 30 (sketch of this exact queueing network), §6.3
(parameter ranges)

## 1. Purpose

Every example in this repo so far has been three stations. This one is the
Quantum-Centric Supercomputing (QCSC) network of the paper: **14 stations, 16
single-server queues, two fork-joins, one open chain** — the first example large enough
to exercise the optimizer and the simulator on the topology the paper is actually about.

It runs at a single operating point under **three workload variants** — balanced,
quantum-dominant, classical-dominant (§2 p. 6) — which differ *only* in service rates.
Same topology, same arrival rate, same budget.

## 2. Scope

**In:** one new example module, `examples/qcsc_network.py`; one offline test module,
`tests/test_example_qcsc.py`; one gated live test added to
`tests/test_integration_qsim.py`.

**Out:** the parameter sweep over λ ∈ {0.4, 0.8, 0.9, 0.95}, r ∈ {1, 2, 4, 8},
p₁₁ ∈ {0.2, 0.5, 0.8} and both cost vectors that would reproduce Figures 2–4. Deferred
by decision; it needs its own spec. No change to `qopt/` itself — the library is used as
it stands.

## 3. Topology

One open chain. Jobs arrive Poisson at rate λ and are routed to the
parallel-sequential stream with probability p₁₁, to the sequential-parallel stream
otherwise. Each stream begins and ends with a classical CPU task (§2: "we envision that
both application classes start with a classical initialization task executed on system
CPUs and end with a classical termination task").

Stream 1 (parallel-sequential): CPU → fork-join → sequential phase → CPU.
Stream 2 (sequential-parallel): CPU → sequential phase → fork-join → CPU.

Each sequential phase splits with probability p₀ into a quantum-first tandem (QPU then
GPU) and with 1−p₀ into a classical-first tandem (GPU then QPU), per §3.1's four-queue
construction.

A fork-join is **one** `qopt` station with two servers, so N̂ = 14 stations and
N = 12 + 2·2 = 16 single-server queues, matching §5's counting.

### 3.1 Stations

| # | name | type | paper rate(s) | derived γ (λ=0.9) |
|---|---|---|---|---|
| 1 | `cpu_init_ps` | G/G/1 (M/M/1) | μ_I^P | 0.450 |
| 2 | `fj_pp` | fork-join | μ_Q^PP, μ_G^PP | 0.450 |
| 3 | `qpu_psq` | G/G/1 | μ_Q^{PS_q} | 0.225 |
| 4 | `gpu_psq` | G/G/1 | μ_G^{PS_q} | 0.225 |
| 5 | `gpu_psg` | G/G/1 | μ_G^{PS_g} | 0.225 |
| 6 | `qpu_psg` | G/G/1 | μ_Q^{PS_g} | 0.225 |
| 7 | `cpu_term_ps` | G/G/1 | μ_T^P | 0.450 |
| 8 | `cpu_init_sp` | G/G/1 | μ_I^S | 0.450 |
| 9 | `qpu_ssq` | G/G/1 | μ_Q^{SS_q} | 0.225 |
| 10 | `gpu_ssq` | G/G/1 | μ_G^{SS_q} | 0.225 |
| 11 | `gpu_ssg` | G/G/1 | μ_G^{SS_g} | 0.225 |
| 12 | `qpu_ssg` | G/G/1 | μ_Q^{SS_g} | 0.225 |
| 13 | `fj_sp` | fork-join | μ_Q^SP, μ_G^SP | 0.450 |
| 14 | `cpu_term_sp` | G/G/1 | μ_T^S | 0.450 |

Names use single underscores only: `Network._validate` rejects `__`, which would collide
with qsim's internal `<node>__b0` / `<node>__join` fork-join names.

All single-server stations are `GG1Station.mm1` (cov_a = cov_s = 1), matching the paper's
exponential service times. That `cov_a = 1` is *correct* only where arrivals really are
Poisson; §7 says where it is not, and that gap is the point of the simulated pass.

### 3.2 Routes

18 edges. `Network` derives every γ in §3.1 from these by the traffic equations; no γ is
hand-supplied anywhere.

```
src          -> cpu_init_ps   p11          src          -> cpu_init_sp   1-p11
cpu_init_ps  -> fj_pp         1.0          cpu_init_sp  -> qpu_ssq       p0
fj_pp        -> qpu_psq       p0           cpu_init_sp  -> gpu_ssg       1-p0
fj_pp        -> gpu_psg       1-p0         qpu_ssq      -> gpu_ssq       1.0
qpu_psq      -> gpu_psq       1.0          gpu_ssq      -> fj_sp         1.0
gpu_psq      -> cpu_term_ps   1.0          gpu_ssg      -> qpu_ssg       1.0
gpu_psg      -> qpu_psg       1.0          qpu_ssg      -> fj_sp         1.0
qpu_psg      -> cpu_term_ps   1.0          fj_sp        -> cpu_term_sp   1.0
cpu_term_ps  -> snk           1.0          cpu_term_sp  -> snk           1.0
```

## 4. Parameters

Named constants at the top of the module.

```
LAMBDA = 0.9        P11 = 0.5        P0 = 0.5        R = 4.0
B_PP  = B_SP  = 1.0                     # parallel-phase base level
B_PSQ = B_PSG = B_SSQ = B_SSG = 2.0     # sequential-phase base levels
MU_CPU = 20.0                           # mu_I, mu_T >> all others (§2)
C_QPU = 4.0         C_GPU = 1.0         C_CPU = 1.0
```

λ, r, p₁₁, p₀ are drawn from §6.3's grids: mid-to-heavy load, a clear heterogeneity
ratio, and a symmetric class split so the two streams carry equal traffic and the two
phase orderings are directly comparable.

The six per-phase base levels are a **modelling choice, not from the paper** — §6.1–6.2
study the phases in isolation with base 1. The parallel phase runs one task per unit
(base 1.0); a sequential phase splits comparable work across two serial tasks, so each is
faster (base 2.0). CPU init/termination is an order of magnitude faster than everything
else, which is the only property §2 asserts about it (μ_I, μ_T ≫ all others).

The cost vector is §6.3's second scenario (Figures 2–4, bottom row) with CPUs priced like
GPUs: quantum capacity is the scarce, expensive resource. **This choice is load-bearing —
see §5.1.**

## 5. Workload variants

One helper, `rates(workload, b) -> (mu_Q, mu_G)`, is the entire difference between the
three variants:

| variant | (μ_Q, μ_G) | within-phase ratio | paper (§2 p. 6) |
|---|---|---|---|
| `balanced` | (b, b) | 1 | μ_Q^ℓ = μ_G^ℓ |
| `quantum_dominant` | (b, R·b) | R = 4 | μ_Q^ℓ < μ_G^ℓ |
| `classical_dominant` | (R·b, b) | R = 4 | μ_Q^ℓ > μ_G^ℓ |

`balanced` is necessarily r = 1: the paper *defines* it as equal quantum and classical
rates, so it cannot carry r = 4. r = 4 is the study's heterogeneity ratio, shared by both
dominant variants, and `balanced` is the r = 1 reference.

Each `ForkJoinStation` takes `mu = min(mu_Q, mu_G)` and `r = max/min` (the class requires
`mu` to be the slower server and `r ≥ 1`). Costs `c1`/`c2` attach to the **server**, not
to the speed: the QPU branch costs `C_QPU` whether or not it is the bottleneck.

### 5.1 Why the cost vector is load-bearing

The topology is symmetric in QPU/GPU: both service orders appear (QG and GQ) with equal
γ, so swapping which side is slower merely permutes stations. Under **unit costs the
quantum-dominant and classical-dominant variants produce identical objectives** — the
comparison would be vacuous. Confirmed against the real code: at unit costs both floors
are 2.4525 and both objectives are 4.490082652796128 — equal bitwise as measured, but do
not assert it that way. `allocate`'s `slack` and `denom` are left-folds over the stations,
and the two variants present those summands in a different order; IEEE-754 addition is
commutative but not associative, so a one-ulp difference in either fold is permitted and
would propagate to every capacity and hence every sojourn time. The bit equality holds for
these particular values, not by construction. §9's test states the symmetry as the claim
actually is — the sojourn vectors are a permutation of each other — with `approx`
tolerances. `C_QPU = 4` breaks
the symmetry: it separates the floors (6.503 vs 5.490) and hence the allocations and the
objectives (4.528844 vs 3.463677). This is §6.3's own "given certain symmetries of our
QCSC models" observation, and §9 tests it.

## 6. Budget and objective

**One absolute budget shared by all three variants**, so that differences in the printed
E[T] are attributable to the workload rather than to the budget:

```
C = 6 x min_feasible_budget(balanced) = 41.040
```

Verified against the real code before adopting these numbers:

| variant | min_feasible_budget | C / floor | converged | objective |
|---|---|---|---|---|
| balanced | 6.840 | 6.00× | yes, 5 iterations | 6.401440 |
| quantum-dominant | 6.503 | 6.31× | yes, 5 iterations | 4.528844 |
| classical-dominant | 5.490 | 7.48× | yes, 5 iterations | 3.463677 |

Minimum slack `S·mu − γ` across all stations and variants was ≥ 1.25, so no variant sits
near its stability boundary.

**Weights ω_i = 1 for all 14 stations.** The optimized objective is therefore the plain
sum of 14 expected sojourn times — the paper's default and what the existing examples do.
It is *not* the mean end-to-end job sojourn time, which would need ω_i = γ_i/λ (visit
ratios). The example prints `Σ (γ_i/λ)·E[T_i]` alongside as a diagnostic; the optimized
objective stays unweighted.

## 7. Where arrivals are assumed Poisson, and what was measured against that

Stated in the module docstring. `GG1Station.mm1` (`cov_a = 1`) assumes Poisson arrivals
at every station; that assumption does not hold uniformly over this topology. The
mechanism reasoning below is unaffected by what follows — what turned out wrong is the
predicted *size* of the resulting gap, not which stations have non-Poisson arrivals. The
two streams run the same phases in opposite order, which makes them a controlled
comparison:

- `cpu_init_ps`, `cpu_init_sp` — a Bernoulli split of a Poisson stream is Poisson.
  `cov_a = 1` is exact here.
- Stream 1's `qpu_psq`, `gpu_psq`, `gpu_psg`, `qpu_psg` sit **downstream of a fork-join**.
  Their arrivals are join completions (a max over two branches), not Poisson. This is
  where `GG1Station.mm1`'s Poisson-arrival assumption is weakest.
- Both fork-joins run at r = 4 (§10: "both fork-joins run at r = 4"), so `t_ul`'s
  heterogeneous-server approximation carries a bias of its own at `fj_pp` and `fj_sp`.
  `fj_sp` additionally receives a **superposition of two tandem departure streams**
  rather than a Poisson input — two error sources stacked there.
- Stream 2's sequential queues are fed by a CPU whose own input is Poisson, so they
  should track the analytic values closely.

**Measured, and not borne out at this operating point.** A live run against
`qsim-service` (stopping rule `precision 0.02`, `alpha 0.05`, `minSamples 1e5`,
`maxSamples 4e6`, `maxWallClockSeconds 300`), kept at
[`docs/qcsc-example/live-run.log`](../../qcsc-example/live-run.log) so every number below
can be checked, converged for all three workloads; the
simulated pass stopped after 1 iteration / 2 simulation calls on `stop_reason =
noise-floor` (the capacity step was smaller than the noise the CIs imply).

One consequence of that early stop bounds what the run can be read as showing. Because the
analytic warm start had already converged and the single simulated step fell inside the
noise floor, both passes end at essentially the same capacity vector — compare any `S*`
column in the log, which agree to 3-4 decimals. So the agreement between the two
objectives (6.373131 / 4.518446 / 3.448158 simulated against 6.401440 / 4.528844 /
3.463677 analytic, within 0.5%) is a statement that the simulator and the closed form agree
*at the analytic optimum*, not that two independent optimizations found the same allocation.
That is the right comparison for the question §7 asks — it isolates model discrepancy from
allocation discrepancy — but it is a weaker claim than "the simulated optimizer reproduces
the analytic one", and nothing here tests the latter. Against that run:

- Across all 42 station rows (14 stations × 3 workloads), every `simulated − analytic`
  gap is at most ~1.1% in magnitude (worst: `qpu_psq` in `balanced`, −1.09%).
- Only 2 of those 42 rows exceed their own CI half-width — `qpu_psq` in `balanced` and
  `cpu_term_sp` in `quantum_dominant`. `qpu_psq` **is** one of the post-fork-join stations
  named above, and it carries the largest gap in the whole run — so this is not a case of
  neither flag landing on a predicted station.
- That does not, by itself, establish the predicted coupling. This test has a known false
  positive rate: at alpha = 0.05, with no bias present anywhere, roughly 2.1 of these 42
  rows are expected to flag by chance, and the run produced exactly 2 — the chance rate,
  not an excess. Narrowing to the 12 rows the prediction actually singled out (stream 1's
  `qpu_psq`, `gpu_psq`, `gpu_psg`, `qpu_psg`, across all three workloads), exactly 1
  flagged, against a chance expectation of ~0.6 — again the chance rate, not an excess. So
  the flag on `qpu_psq` is evidence neither for nor against the predicted coupling. (A
  single flag over 14 rows in one workload is, on its own, within the ~0.7-per-workload
  chance rate at alpha = 0.05 and should be read as noise rather than as coupling; see
  `print_gaps`'s docstring.)
- **The gaps do lean one way, and the per-row test above cannot see it.** In the committed
  log 31 of the 42 are negative — the simulator runs *below* the analytic prediction — with
  a negative mean in every workload (−0.307% balanced, −0.166% quantum-dominant, −0.251%
  classical-dominant) and a two-sided sign test on 31/42 giving p = 0.003. That p is
  optimistic for two separate reasons, neither of which overturns the direction:

  - The 42 rows are not independent. Stations within a workload share one simulation run,
    and the topology makes `qpu_psq`/`qpu_psg`/`qpu_ssq`/`qpu_ssg` analytically identical,
    so their errors are correlated.
  - One of the 31 negatives is at the resolution floor. `qpu_ssg` in `classical_dominant`
    has a gap of −9e-06 (−0.004%), about 180x smaller than that row's own CI half-width
    (±0.735% relative) — a sign, but not a measurement. Counting it is the standard
    treatment, since it is a genuine negative rather than a tie, and this section's per-seed
    table below counts every seed the same way. But it is the marginal row: drop it and the
    test is 30/42, p = 0.008. The conclusion is unchanged either way, and the honest reading
    of the committed log's headline is "p between 0.003 and 0.008, before any correlation
    correction" rather than 0.003 flat.

  **Replicated at three further base seeds, and it holds — with a smaller magnitude than
  the committed log alone suggests.** Note first that re-running the example does *not*
  replicate anything: `SimulationAnalyzer` defaults to `seed=20260729` with
  `seed_policy="fixed"` and the service is deterministic, so a second run reproduces every
  digit (measured). Varying the base seed instead:

  | base seed | negative | mean gap % | sign-test p |
  |---|---|---|---|
  | 20260729 (committed log) | 31/42 | −0.241 | 0.003 |
  | 8675309 | 29/42 | −0.149 | 0.020 |
  | 31415926 | 25/42 | −0.076 | 0.280 |
  | 2718281 | 30/42 | −0.131 | 0.008 |

  All four runs have a negative overall mean (4 of 4; p = 0.125 treating each run as one
  observation, which is the most conservative reading available). Pooled, 115 of 168 rows
  are negative. So the direction is real and reproducible. The *size* is smaller than the
  committed log implies: 20260729 is the most extreme of the four, and the central estimate
  across seeds is ~0.15%, not the ~0.25% that single run shows. One seed (31415926) is not
  significant on its own at p = 0.28, which is what a ~0.15% effect looks like against this
  stopping rule — the bias is at the edge of what precision 0.02 can resolve.

  A per-row CI test is structurally blind to all of this: a uniform bias that small flags
  almost no individual row while being plain in aggregate. It is a bias in the analytic
  form, of the sign expected if the closed form is conservative, but it is an order of
  magnitude smaller than the effect §7 originally predicted, and it is *not* localized to
  the stations the prediction named — see the next bullet.

- **`degraded` is not identically empty across seeds.** The committed run reports zero
  degraded measurements, but that is a property of its seed, not of the model: at seed
  8675309 the `classical_dominant` pass logs one γ-conservation miss (`cpu_term_sp`,
  simulated throughput CI (0.444527, 0.449863) excluding the derived γ = 0.450000). One
  miss in 144 checks (12 conservation-checked stations × 3 workloads × 4 seeds) is well
  inside what a 95% interval implies, so it is not a defect — but do not read "zero
  degraded" in the log as a guarantee the run will always be clean.
- Stream 1's post-fork-join queues are only modestly worse than stream 2's equivalents fed
  directly by a Poisson-input CPU, not the clear separation the prediction implied: mean
  `|gap %|` pooled over all three workloads is 0.458% for stream 1's four post-fork-join
  queues against 0.367% for stream 2's four equivalents (n = 12 each). Per workload the
  ratio is 1.46, 1.47, 0.91 — it leans the predicted way in two of three and reverses in
  the third, which at n = 4 per cell is not distinguishable from noise. So the controlled
  comparison the two streams were built for returns no usable signal at this operating
  point, rather than a null.
- `fj_pp` and `fj_sp`, expected to carry the largest bias, show gaps well under 1% and
  inside the CI half-width in both dominant workloads (r = 4).

Likely reason: the optimizer allocates 6.00×–7.48× the minimum feasible budget (§6), and
minimum slack `S·mu − γ` across all stations and variants was ~1.25 (§6) — every station
runs at modest utilization, where the *shape* of an arrival process matters far less than
it does close to saturation. The prediction above describes a regime this operating
point does not reach; it was tested under mid-to-heavy load but well inside the feasible
region, not near the stability boundary.

Actually probing the prediction would need either a tighter `precision` (so CI
half-widths fall well below the gaps being measured — at 2% the two are comparable) or a
heavier load (a higher λ, or a smaller budget multiple) that pushes utilizations up
toward saturation, where non-Poisson burstiness has more room to matter.

**Replicated at a larger sample, 2026-09-01 (issue #10).**
[`docs/forkjoin-s2-policy/simcheck-output.txt`](../../forkjoin-s2-policy/simcheck-output.txt)
runs this same statistic over **630 station rows** — 14 stations × 3 workloads × 5 seeds ×
3 fork-join ray policies — at the identical stopping rule, and reproduces both headline
figures: mean gap −0.126% at the ray this section measured, worst row 1.165%, against the
~0.15% and ~1.1% recorded above. The negative lean replicates a fifth, sixth and seventh
time (136 / 132 / 134 of 210 rows per policy). It also answers a question this section
could not ask, since it predates `r_star`: the bias is **the same size at the tuned ray as
at the default** (−0.132% against −0.126%), so the closed form is not worse where the tuned
policy operates.

**That does not discharge the two paragraphs above.** That run neither tightened `precision`
nor raised the load — it used `precision 0.02` at the same absolute budget `C = 41.040000`,
which is 6.00×–14.71× the minimum feasible budget depending on the policy, so if anything
*more* slack than the 6.00×–7.48× above rather than less — so
the arrival-coupling prediction remains untested, exactly as described. It is a bigger sample
of the same measurement at the same operating point, plus two new rays. Issue #7 owns the
parameter sweep that would change the operating point.

## 8. Output and execution

Per workload: a 14-row table (`station · gamma · S* · E[T] · zeta`, plus a `95% CI`
column on simulated runs), then the objective, then the visit-ratio-weighted mean. Then
one cross-workload summary: objective per workload plus **cumulative QPU and cumulative
GPU capacity** — the two axes of Figure 2, free to compute from `S*`.

The analytic pass always runs. With `QOPT_QSIM_URL` set, each workload runs a second time
on a **fresh** `Network` (independent station objects, so neither pass observes the
other's mutable state) through `SimulationAnalyzer`, reusing the stopping rule already in
`examples/simulated_mixed_network.py` (`alpha 0.05`, `precision 0.02`, `minSamples 1e5`,
`maxSamples 4e6`, `maxWallClockSeconds 300`), followed by a per-station
`simulated − analytic` gap table. Worst case that is roughly 5 iterations + 1 fresh-seed
evaluation per workload ≈ 18 POSTs, up to ~90 minutes at the wall-clock cap — but measured
(§7): the capacity step from the analytic warm start was already smaller than the noise
the CIs imply, so each workload stopped on `stop_reason = noise-floor` after 1 iteration /
2 simulation calls — 6 POSTs for the whole run, finishing in minutes, well inside the cap.
The simulated run is still executed as a background task rather than blocking, since the
worst case remains possible at a heavier load or tighter precision.

A `--dot` flag prints `network.to_dot()` and exits: a 14-station topology cannot be
verified by reading route tuples.

### 8.1 Degradation and errors

- **Infeasible budget** — checked and reported per workload before any run. All three are
  feasible at C = 41.04 (§6).
- **Missing confidence intervals** — `sojourn_ci` carries `None` in that slot, plus a
  `RuntimeWarning` and a `degraded` entry. Printed, never raised; the `None`-guard
  pattern from `examples/simulated_mixed_network.py::_print_table` is reused.
- **Fork-join γ-conservation** — `ForkJoinStation.sim_conservation_checked` is False
  (qsim-service#8), so 2 of 14 stations skip that witness. The output says so rather than
  letting the silence read as a pass.
- **Service unreachable** — `QsimClient(..., preflight=True)` fails fast.
- **Non-convergence** — `converged=False` with the final residual is printed, not raised.

## 9. Tests

`tests/test_example_qcsc.py`, all offline (`QOPT_QSIM_URL` deleted):

1. The 14 station names in order, and derived γ equal to the exact expected list — this
   is what catches a routing-probability slip.
2. All three variants converge at the shared budget with `S·mu > γ` at every station;
   objectives pinned to their exact values, as `tests/test_example_simulated.py` pins
   its numbers.
3. **The §5.1 symmetry claim**: with unit costs, the two dominant variants' per-station
   sojourn vectors are a permutation of each other and their objectives therefore agree;
   with `C_QPU = 4` the objectives differ. The whole three-variant comparison rests on
   this, so it is a test rather than a prose claim. Stated as the permutation and asserted
   with `approx` rather than as bit equality, for the reason §5.1 gives.
4. `balanced` has `r == 1` on both fork-join stations; both dominant variants have
   `r == 4`.
5. `main()` returns a result with `sim_calls == 0` when `QOPT_QSIM_URL` is unset.

One **gated** test added to `tests/test_integration_qsim.py` (skipped without
`QOPT_QSIM_URL`): a single `SimulationAnalyzer.evaluate()` on the QCSC network returns
E[T] for all 14 stations. Fourteen nodes with two fork-joins is new ground for the
serializer, and one POST is cheap.

## 10. Divergences from the paper, restated

Pre-existing and deliberate, recorded at `docs/optimizer-brainstorm-summary.md:89`; this
example inherits them and does not reopen them. They matter here because both fork-joins
run at r = 4:

- **Fork-join capacity.** The paper sets `S₂ = S₁/r`, which equalizes the two effective
  rates (`S₂μ̂₂ = S₁μ̂₁`) and so destroys the ratio r. `qopt` gives both servers the same
  S, keeping the effective ratio at r for every allocation.
- **Fork-join cost.** `c_FJ = c₁ + c₂` in `qopt`, not the paper's `c₁ + c₂/r`.
