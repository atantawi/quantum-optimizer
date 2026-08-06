# QCSC example — artifacts

Evidence and diagrams for `examples/qcsc_network.py`, the paper's 14-station
Quantum-Centric SuperComputing network. The design is
[`docs/superpowers/specs/2026-07-31-qcsc-example-network-design.md`](../superpowers/specs/2026-07-31-qcsc-example-network-design.md).

## `live-run.log`

The analytic and simulated passes for all three workloads, with the per-station
`DIFFERENCE` tables. This is the run that §7 of the spec cites: it is kept because §7
draws a conclusion from specific per-station gaps, and a conclusion whose evidence lives
only in someone's terminal cannot be checked.

Conditions: `precision 0.02`, `alpha 0.05`, `minSamples 100000`, `maxSamples 4000000`,
`maxWallClockSeconds 300`; `qsim-service` built from commit `df45cd1`, which is the first
build that both honours `minSamples` and writes `alpha` rather than its complement
(qsim-service #11 and #13). Earlier builds silently ignored both, which made an earlier
run of this same example unusable — gaps the size of the confidence intervals and five
throughput checks excluding their derived γ.

What the run shows, in one line: every one of the 42 station rows agrees with the
analytic prediction to within 1.1%, the 2 rows that exceed their own CI half-width are the
number expected by chance at α = 0.05 over 42 rows, and the gaps nonetheless lean
consistently negative (31 of 42; negative mean in all three workloads), which is the one
signal in the run that the per-row test cannot see. That negative lean replicates at three
further base seeds — 4 of 4 runs with a negative mean — though this seed is the most extreme
of the four and the cross-seed central estimate is ~0.15% rather than the ~0.25% here. See
§7 for the per-seed table and what each of those does and does not license.

Regenerate:

```
QOPT_QSIM_URL=http://localhost:8080 python -m examples.qcsc_network
```

**This reproduces bit-identically, which is worth knowing before you read anything into a
re-run.** `SimulationAnalyzer` defaults to `seed=20260729` with `seed_policy="fixed"`, and
the service is deterministic given a seed, so a second run of the command above returns
every digit unchanged — measured, not assumed. A re-run therefore confirms the pipeline
still works and tells you *nothing* new about the statistics in §7: it is the same
measurement, not a second sample.

For an independent sample, vary the base seed instead — the example does not expose a flag
for it, so construct the analyzer directly:

```python
SimulationAnalyzer(network, client, seed=<other seed>)
```

Sample counts do move with the seed, and therefore so do the intervals and the last digits
of every mean.

## `topology.dot`

The 14 stations and 18 routing edges as Graphviz DOT, emitted by the example itself, so
it cannot drift from the code the way a hand-drawn diagram would. Node shapes distinguish
the two fork-join stations (`box3d`) from the single-server queues (`box`).

Regenerate, and render:

```
python -m examples.qcsc_network --dot > docs/qcsc-example/topology.dot
dot -Tpng docs/qcsc-example/topology.dot -o /tmp/qcsc.png
```

## An open modelling choice: the fork-join weight `ω_FJ`

The objective is `Σ ωᵢ E[Tᵢ]`, and this example uses the paper's default `ωᵢ = 1` at every
station — including the two fork-joins. That is worth stating out loud, because a fork-join
station stands in for **two** servers, so inheriting `ω = 1` gives it the same weight in the
objective as a single-server queue.

The paper does not settle it. `ω_FJ` appears exactly once in `docs/analysis.pdf` — p. 18, "in
terms of `c_FJ`, `ω_FJ`" — and is never defined. `c_FJ` on that same page gets both a derivation
and an explicit definition, so the cost of a collapsed fork-join is specified and its weight is
not.

Two independent arguments both give `ω_FJ = 2`: the station stands in for two servers each
carrying `ω = 1`, so it inherits `ω₁ + ω₂`; and minimizing `Σ ωᵢ E[Tᵢ]` equals minimizing mean job
sojourn `Σ (γᵢ/λ) E[Tᵢ]` exactly when `ωᵢ ∝ γᵢ`, which here — visit ratios 0.5 at the CPU stations
and both fork-joins, 0.25 at the eight sequential stations — makes 2 the mean-job-optimal
fork-join weight when `ω_seq = 1`. Measured across the three workloads, `ω_FJ = 2` lowers mean job
sojourn by 1.93–2.37%, and for `balanced` the two measured intervals do not overlap.

**This is not a defect in the example.** `ωᵢ = 1` is the paper's default, `visit_ratio_weighted`
already documents that the optimized objective is a different quantity from mean job sojourn, and
the plain sum of station sojourn times is a legitimate objective. The narrow point is that `ω = 1`
on a collapsed fork-join silently discards one server's worth of weight — a choice worth making
deliberately rather than by inheritance.

Tracked, with the method and the open questions, in
[#8](https://github.com/atantawi/quantum-optimizer/issues/8).
