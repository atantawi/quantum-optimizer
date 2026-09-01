# Choosing S₂ for a fork-join station — artifacts

Issue [#10](https://github.com/atantawi/quantum-optimizer/issues/10): the recursive allocator
(eq 21) sets one capacity per station, which fixes `S₁` on a fork-join and leaves **`S₂`
free**. Both pre-existing policies turn out to be rays of one family, `m₂ = r*·m₁` priced
`c₁ + c₂·r*/r`, with qopt's default at `r* = r` and the paper's rule at `r* = 1` — and neither
dominates. `ForkJoinStation` now takes `r_star`, which can also be *solved*.

## Which file to read

**[`implementation.md`](implementation.md) is the maintained document.** It carries what was
built, the measured results, the simulated cross-check, and every correction to the plan and
to the spike. Read it first, and prefer it wherever it disagrees with anything else here.

[`findings.md`](findings.md) is the **original spike record, superseded in part** and
deliberately not rewritten — so that what was known when the design decisions were taken stays
readable. Its header lists what has since changed. Two things in it are stale as *status*
rather than as reasoning: §9's "no simulated cross-check was run", and §9's claim that every
best ray is closer to homogeneity than `r = 4` (false for `balanced`, whose hardware is
`r = 1`).

## The two probes

Neither changes anything in `qopt/`, and both import `examples/qcsc_network.py` rather than
restating its topology, so rates, costs and budget cannot drift from the example.

| | what it does | output |
|---|---|---|
| [`probe.py`](probe.py) | **Analytic.** Answers findings' questions: is there a locally optimal `S₂`, is the objective unimodal in `r*`, what the price elasticity and the stability floors are, and the decisive network-level comparison of the two incumbents against the family. Never touches the simulator. | [`probe-output.txt`](probe-output.txt) |
| [`simcheck.py`](simcheck.py) | **Simulated.** The cross-check findings §9 named as the outstanding evidence: does the analytic gain survive measurement? 3 workloads × 3 policies × 5 base seeds against `qsim-service`, paired by seed. | [`simcheck-output.txt`](simcheck-output.txt) |

Regenerate:

```
python docs/forkjoin-s2-policy/probe.py > docs/forkjoin-s2-policy/probe-output.txt

QOPT_QSIM_URL=http://localhost:8080 python docs/forkjoin-s2-policy/simcheck.py \
    > docs/forkjoin-s2-policy/simcheck-output.txt
```

## Reading `simcheck-output.txt`

Conditions are in its own header block, and §0 **measures and then gates on** the service's
provenance rather than trusting a tag — a `qsim-service` build predating its #11 and #13 returns
plausible-looking garbage with `success: true` throughout, and an image tag does not tell you
which commit it was built from. See [`../qcsc-example/README.md`](../qcsc-example/README.md) for the
version floor.

**A re-run at the committed seeds is not a replication.** `SimulationAnalyzer` defaults to
`seed_policy="fixed"` and the service is deterministic given a seed, so repeating the command
returns every digit unchanged — measured, not assumed. That confirms the pipeline and tells you
nothing new about the statistics. Vary `--seeds` for an independent sample.

Three things the output prints specifically to stop a misreading, each because a first reading
of this run got it wrong:

- **Per-seed gains under every row**, because the offset between the analytic and the measured
  *mean* gain is smaller than the seed-to-seed scatter. Reading a direction off one seed gets
  it wrong: at the committed seed `balanced`'s gain narrows while the mean widens.
- **Both the conservative and the independent propagated half-width** on each measured
  objective, neither of which is the primary interval. The paired spread across seeds is,
  because it measures the variability instead of assuming a correlation structure across 14
  stations that share one simulation run.
- **Which station rows are new evidence and which merely reproduce spec §7**, plus which policy
  cells are bit-identical to another. The default seed list contains all four of spec §7's
  replication seeds, and at the default ray those cells reproduce its rows bit-for-bit — the
  service is deterministic given a seed — so the pooled bias figures are **not** a replication
  of it. The first write-up said they were. Likewise `balanced`'s `equal-rate` is `invariant-r`
  (hardware `r = 1`) and `classical_dominant`'s `tuned` is `equal-rate` (the ray solves to 1),
  so two of the six policy rows are one measurement printed twice.

The `--precision`, `--seeds`, `--workloads` and `--policies` flags exist so a single
inconclusive cell can be re-run tighter without editing the probe. Report a tightened run
separately: halving `precision` costs roughly 4× the samples and is not comparable to the
0.02 grid.
