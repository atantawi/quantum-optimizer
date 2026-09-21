# Coupled vs separate fork-join `r*`

Analysis only — **no code change is proposed here**. qopt's per-station tuned-`r*` solve is the
right design, and this records why.

| file | what it is |
|---|---|
| `findings.md` | the write-up: the three formulations, why their ray conditions coincide, and what the spend split is worth |
| `probe.py` | every check the write-up cites, self-contained (imports only `qopt` and the stdlib) |
| `probe-output.txt` | captured output, so the numbers are checkable without running anything |
| [`../audit-selfcheck.py`](../audit-selfcheck.py) | checks that `probe.py`'s optimum-confirming audit can actually fail, and has not drifted from the copy in `../slope-calibrated-zeta/` |

Reproduce with:

```
python docs/forkjoin-coupled-vs-separate/probe.py
```

Deterministic, no simulation service needed.

## The short version

Solving all fork-join stations as one coupled problem under a combined `C_FJ` gives **the same ray
condition** as solving each on its own: the within-station stationarity ratio is
`β₁/β₂`, in which the multiplier `ν`, the budget, and the weights all cancel. Coupling can
therefore only move *spends*, never rays.

Measured, the coupled fork-join block is worth at most **0.027%** of the objective — and it cannot
reach the larger **0.043%** effect, which is fork-join against the *single-server* stations. The
decisive instance is a network with one fork-join station: the coupled block is empty there, yet
the gap to the true optimum is at its largest.

The residual loss is a mispriced split, not a misplaced boundary. Fixing the price is
[`../slope-calibrated-zeta/`](../slope-calibrated-zeta/findings.md), which this document motivates.

Equation numbers follow [`../paper-map.md`](../paper-map.md).
