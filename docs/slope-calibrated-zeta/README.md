# Slope-calibrated ζ

A proposal, with evidence. **Nothing here is implemented** — `qopt` on `main` ships the
level-calibrated ζ of eq 22 and this directory argues for changing that.

| file | what it is |
|---|---|
| `findings.md` | the write-up: the claim, the derivation, the costs, and what implementing it would touch |
| `probe.py` | every check the write-up cites, self-contained (imports only `qopt` and the stdlib) |
| `probe-output.txt` | captured output of `probe.py`, so the numbers in `findings.md` are checkable without running anything |
| [`../audit-selfcheck.py`](../audit-selfcheck.py) | checks that `probe.py`'s optimum-confirming audit can actually fail, and has not drifted from the copy in `../forkjoin-coupled-vs-separate/` |

Reproduce with:

```
python docs/slope-calibrated-zeta/probe.py
```

It is deterministic and needs no simulation service.

## The short version

eq 21's optimality condition is a statement about the **slope** of each station's sojourn-time
curve, but eq 22 calibrates ζ to match that curve's **level**. In the form `T = ζ/x` the two are
rigidly tied, so matching one means missing the other. Matching the slope instead —
`ζ = φ·T·x` with `φ = |dT/dS|·x/(μT)` — turns the loop's fixed point from an approximation of the
coupled optimum into the coupled optimum exactly.

`φ ≡ 1` for M/M/1, which is why the issue went unnoticed: it is invisible on the station type
most of the test suite uses. It is **not** fork-join-specific — every G/G/1 whose coefficients of
variation differ from M/M/1 has an S-dependent ζ too, and reaches a larger `φ` than any fork-join
does.

This came out of the prior question in
[`../forkjoin-coupled-vs-separate/`](../forkjoin-coupled-vs-separate/findings.md) — whether the
fork-join stations' `r*` should be coupled. That document owns the ray and KKT results this one
rests on, and they are not repeated here.

Equation numbers follow `../paper-map.md`: `eq 21` and `eq 22` mean `docs/analysis.pdf`. `φ` is
this directory's symbol, chosen because the working draft already uses `κ` for something else.
