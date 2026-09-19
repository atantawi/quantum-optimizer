# Which paper does "eq 21" mean?

**Every `eq N` in this repository refers to `docs/analysis.pdf`**, the snapshot committed in
`37a3a11` (2026-07-10, sha256 `bad90cdd…`). That file is tracked, so every such citation stays
resolvable from a clean checkout forever, with no external dependency. This is the convention;
it is not a description of what the current draft happens to say.

The working draft has moved on and **renumbered**. The substance behind each cited equation did
not change, but the numbers did, and not by a constant offset. A stale reference is therefore
not a dangling one — it silently resolves to a different, plausible-looking equation. See the
trap below.

## Crosswalk

Verified 2026-09-19 by extracting both PDFs and reading the surrounding text, not by assuming an
offset.

| this repo says | `docs/analysis.pdf` | working draft (2026-09-17) | what it is | cites |
|---|---|---|---|---|
| `eq 20` | (20) | **(17)** | closed-form optimal `S*` from the allocation theorem | 2 |
| `eq 21` | (21) | **(18)** | Step 1: per-iteration capacity allocation, `i ∈ [N̂], k+1 ∈ N` | 142 |
| `eq 22` | (22) | **(19)** | Step 3: invert the functional form for `ζ` | 58 |
| `eq 23` | (23) | **(21)** | `E[T_i(Ŝ_i)] = κ_i(Ŝ_i − γ_i) − p_{ι(i),i} κ_{ι(i)}(…)` | 1 |

Those four are the only equation numbers this repo cites. Everything else that looks like a
cross-reference — `spec §6.4`, `findings §7`, `finding 7` — points at **this repo's own**
documents under `docs/`, not at the paper.

Theorem numbering changed shape too: the draft moved from flat numbering to per-section, so
`Theorem 5` (the allocation theorem, which `eq 20`/`eq 21` come from) is now **`Theorem 4.1`**.

### The trap

The offset is `−3` for eq 20/21/22 but `−2` for eq 23, so there is no single shift to apply. And
both numbers this repo cites most are *live* in the new draft with different content:

- new **(21)** is `E[T_i(Ŝ_i)] = κ_i(…)` — unrelated to allocation.
- new **(22)** is an `S_i` formula in *exactly* the same shape as eq 21/18, but it is the
  fixed-point system stated with `ζ_i(Ŝ)` as a function rather than the iterate `ζ^(k)`.

So reading this repo's `eq 22` (the `ζ` inversion) against the new draft lands on an allocation
equation that looks right. That is why the convention is pinned to the committed snapshot rather
than "the latest PDF".

## Why not just renumber everything

1. **The draft is unpublished and still moving.** Renumbering 203 citations now buys a
   consistency that the next revision breaks. The numbers are only worth chasing once they
   freeze.
2. **`docs/analysis.pdf` is committed.** A citation against it is reproducible from the repo
   alone; a citation against a file in a local draft directory is not.
3. **Much of the corpus is a dated record, not live prose.** 71 of the citations are in
   `docs/superpowers/plans/`, `specs/` and `handoff/` — documents that say what was believed and
   decided on a particular date. Rewriting them to match a later paper would falsify the record.

## The convention, stated

- **Existing citations: leave them.** They mean `docs/analysis.pdf`, as recorded above.
- **New documents:** keep citing the pinned numbering, and name the equation the first time it
  appears in a document, e.g. *"eq 21 (the allocation rule)"*. The name survives renumbering;
  the number locates it in the committed PDF.
- **Never** cite a number read off a newer PDF without saying so — that is how
  `docs/direct-r-star-review` ended up citing `eq (20)` and `Theorem 4.2` in the same document,
  mixing the two numberings.
- **When the paper is submitted or published**, that is the moment to do one sweep: re-snapshot
  the PDF, update this table, and decide then whether to renumber live code and README (leaving
  the dated records under `docs/superpowers/` alone).

## Symbol collision

The working draft uses `κ_i` for the sojourn-time functional form (new eq 21, 23, 24). Any new
analysis in this repo that needs a symbol should avoid `κ`; `docs/slope-calibrated-zeta/` uses
`φ` for its slope-correction factor for exactly this reason.
