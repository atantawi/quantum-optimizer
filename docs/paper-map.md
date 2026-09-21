# Which paper does "eq 21" mean?

**Every `eq N` in this repository refers to `docs/analysis.pdf`**, the snapshot committed in
`37a3a11` (2026-07-10, sha256 `bad90cdd…`). That file is tracked, so every such citation stays
resolvable from a clean checkout forever, with no external dependency. This is the convention; it
is not a description of what the current draft happens to say.

One existing citation does **not** satisfy it. See [Known exception](#known-exception).

The working draft has moved on and **renumbered**. The substance behind each cited equation did not
change, but the numbers did, and not by a constant offset. A stale reference is therefore not a
dangling one — it silently resolves to a different, plausible-looking equation. See
[The trap](#the-trap).

## Inventory

Counted against `1adb2d8` — the last commit before this document existed, which is why the tree is
named **inside** the commands. Run them as written from any checkout and they reproduce the figures
below exactly, however much the repository has moved on since:

```sh
# single-number citations, by number
git grep -ohiE "eq(uation)?s?\.? *\(?[0-9]{1,2}\)?([–-][0-9]{1,2})?" 1adb2d8 \
  | grep -vE "[–-][0-9]" | sed -E 's/.*[^0-9]([0-9]+)\)?$/\1/' | sort -n | uniq -c

# range-style citations
git grep -nohiE "eq(uation)?s?\.? *\(?[0-9]{1,2}\)?[–-][0-9]{1,2}" 1adb2d8 | sort | uniq -c
```

**200 single-number citations and 3 ranges, 203 in total.** Only two equation numbers are ever
cited on their own:

| citation | occurrences | `docs/analysis.pdf` | working draft (2026-09-17) | what it is |
|---|---|---|---|---|
| `eq 21` | 142 | (21) | **(18)** | Step 1: per-iteration capacity allocation, `i ∈ [N̂], k+1 ∈ N` |
| `eq 22` | 58 | (22) | **(19)** | Step 3: invert the functional form for `ζ` |

and three citations name a range:

| citation | where | `docs/analysis.pdf` | working draft | matches? |
|---|---|---|---|---|
| `eq 20–21` | `docs/optimizer-brainstorm-summary.md:33` | (20)–(21) | **(17)–(18)** | yes — `Theorem 5` is the allocation theorem, (20) its `S*` |
| `eqs 20–22` | `docs/superpowers/specs/2026-07-10-optimizer-design.md:16` | (20)–(22) | **(17)–(19)** | yes — Steps 0–5 |
| `eqs 23–27` | `docs/optimizer-brainstorm-summary.md:62` | (23)–(27) | *not mappable* | **no** — see below |

Drop the `1adb2d8` and the totals come out higher on any tree that contains this document, because
its tables and prose are themselves citations of `eq 21`, `eq 22`, `eq 20` and `eq 23` — this very
sentence adds four. That is not a caveat to keep updated but the reason the tree is pinned: the
inventory describes the corpus it documents, not whatever tree happens to contain it.

Everything else that looks like a cross-reference — `spec §6.4`, `findings §7`, `finding 7` —
points at **this repo's own** documents under `docs/`, not at the paper.

Theorem numbering changed shape too: the draft moved from flat numbering to per-section, so
`Theorem 5` (the allocation theorem, which `eq 20–21` names) is now **`Theorem 4.1`**.

## Known exception

`docs/optimizer-brainstorm-summary.md:62` reads:

> Paper's full routing model (eqs 23–27) is out of scope.

This does not resolve against `docs/analysis.pdf`:

- (23)–(27) there are the `κ`-substitution machinery — a sojourn-time expression, the
  `(OPTRCA:FF:κ)` problem, the `S*` fixed-point system, `ẑ_i(Ŝ_i)`, and the definition of
  `ζ_i(ẑ_i)`. None of them is a routing model.
- The string "routing" does not appear in `docs/analysis.pdf` at all — nor in the working draft.

So this citation was written against something other than the committed snapshot. Both files
entered the repo in the same commit (`37a3a11`), so it is not a case of the PDF being replaced
under the text.

**Unresolved, deliberately.** Fixing it means knowing what "the paper's full routing model" was
meant to point at, which the repo does not record. The claim it supports — that deriving `γ` from
network topology is out of scope — was in any case superseded by `Network.solve_traffic`. Left for
whoever knows the intent; flagged here so the convention above is not read as covering it.

This range also demonstrates the renumbering problem concretely: `(23)` → `(21)` and `(25)` → `(22)`
in the working draft, while `(24)` — the `(OPTRCA:FF:κ)` display — is unnumbered there. A range that
was contiguous in one document is neither contiguous nor complete in the other.

## The trap

The offset is `−3` for eq 20/21/22 but `−2` for eq 23, so there is no single shift to apply. And
both numbers this repo cites most are *live* in the new draft with different content:

- new **(21)** is `E[T_i(Ŝ_i)] = κ_i(Ŝ_i − γ_i) − p_{ι(i),i} κ_{ι(i)}(…)` — unrelated to allocation.
- new **(22)** is an `S_i` formula in *exactly* the same shape as eq 21/18, but it is the
  fixed-point system stated with `ζ_i(Ŝ)` as a function rather than the iterate `ζ^(k)`.

So reading this repo's `eq 22` (the `ζ` inversion) against the new draft lands on an allocation
equation that looks right. That is why the convention is pinned to the committed snapshot rather
than "the latest PDF".

## Why not just renumber everything

1. **The draft is unpublished and still moving.** Renumbering 203 citations now buys a consistency
   that the next revision breaks. The numbers are only worth chasing once they freeze.
2. **`docs/analysis.pdf` is committed.** A citation against it is reproducible from the repo alone;
   a citation against a file in a local draft directory is not.
3. **Much of the corpus is a dated record, not live prose.** 80 of the citations are under
   `docs/superpowers/` — plans, specs and handoff notes that say what was believed and decided on a
   particular date. Rewriting them to match a later paper would falsify the record.

## The convention, stated

- **Existing citations: leave them.** They mean `docs/analysis.pdf`, as recorded above, with the one
  exception noted.
- **New documents:** keep citing the pinned numbering, and name the equation the first time it
  appears in a document, e.g. *"eq 21 (the allocation rule)"*. The name survives renumbering; the
  number locates it in the committed PDF.
- **Never** cite a number read off a newer PDF without saying so, and never cite a *range* without
  checking that it still describes the same equations — the `eqs 23–27` exception above is what that
  failure looks like once the referent has moved.
- **When the paper is submitted or published**, that is the moment to do one sweep: re-snapshot the
  PDF, update this table, and decide then whether to renumber live code and README (leaving the
  dated records under `docs/superpowers/` alone).

## Symbol collision

The working draft uses `κ_i` for the sojourn-time functional form (new eq 21, 23, 24). New analysis
in this repo that needs a symbol for something else should avoid `κ`;
[`slope-calibrated-zeta/`](slope-calibrated-zeta/findings.md) uses `φ` for its slope-correction
factor for exactly that reason.
