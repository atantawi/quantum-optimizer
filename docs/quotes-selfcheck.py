"""Check that every line a findings document quotes from its probe is verbatim.

Both analysis documents under docs/ rest on one discipline: every number in the prose comes
from the committed `probe-output.txt` beside it, so a reader can check any figure without
running anything. This enforces that discipline, which four review rounds of hand-checking
did not.

Run from the repo root:

    python docs/quotes-selfcheck.py        # exits non-zero if any quoted line has drifted

**Why this exists.** The hand-rolled version was a shell loop over a list of `case` patterns
(`*"%"*`, `*"C/floor"*`, ...). Those matched numeric rows and column headers, so "every quoted
table checked verbatim" was true of what it looked at -- and it never looked at table CAPTIONS.
A caption quoted with one space where the probe prints two survived four reviews and surfaced
only when a pattern was widened by accident. A checker whose coverage is an enumeration of
patterns silently redefines "checked" as "matched", so this one takes every line and decides
nothing by pattern.

**How fences are classified.** A findings document fences both probe output and formulas, and
only the former should be compared. A fence is treated as probe output when at least one of
its lines occurs in `probe-output.txt`; formula and shell fences match nothing and are skipped.

That heuristic has one blind spot, and it is reported rather than hidden: a fence in which
EVERY line had drifted would match nothing and be classified as a formula. So the summary
prints how many fences were skipped and the first line of each, which is what makes the
blind spot auditable -- if a skipped fence is not obviously a formula, look at it.

**What this does NOT cover.** Only FENCED quotes. docs/forkjoin-s2-policy/findings.md also
says "Every number below comes from that file", but reformats its figures into markdown tables
and prose, where a line-for-line comparison cannot apply -- spot-checking three of its numbers
found all three present, by substring. A document in that shape is reported as NOT COVERED
rather than as a clean zero, because a zero that reads like a pass is the same bug this file
exists to fix. Extending to numeric-substring tracing would cover it, at the cost of false
positives on every year, equation number and hand-computed percentage in the prose; that
trade has not been made.

Companion to docs/audit-selfcheck.py, which checks a different thing: that the probes'
optimum-confirming assertions can actually fail.
"""

import glob
import os
import sys

# Enumerated by discovery rather than by hand, so a document added later is covered without
# editing this file -- any docs/*/findings.md with a probe-output.txt beside it.
DOCS = sorted(os.path.dirname(p) for p in glob.glob("docs/*/findings.md")
              if os.path.exists(os.path.join(os.path.dirname(p), "probe-output.txt")))


def fences(path):
    """Every fenced block in a markdown file, as lists of lines."""
    out, cur, inside = [], [], False
    for line in open(path).read().split("\n"):
        if line.startswith("```"):
            if inside:
                out.append(cur)
                cur = []
            inside = not inside
            continue
        if inside:
            cur.append(line)
    return out


def check(doc):
    """Returns (available_lines, lines_checked, failures, skipped_fences)."""
    available = set(l.rstrip() for l in
                    open(os.path.join(doc, "probe-output.txt")).read().split("\n"))
    failures, checked, skipped = [], 0, []
    for block in fences(os.path.join(doc, "findings.md")):
        live = [l for l in block if l.strip()]
        if not live:
            continue
        if not any(l.rstrip() in available for l in live):
            skipped.append(live[0])          # formula or shell fence -- reported, not hidden
            continue
        for line in live:
            checked += 1
            if line.rstrip() not in available:
                failures.append(line)
    return available, checked, failures, skipped


def main():
    if not DOCS:
        print("no docs/*/findings.md with a probe-output.txt beside it -- nothing to check")
        return 1                      # a silent pass here would be the same bug again
    total_bad, uncovered = 0, []
    for doc in DOCS:
        available, checked, failures, skipped = check(doc)
        print(f"{doc}")
        if checked == 0:
            uncovered.append(doc)
            print("   NOT COVERED -- declares a probe output but fences none of it; see"
                  " \"What this does NOT cover\" above")
            continue
        print(f"   {checked} quoted output lines checked, {len(failures)} not verbatim,"
              f" {len(skipped)} fence(s) skipped as not-probe-output")
        for line in failures:
            total_bad += 1
            print(f"   NOT VERBATIM  {line!r}")
            stem = line.strip()[:22]
            for near in [a for a in available if stem and stem in a][:1]:
                print(f"     probe has   {near!r}")
        for skip in skipped:
            print(f"   skipped:      {skip.strip()[:72]!r}")
    print()
    if total_bad:
        print(f"FAILED: {total_bad} quoted line(s) no longer match the committed probe output.")
        return 1
    print("PASSED: every fenced quote matches the committed probe output.")
    print("Confirm the skipped fences above are formulas, not output whose every line drifted.")
    if uncovered:
        print(f"Not covered by this check: {', '.join(uncovered)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
