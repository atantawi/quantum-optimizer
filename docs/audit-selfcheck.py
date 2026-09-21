"""Self-check for `audit_split`, the optimum-confirming helper both probes rely on.

`audit_split` is what lets docs/forkjoin-coupled-vs-separate/ and docs/slope-calibrated-zeta/
claim their reference optimum is independently confirmed, so it is the one piece of either
probe whose own correctness is load-bearing. It is also duplicated across the two files,
which is a second way for it to go wrong. This script checks both.

Run from the repo root:

    python docs/audit-selfcheck.py        # exits non-zero if any check fails

**1. Each assertion must fire when its premise is broken.** The original version of this
audit compared `polish`'s result against the objective `polish` had been initialized from, so
it held by construction and confirmed nothing. Both assertions are therefore mutation-tested
in both directions, and each mutation is followed by a restore that must pass again -- without
the restore, a harness that had simply broken would look like a test that works.

**2. The two copies must not have drifted.** They have already drifted once, and in the
harder-to-see way: the fix that made the perturbed-start assertion two-sided updated the
docstring documenting it in only one file, because the two were worded differently enough
that a textual replacement matched one and skipped the other. So the code is compared with
docstrings removed -- the prose is free to differ per document -- and the docstrings are
compared on the NUMBERS they quote, which is where a stale one betrays itself. Comparing code
alone would have missed the skew that prompted this file.
"""

import ast
import difflib
import importlib.util
import re
import sys

PROBES = (("coupled", "docs/forkjoin-coupled-vs-separate/probe.py"),
          ("slope", "docs/slope-calibrated-zeta/probe.py"))
SHARED = ("audit_split", "polish")
"""Helpers duplicated across both probes, whose CODE must stay identical."""

DOCUMENTS_MEASUREMENTS = ("audit_split",)
"""The subset whose docstring states measured results, so its FIGURES must agree too.

`polish` is excluded deliberately. Its two docstrings differ in how much design rationale
they carry -- one explains why the improvement threshold is relative, the other does not --
and that is prose depth, not a stale claim. Holding it to figure-equality would make this
check fire on a legitimate difference, and a check that cries wolf gets switched off. The
tolerances in `audit_split`'s docstring are different in kind: they are assertions about data,
quoted to justify the constants in the code below them, so a copy quoting different ones is
either out of date or describing a different experiment.
"""


def load(name, path):
    """Import a probe for its helpers. Safe because each guards its own `main()`."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_of(path, func):
    """The function's body as normalized source, docstring removed.

    `ast.unparse` discards formatting, so this compares what the two copies DO rather than
    how they are laid out or worded -- the whole point, since the docstrings are expected to
    differ and the code is not.
    """
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]          # drop the docstring
            if not body:
                raise AssertionError(f"{func} in {path} has no body besides its docstring")
            return "\n".join(ast.unparse(stmt) for stmt in body)
    raise AssertionError(f"{func} not found in {path}")


def figures_in_docstring(path, func):
    """The numeric literals a function's docstring quotes.

    The two copies word their docstrings differently on purpose, but they document the SAME
    measured tolerances and the same sample size, so those figures must agree. This is the
    check that catches a docstring left behind by a change to the code it describes; the
    body comparison above cannot, since it strips docstrings entirely.
    """
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            doc = ast.get_docstring(node) or ""
            return set(re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", doc))
    raise AssertionError(f"{func} not found in {path}")


def fixture(name, module):
    """A network, its water-filled optimum, and that optimum's objective."""
    if name == "coupled":
        specs, stations = module.BASELINE, module.build(module.BASELINE)
    else:
        specs, stations = module.NET_MIXED_COV, module.build(module.NET_MIXED_COV, blend=0.0)
    values = module.vals(specs)
    budget = 1.5 * sum(s.min_spend for s in stations)
    _, spends = module.split(values, budget)
    obj = sum(v.w * v.T(x) for v, x in zip(values, spends))
    return values, spends, budget, obj


def expect_pass(module, values, spends, budget, obj, label):
    try:
        module.audit_split(values, spends, budget, obj)
    except AssertionError as exc:
        return f"{label}: expected to pass, but fired: {exc}"
    return None


def expect_fire(module, values, spends, budget, obj, label, want):
    """The assertion must fire, AND its message must name the right failure mode."""
    try:
        module.audit_split(values, spends, budget, obj)
    except AssertionError as exc:
        if want not in str(exc):
            return f"{label}: fired, but the message did not mention {want!r}: {exc}"
        return None
    return f"{label}: NOT DETECTED -- the assertion is vacuous in this direction"


def check_mutations(name, module):
    """Break each premise in turn; each must be caught, and the restore must pass."""
    failures = []
    values, spends, budget, obj = fixture(name, module)
    real = module.polish

    failures.append(expect_pass(module, values, spends, budget, obj, "unmutated"))

    # A suboptimal "answer": descent from it finds something better, so assertion 1 fires.
    worse = list(spends)
    shift = 0.2 * (spends[0] - values[0].floor)
    worse[0] -= shift
    worse[1] += shift
    worse_obj = sum(v.w * v.T(x) for v, x in zip(values, worse))
    failures.append(expect_fire(module, values, worse, budget, worse_obj,
                                "suboptimal answer", "did not find the optimum"))

    # A descent that does nothing: the perturbed start never recovers, so assertion 2 fires
    # HIGH. This is the mutation the ORIGINAL audit could not detect at all.
    module.polish = lambda items, sp, B: sum(it.w * it.T(x) for it, x in zip(items, sp))
    failures.append(expect_fire(module, values, spends, budget, obj,
                                "descent that does nothing", "did not descend"))
    module.polish = real

    # A perturbed descent that reports something better than the claimed optimum, which
    # refutes it. Only the SECOND call is faked, so assertion 1 still sees the true value and
    # this isolates assertion 2's lower bound.
    calls = {"n": 0}

    def better_basin(items, sp, B):
        calls["n"] += 1
        return real(items, sp, B) if calls["n"] == 1 else obj * 0.99

    module.polish = better_basin
    failures.append(expect_fire(module, values, spends, budget, obj,
                                "perturbed descent finds a better basin", "BETTER basin"))
    module.polish = real

    # The restore must pass. Without this a broken harness reads as a working test.
    failures.append(expect_pass(module, values, spends, budget, obj, "restored"))
    return [f for f in failures if f]


def main():
    failures = []

    print("1a. the two copies of the shared helpers have the same code")
    for func in SHARED:
        bodies = {path: code_of(path, func) for _, path in PROBES}
        if len(set(bodies.values())) == 1:
            print(f"    OK       {func}() is identical in both probes")
        else:
            failures.append(f"{func}() code has drifted between the two probes")
            print(f"    DRIFTED  {func}()")
            a, b = (bodies[p] for _, p in PROBES)
            for line in difflib.unified_diff(a.split("\n"), b.split("\n"),
                                             PROBES[0][1], PROBES[1][1], lineterm=""):
                print(f"             {line}")

    print("\n1b. and the docstrings that state measurements quote the same figures")
    for func in DOCUMENTS_MEASUREMENTS:
        quoted = {path: figures_in_docstring(path, func) for _, path in PROBES}
        (pa, a), (pb, b) = quoted.items()
        if a == b:
            print(f"    OK       {func}() docstrings agree on {sorted(a)}")
        else:
            failures.append(f"{func}() docstrings quote different figures")
            print(f"    SKEWED   {func}()")
            print(f"             only in {pa}: {sorted(a - b)}")
            print(f"             only in {pb}: {sorted(b - a)}")

    print("\n2. every assertion fires when its premise is broken")
    for name, path in PROBES:
        print(f"   {name} ({path})")
        problems = check_mutations(name, load(name, path))
        for problem in problems:
            failures.append(f"{name}: {problem}")
            print(f"      FAIL   {problem}")
        if not problems:
            print("      OK     all four mutations caught, restore passes")

    print()
    if failures:
        print(f"FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED: the audit can fail, and the two copies have not drifted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
