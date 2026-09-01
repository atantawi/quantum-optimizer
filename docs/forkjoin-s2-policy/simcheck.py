"""Simulated cross-check of the fork-join r* policies -- issue #10's last open item.

Everything behind the r* family's headline gains was computed from `t_ul`, the fork-join
sojourn approximation, which is documented exact only at m1 == m2 and whose
heterogeneous-server bias was measured only at r = 4 (spec section 7: ~0.15% mean gap,
~1.1% worst row). The 24.6% `classical_dominant` gain is ~20x that and is structural -- it
comes from `alloc_cost`, not from `t_ul`. The two ~2% gains are only ~2x the worst per-row
gap, so findings section 9 calls them suggestive rather than established. Those two are
what this probe targets.

Nothing in qopt/ changes. `ForkJoinStation.sim_node` already emits the ray the station is
actually on (`_anchor` folds r_star into the effective mu and r), and
`build_qcsc_network(workload, r_star=...)` already selects the policy, so the whole
cross-check is a grid of existing runs. Topology, rates, costs and budget come from the
real example module, so they cannot drift from examples/qcsc_network.py.

DESIGN

  * One absolute budget for every cell -- `shared_budget()`, evaluated at the default ray.
    Same money for all three policies, exactly as findings section 7 did. It is feasible
    under all of them (the tuned/paper floors are the lower ones: 6.8400 / 5.8275 /
    2.7900).
  * 3 workloads x 3 policies x 5 base seeds, PAIRED BY SEED. Within a seed the three
    policies share the base seed, so `seed_policy="fixed"` gives common random numbers and
    the POLICY DIFFERENCE is the low-variance quantity. The spread of that paired
    difference across seeds is the primary interval -- more trustworthy than anything
    propagated from per-station CIs, whose errors are correlated within a run.
  * Four of the five seeds already carry a published model-error baseline (spec section 7's
    replication table), so the bias statistics here are comparable to it.

WHAT TO EXPECT, PRE-REGISTERED so the result can be read rather than rationalized

  * quantum_dominant / classical_dominant: their fork-joins are built at r = 4 and the
    tuned rays (2.316 / 1.000) are CLOSER to homogeneity, where t_ul is more trustworthy
    than at the baseline. Correcting T at the baseline downward SHRINKS the gain, so a
    shrink is the expected direction, not a refutation.
  * balanced: hardware r = 1, so the INCUMBENT already sits at m1 == m2 where t_ul is
    exact, and tuning moves AWAY to 1.447. Its gain can move either way. This is the one
    cell that can genuinely overturn a claim.
  * r_star is not noise-free here. It is a function of the station spend, which descends
    from a measured E[T] (+/-2% noise moves it ~6e-4 relative), so it is reported as an
    interval across seeds rather than as a point.

Run (needs a live qsim-service; see docs/qcsc-example/README.md for the version floor):

    QOPT_QSIM_URL=http://localhost:8080 python docs/forkjoin-s2-policy/simcheck.py \
        > docs/forkjoin-s2-policy/simcheck-output.txt

Tighten one inconclusive cell without editing anything:

    ... simcheck.py --workloads balanced --precision 0.01 --min-samples 200000 \
        --max-wallclock 600
"""

import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qopt import (                                          # noqa: E402
    Optimizer,
    QsimClient,
    R_STAR_EQUAL_RATE,
    R_STAR_INVARIANT_R,
    R_STAR_TUNED,
    SimulationAnalyzer,
    min_feasible_budget,
)
import examples.qcsc_network as qn                           # noqa: E402


POLICIES = (
    ("invariant-r", R_STAR_INVARIANT_R),
    ("equal-rate", R_STAR_EQUAL_RATE),
    ("tuned", R_STAR_TUNED),
)
BASELINE = "invariant-r"
"""Every gain in this probe is measured against qopt's incumbent ray, r* = r."""

SEEDS = (20260729, 8675309, 31415926, 2718281, 16180339)
"""The committed example seed, then spec section 7's three replication seeds, then one new.

Four of the five therefore have a published analytic-vs-simulated baseline to compare the
bias statistics against. 16180339 is added only to get a fifth degree of freedom.
"""

SPEC7_SEEDS = (20260729, 8675309, 31415926, 2718281)
"""The four seeds spec section 7 already published, at the DEFAULT ray.

These are in SEEDS deliberately, but they are NOT new evidence at `invariant-r`: same ray,
same budget, same stopping rule, and the service is deterministic given a seed, so those
cells reproduce spec section 7's rows to the last bit. That makes them a PIPELINE WITNESS --
proof this harness is running the same measurement -- and nothing more. Only seeds outside
this set carry new information about the model bias at the default ray. Every other policy is
new at every seed, because spec section 7 predates `r_star`.

This distinction cost a wrong claim: the first write-up read the pooled 136/210 negative rows
as replicating spec section 7 "a fifth, sixth and seventh time", when 115/168 of them ARE its
published pool and the single new seed is 21/42, exactly chance.
"""

COMMITTED_SEED = 20260729
"""The seed docs/qcsc-example/live-run.log was captured at.

Full per-station tables are printed for this seed only, so the log stays readable while the
one cell that is directly comparable to the committed run is shown in full.
"""

# Regression anchors. The analytic column must land on the numbers PR #13 recorded, or the
# simulator time this probe is about to spend is being spent on a different computation.
# Checked in `main` BEFORE the first POST, off analytic passes that cost nothing.
# From tests/test_example_qcsc.py: EXPECTED_OBJECTIVE, FINDINGS_SECTION_7, TUNED.
ANALYTIC_ANCHOR = {
    #                      invariant-r  equal-rate    tuned                 tuned ray
    "balanced":           (6.401440,    6.401440,     6.2494291783037355,   1.447382318551769),
    "quantum_dominant":   (4.528844,    4.776428,     4.431691350142377,    2.316117567446687),
    "classical_dominant": (3.463677,    2.613335,     2.6133350927675547,   1.0000000000000002),
}
PUBLISHED_DP = 5e-7
"""Half a unit in the last place of the 6-decimal figures published in findings."""

T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
         6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
         21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
         26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
"""Two-sided 95% Student-t quantiles by degrees of freedom.

Hardcoded because qopt takes no runtime dependencies (`statistics` has no t
distribution).

Tabulated to df 30 rather than to the 4 this probe's default seed count needs, because the
fallback matters: `t` approaches 1.96 from ABOVE, so using 1.96 at small df makes the
interval too NARROW, and this interval is what the headline verdict is read off. Above df 30
`_t_975` falls back to 1.96, which is still ~3.9% low at df 31 and only drops under 2% around
df 60 -- so a run with more than 31 paired observations reports a slightly optimistic
interval. Extend the table rather than widen the caveat if that ever matters.
"""


def _t_975(df):
    """Two-sided 95% t quantile, erring wide rather than narrow off the end of the table."""
    if df in T_975:
        return T_975[df]
    return 1.96 if df > max(T_975) else T_975[min(T_975)]


# --------------------------------------------------------------------------------------
# One cell of the grid
# --------------------------------------------------------------------------------------

class Cell:
    """One (workload, policy, seed) measurement, plus its own analytic control."""

    def __init__(self, workload, policy, seed, analytic, simulated, rays, elapsed,
                 net_a, net_s):
        self.workload = workload
        self.policy = policy
        self.seed = seed
        self.analytic = analytic
        self.simulated = simulated
        self.rays = rays
        self.elapsed = elapsed
        # Its OWN networks. A tuned station is left on the ray its run converged to, so
        # borrowing another cell's network would report another seed's answer.
        self.net_a = net_a
        self.net_s = net_s

    @property
    def measured(self):
        return self.simulated.objective

    @property
    def predicted(self):
        return self.analytic.objective


def objective_half_widths(result):
    """(conservative, independent, n_missing) half-widths on the measured objective.

    The objective is `sum_i omega_i * E[T_i]` with omega_i = 1 throughout this example
    (qopt/optimizer.py:306), so it is the plain sum of the 14 station sojourn times and its
    uncertainty is the sum of theirs.

    Two figures, because the truth is between them and neither alone is honest:

      * CONSERVATIVE, `sum_i h_i`, assumes the station errors are perfectly correlated.
      * INDEPENDENT, `sqrt(sum_i h_i^2)`, assumes they are not.

    Stations within one run share a single simulation and the topology makes four of them
    analytically identical (spec section 7), so the errors ARE correlated and the
    conservative figure is the defensible one. The independent figure is printed only to
    show how much of the width is that assumption.

    Neither is the primary interval. The paired spread across seeds is -- see
    `paired_interval` -- because it measures the variability instead of assuming a
    correlation structure.
    """
    if result.sojourn_ci is None:
        return None, None, len(result.sojourn_times)
    total = 0.0
    squares = 0.0
    missing = 0
    for entry in result.sojourn_ci:
        if entry is None:
            missing += 1
            continue
        lower, upper = entry
        half = 0.5 * (upper - lower)
        total += half
        squares += half * half
    return total, math.sqrt(squares), missing


def run_cell(workload, policy_name, r_star, seed, budget, client):
    """Analytic then simulated pass for one cell, each on its own fresh Network.

    Fresh networks per pass for the reason `run_simulated` gives in the example
    (examples/qcsc_network.py:373) and one more that is new here: a tuned station MUTATES
    its r_star during a run (#14), so a station reused across passes would start the second
    one on the first one's converged ray. `Optimizer.run` calls `reset_policy()` itself, so
    reuse would in fact be correct -- building fresh is belt-and-braces, and it keeps the
    two `rays` readings unambiguous.
    """
    net_a = qn.build_qcsc_network(workload, r_star=r_star)
    analytic = Optimizer(net_a, budget=budget).run()

    net_s = qn.build_qcsc_network(workload, r_star=r_star)
    t0 = time.time()
    simulated = Optimizer(
        net_s, budget=budget,
        analyzer=SimulationAnalyzer(net_s, client, seed=seed),
    ).run()
    elapsed = time.time() - t0

    # `Result` carries no r_star field, so the converged ray is read off the stations --
    # which is where a tuned run leaves it (tests/test_example_qcsc.py:497).
    rays = {
        "analytic": [st.r_star for st in net_a.stations if st.name.startswith("fj")],
        "simulated": [st.r_star for st in net_s.stations if st.name.startswith("fj")],
    }
    return Cell(workload, policy_name, seed, analytic, simulated, rays, elapsed,
                net_a, net_s)


def check_analytic_anchors(cells_by_key):
    """Refuse to report anything if the analytic column has drifted from PR #13's numbers.

    A gate rather than a note: every claim here is a comparison against the analytic
    prediction, so if the prediction moved, the comparison is meaningless.
    """
    failures = []
    for workload, anchors in ANALYTIC_ANCHOR.items():
        expected = dict(zip(("invariant-r", "equal-rate", "tuned"), anchors[:3]))
        for policy, want in expected.items():
            key = (workload, policy)
            if key not in cells_by_key:
                continue
            got = cells_by_key[key].predicted
            tol = PUBLISHED_DP if policy != "tuned" else abs(want) * 1e-12
            if abs(got - want) > tol:
                failures.append(f"{workload}/{policy}: analytic {got!r} != recorded {want!r}")
        if ("tuned" in {p for w, p in cells_by_key if w == workload}):
            want_ray = anchors[3]
            got_ray = cells_by_key[(workload, "tuned")].rays["analytic"][0]
            if abs(got_ray - want_ray) > abs(want_ray) * 1e-12:
                failures.append(
                    f"{workload}/tuned ray: analytic {got_ray!r} != recorded {want_ray!r}")
    return failures


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def paired_interval(differences):
    """(mean, half-width, n) for a paired sample, two-sided 95% Student-t.

    The differences are paired by seed: within one seed the policies were simulated under
    common random numbers, so `d_s = obj_baseline(s) - obj_policy(s)` removes the sample
    path the two share. n = 1 has no interval.
    """
    n = len(differences)
    mean = statistics.fmean(differences)
    if n < 2:
        return mean, None, n
    sem = statistics.stdev(differences) / math.sqrt(n)
    return mean, _t_975(n - 1) * sem, n


def gap_stats(analytic, simulated):
    """Per-station (gap %, flagged?) plus the aggregate spec-section-7 statistics."""
    rows = []
    for i, (a, s) in enumerate(zip(analytic.sojourn_times, simulated.sojourn_times)):
        entry = simulated.sojourn_ci[i] if simulated.sojourn_ci is not None else None
        flagged = None
        if entry is not None:
            lower, upper = entry
            flagged = abs(s - a) > 0.5 * (upper - lower)
        rows.append((100.0 * (s - a) / a, flagged))
    pcts = [p for p, _ in rows]
    return {
        "rows": rows,
        "n": len(rows),
        "negative": sum(1 for p in pcts if p < 0.0),
        "mean_pct": statistics.fmean(pcts),
        "worst_pct": max(pcts, key=abs),
        "flagged": sum(1 for _, f in rows if f),
        "no_ci": sum(1 for _, f in rows if f is None),
    }


def sign_test_p(negative, n):
    """Two-sided exact binomial p for `negative` of `n` under p = 1/2.

    The same test spec section 7 applies to its 42 rows, and it carries the same two
    caveats there: the rows are not independent (stations share one run, and four are
    analytically identical), and near-zero gaps are counted as signs rather than ties. So
    this p is optimistic and is reported as a direction, not as a level.
    """
    k = min(negative, n - negative)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    # Integers throughout: `2.0 * tail` and `2.0 ** n` both overflow float above n ~ 1024,
    # which would kill the run in its final report after every pass had been paid for.
    return min(1.0, (2 * tail) / (2 ** n))


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def verify_service(client):
    """Measure that the service honours `minSamples` and writes `alpha`, not its complement.

    A build predating qsim-service #11 and #13 produces plausible-looking garbage -- the
    sample floor has no effect and the confidence intervals come back at the wrong level --
    while still reporting success throughout. A tag cannot tell you which build you have
    (0.1.0 and 0.2.0 were both cut from the same commit), so this is measured. Printed here
    rather than checked in a shell so the committed output carries its own provenance.

    The probe network is a bare M/M/1 built through qopt itself, so this needs no fixture.
    """
    from qopt import GG1Station, Network, Route
    from qopt.qsim.spec import build_request

    st = GG1Station.mm1(mu=3.0, c=1.0, name="mm1")
    net = Network([st], [Route("src", "mm1"), Route("mm1", "snk")], arrival_rate=1.0)

    print("\n\n=== 0. SERVICE PROVENANCE, measured (qsim-service #11 and #13) ===")
    print("\n  An M/M/1 at three sample floors. `samplesAnalyzed` must rise with the floor")
    print("  (#11: minSamples was validated and then never written into the JSIMG document)")
    print("  and `alpha` must come back as the significance level, not its complement (#13).")
    print(f"\n  {'minSamples':>11s} {'samplesAnalyzed':>16s} {'alpha':>7s} "
          f"{'precision':>10s} {'completed':>10s}")
    floors = (1000, 100000, 400000)
    analyzed, alphas = [], []
    for floor in floors:
        stopping = dict(client.stopping, minSamples=floor)
        response = client.post_simulate(
            build_request(net, [1.0], seed=COMMITTED_SEED, stopping=stopping))
        entry = [m for m in response["measures"]
                 if m["type"] == "response-time" and m["station"] == "mm1"][0]
        analyzed.append(entry["samplesAnalyzed"])
        alphas.append(entry["alpha"])
        print(f"  {floor:11d} {entry['samplesAnalyzed']:16d} {entry['alpha']:7.2f} "
              f"{entry['precision']:10.5f} {str(response.get('completed')):>10s}")

    # A GATE, not a report. Printing this and continuing is exactly the failure the two fixes
    # were about: the numbers look plausible either way, and a bad build lands a confident
    # report in the committed output.
    problems = []
    if not all(b > a for a, b in zip(analyzed, analyzed[1:])):
        problems.append(f"samplesAnalyzed does not rise with minSamples ({analyzed}) -- the "
                        f"sample floor is not reaching the engine (qsim-service #11)")
    for floor, seen in zip(floors, analyzed):
        if seen < floor:
            problems.append(f"minSamples={floor} returned only {seen} samples")
    want = client.stopping["alpha"]
    if any(abs(a - want) > 1e-12 for a in alphas):
        problems.append(f"alpha came back {alphas}, requested {want} -- the service may be "
                        f"writing the complement (qsim-service #13)")
    if problems:
        print("\n  PROVENANCE GATE FAILED. This build corrupts results silently, so nothing")
        print("  measured against it is usable. No grid was run.")
        for line in problems:
            print(f"    - {line}")
        raise SystemExit(1)
    print("\n  provenance gate: minSamples binds and alpha round-trips.")
    print("\n  One `[Error] ... Attribute 'minSamples' is not allowed ...` line per /simulate")
    print("  on the service's stderr is EXPECTED and non-fatal: JMT's bundled schema never")
    print("  declared the attribute, but SimLoader reads it anyway.")


def print_conditions(args, budget, url):
    print("SIMULATED CROSS-CHECK of the fork-join r* policies (issue #10, last open item)")
    print(f"\nservice   {url}")
    print(f"stopping  {args.stopping}")
    print(f"budget    C = {budget:.6f}  (BUDGET_MULTIPLE = {qn.BUDGET_MULTIPLE:g} x the "
          f"balanced floor, at the default ray)")
    print(f"seeds     {', '.join(str(s) for s in args.seeds)}")
    print(f"grid      {len(args.workloads)} workloads x {len(args.policies)} policies "
          f"x {len(args.seeds)} seeds = "
          f"{len(args.workloads) * len(args.policies) * len(args.seeds)} simulated passes")
    print(f"baseline  {BASELINE} (qopt's incumbent ray r* = r); every gain is against it")
    print("\nA re-run at these seeds reproduces bit-identically -- the service is "
          "deterministic given a\nseed -- so a repeat confirms the pipeline and is NOT a "
          "second sample. Vary the seeds for that.")


def print_analytic_grid(cells_by_key, budget, workloads, policies):
    print("\n\n=== A. ANALYTIC GRID (no simulator; the predictions under test) ===")
    print(f"  {'workload':20s} {'policy':12s} {'floor':>7s} {'C/floor':>8s} "
          f"{'objective':>11s} {'gain %':>8s} {'fj ray':>9s} "
          f"{'cum QPU':>9s} {'cum GPU':>9s} {'cum CPU':>9s}")
    for workload in workloads:
        base = cells_by_key.get((workload, BASELINE))
        for policy, _ in policies:
            key = (workload, policy)
            if key not in cells_by_key:
                continue
            cell = cells_by_key[key]
            floor = min_feasible_budget(cell.net_a.stations)
            by_unit = qn.capacity_by_unit(cell.net_a, cell.analytic.capacities)
            gain = ("" if base is None
                    else f"{100.0 * (base.predicted - cell.predicted) / base.predicted:8.2f}")
            print(f"  {workload:20s} {policy:12s} {floor:7.4f} {budget / floor:8.2f} "
                  f"{cell.predicted:11.6f} {gain:>8s} "
                  f"{cell.rays['analytic'][0]:9.4f} "
                  f"{by_unit['qpu']:9.4f} {by_unit['gpu']:9.4f} {by_unit['cpu']:9.4f}")
    print("\n  Both fork-joins share gamma and rates, so fj_pp and fj_sp always land on the")
    print("  same ray -- three workloads are three data points, not six (findings 9).")
    print("  `floor` moves with the policy: the tuned/paper floors are the lower ones, which")
    print("  is why one shared C is feasible everywhere (implementation.md, min_spend).")


def print_cell_detail(cell, net_s, full):
    tag = f"{cell.workload} / {cell.policy} / seed {cell.seed}"
    cons, indep, missing = objective_half_widths(cell.simulated)
    stats = gap_stats(cell.analytic, cell.simulated)
    r = cell.simulated
    print(f"\n  CELL {tag}")
    print(f"    analytic {cell.predicted:.6f}   measured {cell.measured:.6f}   "
          f"delta {100.0 * (cell.measured - cell.predicted) / cell.predicted:+.3f}%")
    if cons is not None:
        print(f"    objective half-width: conservative +/-{cons:.6f} "
              f"({100.0 * cons / cell.measured:.3f}%), independent +/-{indep:.6f} "
              f"({100.0 * indep / cell.measured:.3f}%), {missing} station(s) without a CI")
    print(f"    ray analytic {cell.rays['analytic'][0]:.6f}   "
          f"simulated {cell.rays['simulated'][0]:.6f}")
    print(f"    stop_reason = {r.stop_reason}   iterations = {r.iterations}   "
          f"sim_calls = {r.sim_calls}   warm_start = {r.warm_start_iterations}   "
          f"converged = {r.converged}")
    print(f"    station gaps: {stats['negative']}/{stats['n']} negative, "
          f"mean {stats['mean_pct']:+.3f}%, worst {stats['worst_pct']:+.3f}%, "
          f"{stats['flagged']} over their own CI half-width")
    if r.degraded:
        print(f"    DEGRADED ({len(r.degraded)}):")
        for entry in r.degraded:
            print(f"      - {entry}")
    if full:
        qn.print_table(f"  FULL TABLE  ({tag})", net_s, r)
        qn.print_gaps(tag, net_s, cell.analytic, r)


def print_headline(cells, workloads, policies):
    """The result the issue is waiting for: does the analytic gain survive measurement?"""
    print("\n\n=== C. THE HEADLINE: measured gain vs analytic gain, paired by seed ===")
    print("\n  d_s = objective(invariant-r, seed s) - objective(policy, seed s), under common")
    print("  random numbers. Interval is two-sided 95% Student-t on the paired differences")
    print("  across seeds -- it measures the variability rather than assuming a correlation")
    print("  structure, which is why it and not the propagated half-width is the primary")
    print("  evidence.")
    print(f"\n  {'workload':20s} {'policy':12s} {'analytic %':>11s} {'measured %':>11s} "
          f"{'95% CI on measured %':>24s} {'n':>3s} {'verdict':>12s}")
    verdicts = {}
    inside, outside = set(), set()
    for workload in workloads:
        base_by_seed = {c.seed: c for c in cells
                        if c.workload == workload and c.policy == BASELINE}
        if not base_by_seed:
            continue
        base_predicted = next(iter(base_by_seed.values())).predicted
        for policy, _ in policies:
            if policy == BASELINE:
                continue
            paired = [(base_by_seed[c.seed], c) for c in cells
                      if c.workload == workload and c.policy == policy
                      and c.seed in base_by_seed]
            if not paired:
                continue
            analytic_pct = 100.0 * (base_predicted - paired[0][1].predicted) / base_predicted
            pcts = [100.0 * (b.measured - p.measured) / b.measured for b, p in paired]
            mean, half, n = paired_interval(pcts)
            if all(p == 0.0 for p in pcts):
                # findings 4: r = 1 makes the two incumbents coincide, so this is one
                # policy measured twice, not a comparison that failed to resolve.
                verdict, ci = "identical", "exactly 0"
            elif half is None:
                verdict, ci = "n=1", "--"
            else:
                lo, hi = mean - half, mean + half
                ci = f"({lo:+.3f}, {hi:+.3f})"
                if lo > 0.0:
                    verdict = "CONFIRMED" if analytic_pct > 0 else "SIGN FLIP"
                elif hi < 0.0:
                    verdict = "SIGN FLIP" if analytic_pct > 0 else "CONFIRMED"
                else:
                    verdict = "inconclusive"
            verdicts[(workload, policy)] = verdict
            if half is not None and pcts and any(p != 0.0 for p in pcts):
                label = f"{workload}/{policy}"
                target = (inside if mean - half <= analytic_pct <= mean + half
                          else outside)
                target.add(label)
            print(f"  {workload:20s} {policy:12s} {analytic_pct:11.2f} {mean:11.2f} "
                  f"{ci:>24s} {n:3d} {verdict:>12s}")
            # Per seed, because the MEAN's offset from the analytic gain is smaller than the
            # seed-to-seed scatter and reading a direction off one seed gets it wrong: at
            # the committed seed `balanced` narrows while the mean widens.
            for base, cell in sorted(paired, key=lambda pair: pair[1].seed):
                pct = 100.0 * (base.measured - cell.measured) / base.measured
                print(f"  {'':20s} {'':12s} {'':>11s} {pct:11.3f}   seed {cell.seed:<9d} "
                      f"baseline {base.measured:.6f} -> {cell.measured:.6f}")
    print("\n  CONFIRMED   the paired interval excludes zero on the side the analytic model")
    print("              predicted. It does NOT assert the analytic MAGNITUDE -- read the")
    print("              measured column against the analytic one for that.")
    print("  SIGN FLIP   the interval excludes zero on the OTHER side. The policy ranking")
    print("              the analytic model reports is wrong at this operating point.")
    widths = [100.0 * objective_half_widths(c.simulated)[0] / c.measured for c in cells
              if objective_half_widths(c.simulated)[0] is not None]
    if not widths:
        print("\n  No per-station CIs came back, so there is no propagated half-width to show.")
        return verdicts
    print(f"\n  For contrast, the CONSERVATIVE propagated half-width on a SINGLE measured")
    print(f"  objective runs {min(widths):.3f}% to {max(widths):.3f}% over these "
          f"{len(widths)} cells. A ~2% gain")
    print("  is barely two of those, which is why the paired interval above and not the")
    print("  propagated one is the primary evidence: pairing removes the shared sample path")
    print("  instead of assuming a correlation structure across 14 stations.")
    print("\n  inconclusive  the interval spans zero. Tighten `--precision` on this workload")
    print("              (4x the samples per halving) or add seeds; do not read a sign off")
    print("              the point estimate alone.")
    print("\n  The per-seed rows are there to stop a mechanism being read off one seed.")
    if inside:
        joined = ", ".join(sorted(inside))
        print(f"  The analytic gain sits INSIDE both the per-seed range and the paired interval")
        print(f"  for: {joined}. For those, the offset between the analytic and the measured")
        print("  MEAN gain is NOT resolved at this stopping rule -- only its sign and rough")
        print("  size are, and no mechanism should be claimed for it.")
    if outside:
        joined = ", ".join(sorted(outside))
        print(f"  The analytic gain falls OUTSIDE the paired interval for: {joined}. There the")
        print("  measured mean gain differs from the prediction by more than the seed-to-seed")
        print("  scatter, which IS a resolved discrepancy and needs explaining.")
    print("  identical   every paired difference is exactly zero, because the two policies")
    print("              ARE the same ray here -- r = 1 collapses r* = r onto r* = 1")
    print("              (findings 4). Not a resolution failure.")
    return verdicts


def print_rays(cells, workloads):
    print("\n\n=== D. THE TUNED RAY IS NOT NOISE-FREE: r* across seeds ===")
    print("\n  r* descends from the station spend, which descends from a measured E[T], so a")
    print("  simulated run's ray carries that run's sample path. Reported as an interval.")
    print(f"\n  {'workload':20s} {'analytic r*':>12s} {'mean r*':>10s} {'min':>10s} "
          f"{'max':>10s} {'spread':>10s} {'rel spread':>11s} {'n':>3s}")
    for workload in workloads:
        # Both fork-joins pooled. They are analytically identical (same gamma, same rates,
        # so the same ray to the last bit), so any difference between them on the simulated
        # path is sample-path noise too -- the same quantity this block is measuring.
        rays = [ray for c in cells
                if c.workload == workload and c.policy == "tuned"
                for ray in c.rays["simulated"]]
        if not rays:
            continue
        analytic = [c.rays["analytic"][0] for c in cells
                    if c.workload == workload and c.policy == "tuned"][0]
        spread = max(rays) - min(rays)
        print(f"  {workload:20s} {analytic:12.6f} {statistics.fmean(rays):10.6f} "
              f"{min(rays):10.6f} {max(rays):10.6f} {spread:10.2e} "
              f"{spread / statistics.fmean(rays):11.2e} {len(rays):3d}")
    print("\n  implementation.md measured ~6e-4 relative movement from +/-2% injected noise in")
    print("  E[T]; this column is the same quantity at the stopping rule actually used.")
    print("  n = seeds x 2 fork-joins. The analytic ray is bit-identical across both")
    print("  stations, so every digit of spread here is the sample path.")


def print_bias(cells, workloads, policies):
    print("\n\n=== E. MODEL BIAS at the new operating points (spec section 7's statistic) ===")
    print("\n  Spec section 7 measured the analytic-vs-simulated gap at the INCUMBENT ray only,")
    print("  pooling ~0.15% mean with a ~1.1% worst row and a consistent negative lean. The")
    print("  same statistic here, per policy, says whether the tuned ray's operating point is")
    print("  better or worse served by the closed form -- which is the whole reason the two")
    print("  ~2% gains were only suggestive.")
    print(f"\n  {'policy':12s} {'rows':>6s} {'negative':>9s} {'mean %':>8s} {'worst %':>9s} "
          f"{'flagged':>8s} {'sign-test p':>12s}")

    def block(rows):
        pcts = [pct for pct, _ in rows]
        negative = sum(1 for pct in pcts if pct < 0.0)
        return (len(rows), negative, statistics.fmean(pcts), max(pcts, key=abs),
                sum(1 for _, f in rows if f), sign_test_p(negative, len(rows)))

    for policy, _ in policies:
        rows = [r for cell in cells if cell.policy == policy
                for r in gap_stats(cell.analytic, cell.simulated)["rows"]]
        if not rows:
            continue
        n, negative, mean, worst, flagged, pval = block(rows)
        print(f"  {policy:12s} {n:6d} {negative:4d}/{n:<4d} {mean:8.3f} {worst:9.3f} "
              f"{flagged:8d} {pval:12.4f}")

    # WHICH OF THOSE ROWS ARE NEW. The `invariant-r` cells at spec section 7's own seeds are
    # the same ray, budget and stopping rule it published, against a service that is
    # deterministic given a seed -- so they reproduce its rows bit-for-bit. That is a pipeline
    # witness, NOT a replication, and reading it as one is a mistake this block exists to stop.
    print("\n  Which of those rows are NEW EVIDENCE, and which reproduce spec section 7:")
    print(f"\n  {'bucket':44s} {'rows':>6s} {'negative':>9s} {'mean %':>8s} "
          f"{'sign-test p':>12s}")
    buckets = {
        f"invariant-r at spec 7's seeds (REPRODUCES it)": [
            c for c in cells if c.policy == BASELINE and c.seed in SPEC7_SEEDS],
        f"invariant-r at seeds spec 7 never ran (NEW)": [
            c for c in cells if c.policy == BASELINE and c.seed not in SPEC7_SEEDS],
        f"the two new rays, all seeds (NEW -- spec 7 predates r_star)": [
            c for c in cells if c.policy != BASELINE],
    }
    for label, group in buckets.items():
        rows = [r for c in group for r in gap_stats(c.analytic, c.simulated)["rows"]]
        if not rows:
            continue
        n, negative, mean, _, _, pval = block(rows)
        print(f"  {label:44s} {n:6d} {negative:4d}/{n:<4d} {mean:8.3f} {pval:12.4f}")
    reproduced = len([r for c in buckets[f"invariant-r at spec 7's seeds (REPRODUCES it)"]
                      for r in gap_stats(c.analytic, c.simulated)["rows"]])
    print(f"\n  So of the {sum(len(gap_stats(c.analytic, c.simulated)['rows']) for c in cells)}"
          f" rows in the table above, {reproduced} are spec section 7's published sample, not a")
    print("  new draw from it. Do NOT describe the pooled figures as replicating it -- the")
    print("  first write-up of this run did, and the arithmetic says otherwise. What IS new is")
    print("  the two rays (which spec 7 could not measure) and the seeds it never ran.")

    # And which cells are bit-identical to another policy, so the table's rows are not read as
    # independent confirmations. balanced has r = 1, so equal-rate IS invariant-r there;
    # classical_dominant's tuned ray solves to exactly 1, so tuned IS equal-rate there.
    print("\n  Bit-identical policy pairs (one measurement appearing twice, not two):")
    found = False
    for workload in workloads:
        for i, (a, _) in enumerate(policies):
            for b, _ in policies[i + 1:]:
                left = {c.seed: c.simulated.sojourn_times for c in cells
                        if c.workload == workload and c.policy == a}
                right = {c.seed: c.simulated.sojourn_times for c in cells
                         if c.workload == workload and c.policy == b}
                if left and left == right:
                    print(f"    {workload}: {a} == {b}")
                    found = True
    if not found:
        print("    none")
    detail_seed = COMMITTED_SEED if COMMITTED_SEED in {c.seed for c in cells} \
        else min(c.seed for c in cells)
    print(f"\n  The fork-join rows are the ones under test. Isolated, at seed {detail_seed}:")
    print(f"\n  {'workload':20s} {'policy':12s} {'station':8s} {'ray':>9s} "
          f"{'analytic':>10s} {'measured':>10s} {'gap %':>8s} {'over CI?':>9s}")
    for workload in workloads:
        for policy, _ in policies:
            for cell in cells:
                if cell.workload != workload or cell.policy != policy:
                    continue
                if cell.seed != detail_seed:
                    continue
                stats = gap_stats(cell.analytic, cell.simulated)
                for i, name in enumerate(["fj_pp", "fj_sp"]):
                    idx = FJ_INDEX[name]
                    pct, flagged = stats["rows"][idx]
                    mark = "no CI" if flagged is None else ("yes" if flagged else "no")
                    print(f"  {workload:20s} {policy:12s} {name:8s} "
                          f"{cell.rays['simulated'][i]:9.4f} "
                          f"{cell.analytic.sojourn_times[idx]:10.6f} "
                          f"{cell.simulated.sojourn_times[idx]:10.6f} {pct:8.3f} "
                          f"{mark:>9s}")
    # `Result.degraded` is the whole quality audit -- weak measures, a station with no CI,
    # completed=false -- so filter to conservation misses by their own message rather than
    # taking len(). And it accumulates once per evaluate(), which is `sim_calls` times per
    # cell, so the denominator is per-evaluation, not per-cell.
    checked = sum(1 for st in cells[0].net_s.stations if st.sim_conservation_checked)
    misses = sum(1 for c in cells for entry in c.simulated.degraded
                 if "excludes derived gamma" in entry)
    other = sum(len(c.simulated.degraded) for c in cells) - misses
    total = checked * sum(c.simulated.sim_calls for c in cells)
    rate = 100.0 * misses / total if total else float("nan")
    print(f"\n  gamma conservation: {misses} miss(es) over {total} checks "
          f"({checked} checked stations x {sum(c.simulated.sim_calls for c in cells)} "
          f"evaluations) = {rate:.2f}%.")
    verdict = ("fewer than chance" if rate < 5.0 else
               "MORE than the 5% chance rate -- investigate before trusting this run")
    print(f"  Against the 5% a 95% interval implies: {verdict}.")
    print("  Misses arrive in tandem pairs that share a stream, so the effective count is")
    print("  lower still. Do not read a clean run as a guarantee either (spec 7 saw 1 miss in")
    print("  144 checks at one seed and none at another).")
    if other:
        print(f"  {other} further degraded entr(ies) are NOT conservation misses (weak measure,")
        print("  missing CI, or a cap firing) and are listed per cell in section B.")
    print(f"\n  (one seed only, to keep this block one screen. fj_pp and fj_sp")
    print("  carry no throughput witness -- gamma conservation skips them, qsim-service#8.)")
    print("  `sign-test p` is two-sided exact binomial and is OPTIMISTIC: the rows are not")
    print("  independent (stations share one run; four are analytically identical) and")
    print("  near-zero gaps count as signs, not ties. Read the direction, not the level.")


FJ_INDEX = {}
"""station name -> index, filled once the first network is built."""


# --------------------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--workloads", default=",".join(qn.WORKLOADS),
                   help="comma-separated subset of the three workloads")
    p.add_argument("--policies", default=",".join(name for name, _ in POLICIES),
                   help="comma-separated subset of invariant-r,equal-rate,tuned")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                   help="comma-separated base seeds")
    p.add_argument("--precision", type=float, default=qn.STOPPING["precision"],
                   help="relative CI half-width target; halving costs ~4x the samples")
    p.add_argument("--min-samples", type=int, default=qn.STOPPING["minSamples"])
    p.add_argument("--max-samples", type=int, default=qn.STOPPING["maxSamples"])
    p.add_argument("--max-wallclock", type=int,
                   default=qn.STOPPING["maxWallClockSeconds"])
    p.add_argument("--full-gaps", action="store_true",
                   help="print the full 14-row table for every cell, not just the "
                        "committed seed")
    args = p.parse_args(argv)

    args.workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    if not args.workloads:
        # An empty grid leaves `cells` empty, and every summary block indexes it -- the crash
        # lands after the argument parsing that should have caught it.
        p.error("--workloads is empty; at least one workload is needed")
    if len(set(args.workloads)) != len(args.workloads):
        p.error(f"--workloads contains duplicates {args.workloads}")
    unknown = set(args.workloads) - set(qn.WORKLOADS)
    if unknown:
        p.error(f"unknown workload(s) {sorted(unknown)}; expected {qn.WORKLOADS}")
    wanted = [w.strip() for w in args.policies.split(",") if w.strip()]
    known = {name: value for name, value in POLICIES}
    unknown = set(wanted) - set(known)
    if unknown:
        p.error(f"unknown polic(ies) {sorted(unknown)}; expected {sorted(known)}")
    if BASELINE not in wanted:
        p.error(f"the {BASELINE!r} baseline must be included -- every gain is against it")
    args.policies = [(name, known[name]) for name, _ in POLICIES if name in wanted]
    try:
        args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError as exc:
        p.error(f"--seeds must be integers: {exc}")
    if not args.seeds:
        p.error("--seeds is empty; at least one base seed is needed")
    if len(set(args.seeds)) != len(args.seeds):
        # A repeated seed is the SAME sample path, so stdev over the pair is 0, the interval
        # collapses to zero width and the row prints CONFIRMED off one measurement.
        p.error(f"--seeds contains duplicates {args.seeds}; the service is deterministic given "
                f"a seed, so a repeat is the same sample path and would collapse the interval")
    args.stopping = {
        "alpha": qn.STOPPING["alpha"],
        "precision": args.precision,
        "minSamples": args.min_samples,
        "maxSamples": args.max_samples,
        "maxWallClockSeconds": args.max_wallclock,
    }
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    url = os.environ.get("QOPT_QSIM_URL")
    if not url:
        print("Set QOPT_QSIM_URL=http://localhost:8080. This probe is the simulated "
              "cross-check; there is nothing to do without a service.")
        return 1

    budget = qn.shared_budget()

    # Nothing is printed until the service has answered, so a run against a service that is
    # down leaves NO partial output rather than a header that looks like the start of a good
    # run. `preflight=True` makes the constructor itself do the health check.
    client = QsimClient(url, stopping=args.stopping, preflight=True)

    net0 = qn.build_qcsc_network("balanced")
    FJ_INDEX.update({st.name: i for i, st in enumerate(net0.stations)
                     if st.name.startswith("fj")})

    # FJ_INDEX is read for every workload, so the three topologies must agree on the station
    # order. They do -- the workloads differ only in service rates -- and this pins it.
    for workload in args.workloads:
        order = [st.name for st in qn.build_qcsc_network(workload).stations]
        assert all(order[i] == name for name, i in FJ_INDEX.items()), \
            f"station order differs in {workload!r}; FJ_INDEX would read the wrong rows"

    # FINDING: gate the analytic column BEFORE spending any simulator time, not after. The
    # analytic passes are free (no POSTs), so run them first: if the predictions have drifted
    # from PR #13's recorded numbers, every comparison would be against a different
    # computation and the 12 minutes of simulation would be wasted.
    print_conditions(args, budget, url)
    preflight = {}
    for workload in args.workloads:
        for policy, r_star in args.policies:
            net = qn.build_qcsc_network(workload, r_star=r_star)
            result = Optimizer(net, budget=budget).run()
            preflight[(workload, policy)] = Cell(
                workload, policy, None, result, result,
                {"analytic": [st.r_star for st in net.stations
                              if st.name.startswith("fj")]}, 0.0, net, net)
    failures = check_analytic_anchors(preflight)
    if failures:
        print("\n\nGATE FAILED -- the analytic column has drifted from PR #13's recorded")
        print("numbers, so every comparison below would be against a different computation.")
        print("No simulator time was spent.")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"\n  analytic gate: all {len(preflight)} cells match PR #13's recorded objectives "
          f"and rays.")

    verify_service(client)

    cells = []
    cells_by_key = {}
    t_start = time.time()
    total = len(args.workloads) * len(args.policies) * len(args.seeds)
    done = 0
    for workload in args.workloads:
        for policy, r_star in args.policies:
            for seed in args.seeds:
                cell = run_cell(workload, policy, r_star, seed, budget, client)
                cells.append(cell)
                cells_by_key.setdefault((workload, policy), cell)
                done += 1
                print(f"\n[{done}/{total}] {workload} / {policy} / seed {seed}  "
                      f"({cell.elapsed:.1f}s, {time.time() - t_start:.0f}s elapsed)",
                      file=sys.stderr, flush=True)

    print_analytic_grid(cells_by_key, budget, args.workloads, args.policies)

    print("\n\n=== B. PER-CELL MEASUREMENTS ===")
    for cell in cells:
        print_cell_detail(cell, cell.net_s,
                          args.full_gaps or cell.seed == COMMITTED_SEED)

    verdicts = print_headline(cells, args.workloads, args.policies)
    print_rays(cells, args.workloads)
    print_bias(cells, args.workloads, args.policies)

    tally = {}
    for verdict in verdicts.values():
        tally[verdict] = tally.get(verdict, 0) + 1
    print("\n\n=== VERDICT ===")
    print("\n  " + ", ".join(f"{n} {v}" for v, n in sorted(tally.items())))
    if tally.get("inconclusive"):
        print("\n  Inconclusive rows need a tighter stopping rule or more seeds before they")
        print("  say anything. Re-run just those workloads with --precision halved (~4x the")
        print("  samples) and report them separately -- they are not comparable to this grid.")
    if tally.get("SIGN FLIP"):
        print("\n  A SIGN FLIP means the analytic model ranks these policies the wrong way at")
        print("  this operating point. That is a finding about t_ul, not about the run.")
    # Deliberately no wall clock on stdout: this file is committed as evidence, and a
    # re-run should diff BYTE-IDENTICAL against it so that a real change stands out. Timing
    # is on stderr with the progress lines.
    print(f"\n{total} simulated passes. Every figure above is reproducible from this file's")
    print("own conditions block, and a re-run at these seeds reproduces it byte-for-byte.")
    print(f"\n  elapsed: see stderr", file=sys.stderr)
    print(f"{total} passes in {time.time() - t_start:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
