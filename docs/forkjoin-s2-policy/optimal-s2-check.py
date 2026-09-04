"""Is the Optimal-S2.docx procedure valid, and does it agree with qopt's tuned r*?

Checks the alternative S_2 procedure in docs/forkjoin-s2-policy/Optimal-S2.docx against
the condition qopt actually solves. Every claim in
docs/forkjoin-s2-policy/optimal-s2-proposal-review.md is produced here, and the six
questions that have a yes/no answer are written as GATES so that a re-run checks itself
rather than only printing.

Changes nothing in qopt/. The docx's F(B) is transcribed verbatim below and is the only
hand-written formula in this file: everything it is compared against is imported from
qopt, and the QCSC networks come from examples/qcsc_network.py, so rates, costs and
budget cannot drift from the example.

Run: python docs/forkjoin-s2-policy/optimal-s2-check.py

Stdout is deterministic -- no wall clock, no seeds -- so a re-run is checkable by diff.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qopt import ForkJoinStation, Optimizer, R_STAR_TUNED, optimal_ray
from qopt.forkjoin_approx import t_ul
from qopt.forkjoin_policy import _dt_dm1
import examples.qcsc_network as qn


# --------------------------------------------------------------------------------------
# The docx's own formulas, transcribed. Its notation: A = S_1*mu_1, B = S_2*mu_2 are
# EFFECTIVE rates (qopt's m1, m2), mu_2 is server 2's per-unit-capacity rate (qopt's
# mu_base*r_base), g = gamma, x = chi is the Lagrange multiplier of
#     L = T + chi*(S_1 c_1 + S_2 c_2 - C).
#
# Note what is NOT an argument: C. It cancels out of dL/dS_2, so the budget the docx
# takes "from step 1" cannot reach the answer. G4 gates on that.
# --------------------------------------------------------------------------------------

def T_docx(A, B, g):
    """The docx's T, before it is differentiated."""
    alpha = g / (8 * A)
    t_ub = 1 / (A - g) + 1 / (B - g) - 1 / (A + B - 2 * g)
    return (1 - alpha - g / (8 * B)) * t_ub + (alpha + g / (8 * B)) * (1 / (A - g))


def dT_dB_docx(A, B, g):
    """The docx's dT/dB = (dP/dB)Q + P(dQ/dB)."""
    dP = g / (8 * B ** 2)
    Q = 1 / (B - g) - 1 / (A + B - 2 * g)
    P = 1 - g / (8 * A) - g / (8 * B)
    dQ = -1 / (B - g) ** 2 + 1 / (A + B - 2 * g) ** 2
    return dP * Q + P * dQ


def F_docx(B, A, g, mu2, c2, chi):
    """The docx's F(B), whose root it calls the optimal S_2*mu_2."""
    return mu2 * dT_dB_docx(A, B, g) + chi * c2


def root_of_F(A, g, mu2, c2, chi):
    """The docx's own recipe: bracket above B = g, then bisect. F(g+) < 0 < F(inf)."""
    lo, hi = g + 1e-13 * max(1.0, g), max(A, g) * 4 + 10.0
    while F_docx(hi, A, g, mu2, c2, chi) < 0:
        hi *= 2
    while F_docx(lo, A, g, mu2, c2, chi) > 0:
        lo = g + (lo - g) / 2
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if F_docx(mid, A, g, mu2, c2, chi) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# What qopt solves, for comparison. beta_k = c_k/mu_k is the price of a unit of effective
# rate on server k, so the station's spend is exactly beta_1*m1 + beta_2*m2.
# --------------------------------------------------------------------------------------

def betas(mu_base, r, c1, c2):
    return c1 / mu_base, c2 / (r * mu_base)


def ours(gamma, mu_base, r, c1, c2, spend):
    """(m1, m2, nu) at qopt's optimum: the ray, and the station's own shadow price."""
    b1, b2 = betas(mu_base, r, c1, c2)
    r_star = optimal_ray(gamma, mu_base, r, c1, c2, spend)
    m1 = spend / (b1 + b2 * r_star)
    m2 = r_star * m1
    return m1, m2, -_dt_dm1(gamma, m1, m2) / b1


def fd_dT_dm2(g, m1, m2, h):
    """5-point central difference of t_ul in m2, as an independent witness."""
    f = lambda x: t_ul(g, m1, x)
    return (f(m2 - 2*h) - 8*f(m2 - h) + 8*f(m2 + h) - f(m2 + 2*h)) / (12 * h)


def rule(title):
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


VERDICTS = []


def gate(name, ok, detail):
    VERDICTS.append((name, ok))
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} - {detail}")


# The four hand-built stations. gamma is the fork-join arrival rate the QCSC traffic
# equations produce (0.45); the costs are chosen to put beta_1/beta_2 on both sides of 1,
# which is what decides whether the docx's bottleneck assumption holds.
CASES = [
    #  label                        gamma  mu_base    r    c1    c2  spend
    ("beta1 > beta2  (r*>1)",        0.45,     1.0, 4.0,  4.0,  1.0,   3.0),
    ("beta1 > beta2  (r*>1)",        0.45,     1.0, 1.0,  4.0,  1.0,   3.0),
    ("beta1 = beta2  (r*=1, kink)",  0.45,     1.0, 4.0,  1.0,  4.0,   3.0),
    ("beta2 > beta1  (r*<1)",        0.45,     1.0, 1.0,  1.0,  4.0,   3.0),
]


def main():
    print("Optimal-S2.docx vs qopt's tuned r_star")
    print("gates: G1 T, G2 dT/dB, G3 equivalence at chi=nu, G4 C cancels, "
          "G5 constraint is an identity, G8 the kink")

    # ----------------------------------------------------------------------------------
    rule("G1 (GATE) the docx's T against t_ul, on each bottleneck branch")
    print("  The docx sets T_bot = 1/(A-g), which is t_ul's max(...) only when A <= B --")
    print("  i.e. only when server 1 is the (effectively) slower one. Closed form of the")
    print("  discrepancy where server 2 binds:  g*(B^2 - A^2) / (8*A*B*(A-g)*(B-g)).")
    print()
    print("    A        B      branch    T_docx        t_ul          difference    predicted")
    print("  " + "-" * 82)
    worst_ok, worst_bad = 0.0, 0.0
    for A, B in ((0.7, 1.4), (1.0, 1.0), (1.4, 0.7), (2.0, 0.6), (0.6, 3.0)):
        g = 0.45
        td, tu = T_docx(A, B, g), t_ul(g, A, B)
        pred = g * (B*B - A*A) / (8*A*B*(A-g)*(B-g))
        branch = "A<=B" if A <= B else "B<A "
        print(f"  {A:5.2f} {B:8.2f}   {branch}   {td:12.8f} {tu:12.8f} "
              f"{td-tu:+14.6e} {pred:+13.6e}")
        if A <= B:
            worst_ok = max(worst_ok, abs(td - tu))
        else:
            worst_bad = max(worst_bad, abs(td - tu) / tu)
    print(f"\n  max |T_docx - t_ul| where A <= B : {worst_ok:.3e}   (want 0)")
    print(f"  max relative error where B < A   : {worst_bad:.3%}   (want > 0: it is a"
          " different function)")
    gate("G1", worst_ok == 0.0 and worst_bad > 0.01,
         "the docx's T is t_ul exactly where server 1 binds, and a different function "
         "where server 2 binds")

    # ----------------------------------------------------------------------------------
    rule("G2 (GATE) the docx's dT/dB against qopt's implemented dT/dm2")
    print("  qopt's partial is _dt_dm1(g, m2, m1) -- t_ul is symmetric in its two rates.")
    print("  A 5-point finite difference of t_ul witnesses both independently.")
    print()
    print("    A        B     docx dT/dB    qopt dT/dm2   rel.diff   5pt FD        FD rel")
    print("  " + "-" * 82)
    worst_pair, worst_fd = 0.0, 0.0
    for A, B in ((0.7, 1.4), (0.6, 3.0), (1.0, 2.5), (0.5, 0.9), (3.0, 12.0)):
        g = 0.45
        d_docx = dT_dB_docx(A, B, g)
        d_ours = _dt_dm1(g, B, A)
        d_fd = fd_dT_dm2(g, A, B, 1e-5 * B)
        rel = abs(d_docx - d_ours) / abs(d_ours)
        relfd = abs(d_docx - d_fd) / abs(d_fd)
        worst_pair, worst_fd = max(worst_pair, rel), max(worst_fd, relfd)
        print(f"  {A:5.2f} {B:8.2f} {d_docx:14.8f} {d_ours:14.8f} {rel:9.2e} "
              f"{d_fd:13.8f} {relfd:9.2e}")
    print(f"\n  max rel.diff docx vs qopt : {worst_pair:.3e}  (want ~0: the same expression)")
    print(f"  max rel.diff docx vs 5pt FD: {worst_fd:.3e}  (want small: both are dT/dm2)")
    gate("G2", worst_pair < 1e-14 and worst_fd < 1e-5,
         "the docx's dT/dB is qopt's dT/dm2, to floating point")

    # ----------------------------------------------------------------------------------
    rule("G3 (GATE) F(B) = 0 at chi = nu reproduces qopt's ray")
    print("  F(B)=0 is exactly qopt's SECOND stationarity equation, dT/dm2 = -chi*beta_2.")
    print("  qopt solves the RATIO of that and its m1 twin, which eliminates chi. So the")
    print("  two must agree when, and only when, chi is the station's own shadow price")
    print("  nu = |dT/dm1|/beta_1 = |dT/dm2|/beta_2.")
    print()
    hdr = ("  case                          r* (ours)   r* (docx)   rel.diff   "
           "F at ours    branch")
    print(hdr)
    print("  " + "-" * 82)
    worst_eq, kink_seen = 0.0, False
    for label, g, mu_base, r, c1, c2, spend in CASES:
        b1, b2 = betas(mu_base, r, c1, c2)
        m1, m2, nu = ours(g, mu_base, r, c1, c2, spend)
        mu2 = mu_base * r
        Bd = root_of_F(m1, g, mu2, c2, nu)
        rel = abs(Bd - m2) / m2
        branch = "A<=B ok" if m1 <= m2 else "B<A BAD"
        if m1 == m2:
            kink_seen = True
        else:
            worst_eq = max(worst_eq, rel) if m1 < m2 else worst_eq
        print(f"  {label:28s} {m2/m1:10.7f} {Bd/m1:11.7f} {rel:10.2e} "
              f"{F_docx(m2, m1, g, mu2, c2, nu):+12.4e}   {branch}")
    print(f"\n  max rel.diff in r*, where the docx's branch holds: {worst_eq:.3e}")
    gate("G3", worst_eq < 1e-12,
         "on its own branch the docx procedure reproduces qopt's ray exactly, GIVEN "
         "chi = nu")

    # ----------------------------------------------------------------------------------
    rule("G4 (GATE) the budget C cannot reach the answer")
    print("  The docx takes C = c_FJ from the optimization loop, but C enters L only as an")
    print("  additive constant, so it cancels from dL/dS_2 and appears nowhere in F. The")
    print("  root is therefore bit-for-bit independent of C over any range at all.")
    print()
    g, mu_base, r, c1, c2 = 0.45, 1.0, 4.0, 4.0, 1.0
    m1, m2, nu = ours(g, mu_base, r, c1, c2, 3.0)
    roots = {root_of_F(m1, g, mu_base * r, c2, nu) for _ in range(1)}
    print("  (F's signature has no C argument to vary -- that IS the finding.)")
    print(f"  root at the one chi supplied: B = {sorted(roots)[0]:.15f}")
    print("\n  And the constraint the docx imposes has no content either: with S_1 and C")
    print("  both taken from step 1, S_2 = (C - c_1 S_1)/c_2 is already determined, so")
    print("  min over S_2 has zero degrees of freedom. G5 measures that.")
    gate("G4", "C" not in F_docx.__code__.co_varnames,
         "C is not an argument of F: the budget from step 1 cannot influence S_2")

    # ----------------------------------------------------------------------------------
    rule("G5 (GATE) at qopt's converged solutions the constraint is an identity")
    print("  spend = S_1*(c_1 + c_2 r*/r) = c_1 S_1 + c_2 S_2 holds for EVERY r*, so")
    print("  using the constraint to pin chi instead cannot move the ray: any r* is")
    print("  self-consistent. Measured at all three QCSC fixed points.")
    print()
    print("  The two are the same NUMBER but not the same float expression --")
    print("  (S*alloc_cost - c_1*S)/c_2 subtracts and divides where S*(r*/r) multiplies --")
    print("  so agreement is to within an ulp, not bit-for-bit. The ulp column is the")
    print("  measurement; do not restate it as exact equality.")
    print()
    C = qn.shared_budget()
    print(f"  shared budget C = {C:.6f}")
    print()
    print("  workload            stn      r*        S_2 from constraint  S_2 = (r*/r)S_1"
          "    ulps")
    print("  " + "-" * 84)
    worst_ulps = 0
    converged = {}
    for workload in qn.WORKLOADS:
        net = qn.build_qcsc_network(workload, r_star=R_STAR_TUNED)
        res = Optimizer(net, budget=C).run()
        converged[workload] = (net, res)
        for st, S in zip(net.stations, res.capacities):
            if not isinstance(st, ForkJoinStation):
                continue
            from_constraint = (S * st.alloc_cost - st.c1 * S) / st.c2
            ours_S2 = S * (st.r_star / st.r_base)
            ulps = round(abs(from_constraint - ours_S2) / math.ulp(ours_S2))
            worst_ulps = max(worst_ulps, ulps)
            print(f"  {workload:19s} {st.name:6s} {st.r_star:9.6f} "
                  f"{from_constraint:20.15f} {ours_S2:18.15f} {ulps:7d}")
    print(f"\n  worst disagreement: {worst_ulps} ulp  (threshold 8 -- the constraint path")
    print("  subtracts then divides, so a few ulp is what 'the same number' looks like;")
    print("  anything larger would mean the two expressions disagree for a REASON)")
    gate("G5", worst_ulps <= 8,
         f"the equality constraint reproduces qopt's S_2 to within {worst_ulps} ulp, so "
         "it adds no information and leaves nothing to optimize")

    # ----------------------------------------------------------------------------------
    rule("G6 how wrong is the ray when chi is wrong")
    print("  chi is the only input that reaches the answer, so this is the whole error")
    print("  budget of the procedure. Spend is what a wrong chi breaks: the station no")
    print("  longer costs the share eq 21 gave it.")
    print()
    for label, g, mu_base, r, c1, c2, spend in CASES[:1]:
        b1, b2 = betas(mu_base, r, c1, c2)
        m1, m2, nu = ours(g, mu_base, r, c1, c2, spend)
        mu2 = mu_base * r
        print(f"  {label}, gamma={g} mu={mu_base} r={r} c1={c1} c2={c2} spend={spend}")
        print(f"  qopt: r* = {m2/m1:.7f}  nu = {nu:.7f}  T = {t_ul(g, m1, m2):.7f}")
        print()
        print("    chi/nu        B         r*      spend    budget err       T    T err")
        print("  " + "-" * 74)
        for f in (0.5, 0.8, 1.0, 1.25, 2.0):
            Bw = root_of_F(m1, g, mu2, c2, nu * f)
            sw = b1 * m1 + b2 * Bw
            Tw = t_ul(g, m1, Bw)
            print(f"  {f:7.2f} {Bw:10.6f} {Bw/m1:9.6f} {sw:10.6f} {sw/spend-1:+11.2%} "
                  f"{Tw:9.6f} {Tw/t_ul(g, m1, m2)-1:+7.2%}")

    # ----------------------------------------------------------------------------------
    rule("G7 the only multiplier the loop has is not the one F needs")
    print("  eq 21 carries no explicit chi, but implies one. Its stationarity gives")
    print("  S_i = gamma_i/mu_i + sqrt(w_i zeta_i/(nu c_i mu_i)), hence")
    print("      nu_net = w*zeta*mu / (alloc_cost*(S*mu - gamma)^2),")
    print("  equal across stations at eq 21's solution -- which is the check that it has")
    print("  been recovered correctly. But nu_net prices the ZETA-LINEARIZED surrogate")
    print("  w*zeta/(S*mu - gamma), which matches T in value at the fixed point and not")
    print("  in derivative. So it is not the station's nu.")
    print()
    print("  workload            spread(nu_net)   nu_net   nu_station   ratio     r* err"
          "   spend err")
    print("  " + "-" * 86)
    for workload in qn.WORKLOADS:
        net, res = converged[workload]
        nus = [st.weight * z * st.mu / (st.alloc_cost * (S * st.mu - st.gamma) ** 2)
               for st, S, z in zip(net.stations, res.capacities, res.zeta)]
        spread = max(nus) / min(nus) - 1
        nu_net = sum(nus) / len(nus)
        st = next(s for s in net.stations if isinstance(s, ForkJoinStation))
        S = res.capacities[net.stations.index(st)]
        b1, b2 = betas(st.mu_base, st.r_base, st.c1, st.c2)
        m1 = S * st.mu_base
        spend = S * st.alloc_cost
        nu_st = -_dt_dm1(st.gamma, m1, st.r_star * m1) / b1
        Bn = root_of_F(m1, st.gamma, st.mu_base * st.r_base, st.c2, nu_net)
        sp = b1 * m1 + b2 * Bn
        print(f"  {workload:19s} {spread:14.2e} {nu_net:8.6f} {nu_st:12.6f} "
              f"{st.weight*nu_st/nu_net:8.6f} {Bn/m1/st.r_star-1:+9.2%} "
              f"{sp/spend-1:+10.2%}")

    # ----------------------------------------------------------------------------------
    rule("G8 (GATE) the kink r* = 1, which a smooth F = 0 cannot represent")
    print("  t_ul's T_bot kinks at m1 = m2, where the correct condition is one-sided")
    print("  (derivation section 6). classical_dominant's answer IS that kink. The docx's")
    print("  F pairs the two rates' one-sided derivatives from opposite branches, so its")
    print("  root steps past the kink into the region where its own T is wrong.")
    print()
    net, res = converged["classical_dominant"]
    st = next(s for s in net.stations if isinstance(s, ForkJoinStation))
    S = res.capacities[net.stations.index(st)]
    g = st.gamma
    b1, b2 = betas(st.mu_base, st.r_base, st.c1, st.c2)
    m1 = S * st.mu_base
    spend = S * st.alloc_cost
    nu_st = -_dt_dm1(g, m1, st.r_star * m1) / b1
    Bd = root_of_F(m1, g, st.mu_base * st.r_base, st.c2, nu_st)
    print(f"  classical_dominant {st.name}: beta_1 = {b1:g}, beta_2 = {b2:g}, "
          f"spend = {spend:.6f}")
    print()
    # Same-spend comparison: both rays priced on the station's own spend line.
    print("    ray r*        m1         m2       T at the SAME spend")
    print("  " + "-" * 58)
    best = None
    N = 200001
    hi = (spend - b2 * g) / b1
    for i in range(1, N):
        a = g + i * (hi - g) / N
        b = (spend - b1 * a) / b2
        if b <= g:
            break
        T = t_ul(g, a, b)
        if best is None or T < best[0]:
            best = (T, a, b)
    for tag, rs in (("qopt", st.r_star), ("docx", Bd / m1)):
        a = spend / (b1 + b2 * rs)
        print(f"  {tag}  {rs:10.7f} {a:10.6f} {rs*a:10.6f}   {t_ul(g, a, rs*a):.9f}")
    print(f"  grid  {best[2]/best[1]:10.7f} {best[1]:10.6f} {best[2]:10.6f}   "
          f"{best[0]:.9f}   <- {N-1}-point scan of the spend line")
    a_d = spend / (b1 + b2 * (Bd / m1))
    T_ours = t_ul(g, spend / (b1 + b2 * st.r_star), st.r_star * spend / (b1 + b2 * st.r_star))
    T_docx_ray = t_ul(g, a_d, (Bd / m1) * a_d)
    kink_ulps = round(abs(st.r_star - 1.0) / math.ulp(1.0))
    print(f"\n  qopt's r* = {st.r_star!r}, i.e. the kink to within {kink_ulps} ulp.")
    print("  (Exactly 1.0 at the station level, where the prices are exactly symmetric;")
    print("  here m1 and m2 are recovered from a network spend by subtraction.)")
    gate("G8", kink_ulps <= 4 and Bd / m1 < 1.0 - 1e-6 and T_docx_ray > T_ours,
         f"qopt returns the kink (r* = 1 to {kink_ulps} ulp) and the docx root overshoots "
         f"to {Bd/m1:.6f}, worse by {T_docx_ray/T_ours-1:+.4%} at equal spend")

    # ----------------------------------------------------------------------------------
    rule("SUMMARY")
    for name, ok in VERDICTS:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    failed = [n for n, ok in VERDICTS if not ok]
    print(f"\n  {len(VERDICTS) - len(failed)}/{len(VERDICTS)} gates pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
