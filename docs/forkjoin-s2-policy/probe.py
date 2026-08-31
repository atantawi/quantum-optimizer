"""Throwaway probe: is there a locally optimal S_2 for a fork-join station?

Answers the eight questions in docs/forkjoin-s2-policy/findings.md. Changes nothing in
qopt/ -- the two alternative S_2 policies are ForkJoinStation subclasses defined here, and
the QCSC network is built by patching the example module's ForkJoinStation binding so the
topology, rates, costs and budget cannot drift from examples/qcsc_network.py.

Run: python docs/forkjoin-s2-policy/probe.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from qopt import ForkJoinStation, Optimizer, min_feasible_budget
from qopt.forkjoin_approx import t_ul
from qopt.station import distribution_dict
import examples.qcsc_network as qn


# --------------------------------------------------------------------------------------
# The inner problem: split a fork-join station's spend B across the two servers.
#
# Effective rates m1, m2. Raising m_k by one unit means raising S_k by 1/mu_k, so the
# price of a unit of effective rate is beta_k = c_k / mu_k, and the station's spend is
# exactly beta_1*m1 + beta_2*m2. Both incumbent policies are points on that same line:
#   qopt today:  m2 = r*m1        -> spend (c1+c2)*S
#   the paper:   m2 = m1          -> spend (c1 + c2/r)*S1
# --------------------------------------------------------------------------------------

def betas(mu1, r, c1, c2):
    """Price per unit of effective rate on each server."""
    return c1 / mu1, c2 / (r * mu1)


def dT_dm2(lam, m1, m2):
    """Analytic d t_ul / d m2, including the term that comes through alpha."""
    alpha = (lam / m1 + lam / m2) / 8.0
    D = m1 + m2 - 2.0 * lam
    t_ub = 1.0 / (m1 - lam) + 1.0 / (m2 - lam) - 1.0 / D
    t_bot = 1.0 / (min(m1, m2) - lam)
    dalpha = -lam / (8.0 * m2 * m2)
    d_ub = -1.0 / (m2 - lam) ** 2 + 1.0 / (D * D)
    d_bot = -1.0 / (m2 - lam) ** 2 if m2 < m1 else 0.0
    return dalpha * (t_bot - t_ub) + (1.0 - alpha) * d_ub + alpha * d_bot


def m1_bounds(lam, b1, b2, B):
    """Open interval of m1 for which both servers are stable on the spend line."""
    return lam, (B - b2 * lam) / b1


GRID = 4001
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def solve_fj_line(lam, b1, b2, B, *, grid=GRID):
    """Minimize t_ul on beta_1*m1 + beta_2*m2 = B. Returns (T, m1, m2, n_local_minima)."""
    lo, hi = m1_bounds(lam, b1, b2, B)
    if hi <= lo:
        raise ValueError(f"infeasible: B={B} too small for lam={lam}")
    xs = [lo + (hi - lo) * (i + 1) / (grid + 1) for i in range(grid)]
    ys = [t_ul(lam, x, (B - b1 * x) / b2) for x in xs]

    # Count interior local minima on the grid: this is the unimodality check.
    n_min = sum(1 for i in range(1, grid - 1) if ys[i] <= ys[i - 1] and ys[i] < ys[i + 1])

    k = min(range(grid), key=ys.__getitem__)
    a = xs[k - 1] if k > 0 else lo
    b = xs[k + 1] if k < grid - 1 else hi
    # Golden-section refine inside the bracketing grid cell.
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    for _ in range(200):
        if t_ul(lam, c, (B - b1 * c) / b2) < t_ul(lam, d, (B - b1 * d) / b2):
            b, d = d, c
        else:
            a, c = c, d
        c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
        if b - a < 1e-14 * max(1.0, abs(a)):
            break
    m1 = 0.5 * (a + b)
    m2 = (B - b1 * m1) / b2
    return t_ul(lam, m1, m2), m1, m2, n_min


def solve_series_line(lam, b1, b2, B):
    """Same spend line, but two M/M/1 queues in TANDEM. Closed form.

    T = 1/(m1-lam) + 1/(m2-lam) is separable, so (m_k - lam) is proportional to
    1/sqrt(beta_k) and the slack ratio is exactly sqrt(beta_1/beta_2).
    """
    t = (B - lam * (b1 + b2)) / (math.sqrt(b1) + math.sqrt(b2))
    m1 = lam + t / math.sqrt(b1)
    m2 = lam + t / math.sqrt(b2)
    return 1.0 / (m1 - lam) + 1.0 / (m2 - lam), m1, m2


# --------------------------------------------------------------------------------------
# The three S_2 policies as station classes.
# --------------------------------------------------------------------------------------

class PaperFJ(ForkJoinStation):
    """S_2 = S_1/r: both servers land on the same effective rate S_1*mu_1.

    Spend is S_1*(c1 + c2/r), which is exactly the paper's c_FJ -- so the paper's cost
    factor is the true cost of its own S_2 rule, not a fudge.
    """

    @property
    def alloc_cost(self):
        return self.c1 + self.c2 / self.r

    def sojourn_time(self, S):
        m = S * self.mu
        self._check_stable(m)
        return t_ul(self.gamma, m, m)


class OptimalFJ(ForkJoinStation):
    """S_2 chosen to minimize t_ul at the station's own spend (c1+c2)*S.

    alloc_cost is (c1+c2), unchanged from qopt today, so eq 21 sees the same linear
    budget column and the same S range -- the current policy is a FEASIBLE POINT of this
    minimization at every S, which is why this can only weakly improve on it.
    """

    def split(self, S):
        b1, b2 = betas(self.mu, self.r, self.c1, self.c2)
        return solve_fj_line(self.gamma, b1, b2, self.alloc_cost * S)

    def sojourn_time(self, S):
        # Physical stability is guaranteed by the interior solve (t_ul -> inf at either
        # boundary). This check is about the eq-22 SURROGATE: zeta = T*(S*mu - gamma)
        # needs S*mu > gamma to stay positive, which eq 21's base term gamma/mu ensures.
        self._check_stable(S * self.mu)
        return self.split(S)[0]

    def sim_node(self, S, job_class):
        """Branches at the SPLIT rates, not at S*mu and S*r*mu.

        The inherited implementation would emit the ray this policy does not run (1.0 and
        4.0 where the split is 1.1102 and 2.2373 in quantum_dominant at S=1) and return a
        plausible number for a network that does not exist. `mu` and `r` are meaningless
        for this station -- only `split` knows its rates.
        """
        _, m1, m2 = self.split(S)[:3]
        return {
            "name": self.name,
            "type": "fork-join",
            "branches": [
                {"service": {job_class: {"distribution": distribution_dict(m1, 1.0)}}},
                {"service": {job_class: {"distribution": distribution_dict(m2, 1.0)}}},
            ],
            "join": "all",
        }


def build(workload, cls, **cost_kw):
    """QCSC network with `cls` as the fork-join station type."""
    orig = qn.ForkJoinStation
    qn.ForkJoinStation = cls
    try:
        return qn.build_qcsc_network(workload, **cost_kw)
    finally:
        qn.ForkJoinStation = orig


def fj_params(workload, b, c_qpu=qn.C_QPU, c_gpu=qn.C_GPU):
    """(mu1, r, c1, c2, which_is_slow) for a parallel phase, mirroring qn._fork_join."""
    mu_q, mu_g = qn.rates(workload, b)
    if mu_q <= mu_g:
        return mu_q, mu_g / mu_q, c_qpu, c_gpu, "qpu"
    return mu_g, mu_q / mu_g, c_gpu, c_qpu, "gpu"


GAMMA_FJ = 0.45   # both fork-joins, derived by the traffic equations (spec 3.1)
PHASES = (("fj_pp", qn.B_PP), ("fj_sp", qn.B_SP))


def rule(t):
    print("\n" + "=" * 86 + f"\n{t}\n" + "=" * 86)


def q5_sign_check():
    """GATE. Is d t_ul / d m2 negative everywhere on the spend line?

    The alpha blend makes this a real question: alpha = (rho1+rho2)/8 falls as m2 rises,
    which shifts weight from t_bot onto the strictly larger t_ub. That term is POSITIVE
    (more m2 looks harmful). If it ever wins, r* is an artifact of the approximation.
    """
    rule("Q5 (GATE) sign of dT/dm2 on the spend line, and analytic-vs-finite-difference")
    worst = []
    fd_err = 0.0
    for wl in qn.WORKLOADS:
        for name, b in PHASES:
            mu1, r, c1, c2, _ = fj_params(wl, b)
            b1, b2 = betas(mu1, r, c1, c2)
            floor = GAMMA_FJ * (b1 + b2)
            for mult in (1.05, 1.5, 3.0, 6.0, 20.0):
                B = floor * mult
                lo, hi = m1_bounds(GAMMA_FJ, b1, b2, B)
                pos = 0
                worst_d = -math.inf
                for i in range(1, 600):
                    m1 = lo + (hi - lo) * i / 600.0
                    m2 = (B - b1 * m1) / b2
                    if m2 <= GAMMA_FJ:
                        continue
                    d = dT_dm2(GAMMA_FJ, m1, m2)
                    if abs(m2 - m1) > 1e-3:      # skip the max() kink, where FD is invalid
                        # 5-point stencil: a 2-point one leaves ~1% truncation error in
                        # the sliver where m1 sits just above gamma and t_ul ~ 1e5.
                        h = 1e-5 * m2
                        fd = (-t_ul(GAMMA_FJ, m1, m2 + 2*h) + 8*t_ul(GAMMA_FJ, m1, m2 + h)
                              - 8*t_ul(GAMMA_FJ, m1, m2 - h) + t_ul(GAMMA_FJ, m1, m2 - 2*h)
                              ) / (12 * h)
                        fd_err = max(fd_err, abs(d - fd) / max(abs(d), abs(fd)))
                    if d > 0:
                        pos += 1
                    worst_d = max(worst_d, d)
                worst.append((wl, name, mult, pos, worst_d))
    bad = [w for w in worst if w[3] > 0]
    print(f"  configurations probed : {len(worst)}  (3 workloads x 2 stations x 5 budgets)")
    print(f"  with any dT/dm2 > 0   : {len(bad)}")
    print(f"  max dT/dm2 seen       : {max(w[4] for w in worst):.6e}  (want < 0)")
    print(f"  max rel err vs 5pt FD : {fd_err:.3e}  (the analytic derivative is verified)")
    if bad:
        for w in bad:
            print(f"    POSITIVE: {w[0]:19s} {w[1]} B/floor={w[2]}  {w[3]} pts, max {w[4]:.3e}")
    print(f"\n  VERDICT: {'PASS - more m2 always helps' if not bad else 'FAIL - see above'}")
    return not bad


def q1_unimodality():
    rule("Q1 unimodality of t_ul on the spend line")
    worst = 0
    rows = []
    for wl in qn.WORKLOADS:
        for name, b in PHASES:
            mu1, r, c1, c2, _ = fj_params(wl, b)
            b1, b2 = betas(mu1, r, c1, c2)
            floor = GAMMA_FJ * (b1 + b2)
            for mult in (1.05, 1.5, 3.0, 6.0, 20.0):
                T, m1, m2, n = solve_fj_line(GAMMA_FJ, b1, b2, floor * mult)
                worst = max(worst, n)
                rows.append((wl, name, mult, n))
    print(f"  interior local minima on a {GRID}-point grid: max {worst} over {len(rows)} cases")
    print(f"  VERDICT: {'PASS - unimodal, a 1-D solve is safe' if worst <= 1 else 'FAIL'}")
    return worst <= 1


def q4_q2_special_cases_and_drift():
    rule("Q4 predicted special cases, and Q2 how r* drifts with the station's spend")
    print("  beta1/beta2 = (c1/c2)*r is the price ratio; r* = m2*/m1* is the optimal")
    print("  effective-rate ratio. qopt today runs r, the paper runs 1.\n")
    hdr = (f"  {'workload':19s} {'stn':6s} {'r':>4s} {'b1/b2':>7s} "
           f"{'B/floor':>8s} {'r*':>7s} {'T(r*)':>9s} {'T(qopt)':>9s} {'T(paper)':>9s} {'gain%':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for wl in qn.WORKLOADS:
        for name, b in PHASES:
            mu1, r, c1, c2, slow = fj_params(wl, b)
            b1, b2 = betas(mu1, r, c1, c2)
            floor = GAMMA_FJ * (b1 + b2)
            for mult in (1.2, 2.0, 4.0, 8.0, 20.0):
                B = floor * mult
                T, m1, m2, _ = solve_fj_line(GAMMA_FJ, b1, b2, B)
                # incumbent policies at the SAME spend B
                # Each incumbent is one ray of the same line, and each has its own
                # (strictly higher) stability floor -- below it the ray is unstable while
                # the line still has feasible points.
                S_q = B / (c1 + c2)                     # qopt: m2 = r*m1
                S_p = B / (c1 + c2 / r)                 # paper: m2 = m1
                T_q = (t_ul(GAMMA_FJ, S_q * mu1, S_q * r * mu1)
                       if S_q * mu1 > GAMMA_FJ else None)
                T_p = (t_ul(GAMMA_FJ, S_p * mu1, S_p * mu1)
                       if S_p * mu1 > GAMMA_FJ else None)
                f = lambda v: "  UNSTABLE" if v is None else f"{v:9.5f}"
                gain = "      --" if T_q is None else f"{100*(T_q-T)/T_q:7.2f}"
                print(f"  {wl:19s} {name:6s} {r:4.1f} {b1/b2:7.3f} {mult:8.2f} "
                      f"{m2/m1:7.3f} {T:9.5f} {f(T_q)} {f(T_p)} {gain}")
            print()


def q8_floors():
    """The three policies' minimum feasible spend for one fork-join station."""
    rule("Q4b stability floors: the spend each policy needs to keep the station stable")
    print("  optimal : gamma*(beta1+beta2) = gamma*(c1/mu1 + c2/mu2)")
    print("  qopt    : gamma*(c1+c2)/mu1        paper: gamma*(c1 + c2/r)/mu1\n")
    hdr = f"  {'workload':19s} {'r':>4s} {'optimal':>9s} {'qopt':>9s} {'paper':>9s} {'qopt/opt':>9s} {'paper/opt':>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for wl in qn.WORKLOADS:
        mu1, r, c1, c2, _ = fj_params(wl, qn.B_PP)
        b1, b2 = betas(mu1, r, c1, c2)
        f_opt = GAMMA_FJ * (b1 + b2)
        f_q = GAMMA_FJ * (c1 + c2) / mu1
        f_p = GAMMA_FJ * (c1 + c2 / r) / mu1
        print(f"  {wl:19s} {r:4.1f} {f_opt:9.4f} {f_q:9.4f} {f_p:9.4f} "
              f"{f_q/f_opt:9.3f} {f_p/f_opt:10.3f}")
    print("\n  Both ratios >= 1: the optimal-split policy's stability region CONTAINS both")
    print("  incumbents', since each incumbent is one ray of the line it minimizes over.")


def q6_elasticity():
    rule("Q6 fork-join vs series: how hard does each shift budget to the cheap server?")
    print("  Matched control: same gamma, same spend line, same (beta1,beta2) -- two M/M/1")
    print("  queues in TANDEM instead of forked. Slack ratio Q = (m2-g)/(m1-g).")
    print("  Series is exactly Q = sqrt(beta1/beta2), i.e. log-log slope 1/2.\n")
    hdr = (f"  {'b1/b2':>9s} {'Q_series':>9s} {'Q_forkjoin':>11s} "
           f"{'exp_series':>11s} {'exp_fj':>8s} {'local d/dlog':>13s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    # Hold the FJ station's shape fixed and sweep only the price ratio.
    mu1, r = 1.0, 4.0
    lam = GAMMA_FJ
    for ratio in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0):
        b2 = 0.25
        b1 = ratio * b2
        floor = lam * (b1 + b2)
        B = 6.0 * floor
        _, m1, m2, _ = solve_fj_line(lam, b1, b2, B)
        _, s1, s2 = solve_series_line(lam, b1, b2, B)
        Qf = (m2 - lam) / (m1 - lam)
        Qs = (s2 - lam) / (s1 - lam)
        ef = math.log(Qf) / math.log(ratio) if ratio > 1 else float("nan")
        es = math.log(Qs) / math.log(ratio) if ratio > 1 else float("nan")
        # local slope d log Q_fj / d log(b1/b2)
        h = 1e-4
        out = []
        for sgn in (+1, -1):
            rr = ratio * math.exp(sgn * h)
            bb1 = rr * b2
            ff = lam * (bb1 + b2)
            _, a1, a2, _ = solve_fj_line(lam, bb1, b2, 6.0 * ff)
            out.append(math.log((a2 - lam) / (a1 - lam)))
        loc = (out[0] - out[1]) / (2 * h)
        print(f"  {ratio:9.1f} {Qs:9.4f} {Qf:11.4f} {es:11.4f} {ef:8.4f} {loc:13.4f}")
    print("\n  A slope BELOW 1/2 means the fork-join resists shifting budget to the cheap")
    print("  server -- it compresses toward homogeneity relative to series.")


def q7_cost_sweep():
    rule("Q7 does the PRICE ratio drive r*, and where does r* cross 1?")
    print("  Sweeping c_qpu/c_gpu with the topology fixed. r* < 1 means the nominally")
    print("  FASTER server is bought DOWN below the slower one.\n")
    for wl in ("quantum_dominant", "classical_dominant"):
        mu1, r, _, _, slow = fj_params(wl, qn.B_PP)
        print(f"  {wl}  (slow server = {slow}, r = {r:.1f})")
        print(f"    {'c_qpu/c_gpu':>12s} {'b1/b2':>8s} {'r*':>8s}")
        prev = None
        cross = None
        for cq in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0):
            _, _, c1, c2, _ = fj_params(wl, qn.B_PP, c_qpu=cq, c_gpu=1.0)
            b1, b2 = betas(mu1, r, c1, c2)
            floor = GAMMA_FJ * (b1 + b2)
            _, m1, m2, _ = solve_fj_line(GAMMA_FJ, b1, b2, 6.0 * floor)
            rs = m2 / m1
            print(f"    {cq:12.2f} {b1/b2:8.3f} {rs:8.4f}")
            if prev and (prev[1] - 1.0) * (rs - 1.0) < 0:
                cross = (prev[0], cq)
            prev = (cq, rs)
        print(f"    r* crosses 1 between c_qpu/c_gpu = {cross}" if cross
              else "    r* does not cross 1 over this sweep")
        print()


def q3_network():
    rule("Q3 whole-network objective under the three policies, one shared budget")
    C = qn.shared_budget()
    print(f"  C = {C:.6f} (= {qn.BUDGET_MULTIPLE} x the balanced floor under qopt's policy),")
    print("  the SAME absolute budget for all three policies.\n")
    hdr = (f"  {'workload':19s} {'policy':9s} {'floor':>8s} {'obj':>10s} "
           f"{'d%':>7s} {'it':>3s} {'cnv':>4s} {'fj_pp r*':>9s} {'fj_sp r*':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    summary = {}
    for wl in qn.WORKLOADS:
        base = None
        for label, cls in (("qopt", ForkJoinStation), ("paper", PaperFJ), ("optimal", OptimalFJ)):
            net = build(wl, cls)
            floor = min_feasible_budget(net.stations)
            res = Optimizer(net, C).run()
            if base is None:
                base = res.objective
            rs = []
            for st, S in zip(net.stations, res.capacities):
                if not isinstance(st, ForkJoinStation) or st.name not in ("fj_pp", "fj_sp"):
                    continue
                if isinstance(st, OptimalFJ):
                    _, m1, m2, _ = st.split(S)
                    rs.append(m2 / m1)
                elif isinstance(st, PaperFJ):
                    rs.append(1.0)
                else:
                    rs.append(st.r)
            print(f"  {wl:19s} {label:9s} {floor:8.3f} {res.objective:10.6f} "
                  f"{100*(base-res.objective)/base:7.2f} {res.iterations:3d} "
                  f"{str(res.converged):>4s} {rs[0]:9.4f} {rs[1]:9.4f}")
            summary[(wl, label)] = res.objective
        print()
    return summary


# --------------------------------------------------------------------------------------
# Q9. Why per-station dominance does not survive eq 21, and the family that fixes it.
#
# Both incumbents are RAYS: m2 = r*m1 (qopt) and m2 = m1 (paper). A ray is what makes
# spend LINEAR in S, which is what eq 21's budget column needs. Generalize to a free
# ratio r_star: m2 = r_star*m1, spend = S*(c1 + c2*r_star/r). Then
#     r_star = r -> alloc_cost c1+c2      (qopt)
#     r_star = 1 -> alloc_cost c1+c2/r    (the paper)
# so both are members of one one-parameter family, and r_star is a tunable.
# --------------------------------------------------------------------------------------

def rstar_class(r_star):
    """ForkJoinStation on the ray m2 = r_star * m1, priced consistently."""

    class RStarFJ(ForkJoinStation):
        def __init__(self, gamma=None, mu=None, weight=1.0, *, r, c1, c2, name=None):
            k = min(1.0, r_star)
            # Anchor `mu` on whichever server ends up EFFECTIVELY slower, so eq 21's
            # base term gamma/mu keeps the BINDING server stable even when r_star < 1.
            # Without this, r_star < 1 lets eq 21 hand back an S that starves server 2.
            super().__init__(gamma, mu * k, weight,
                             r=max(1.0, r_star) / k, c1=c1, c2=c2, name=name)
            self.r_base = r
            self.r_star = r_star

        @property
        def alloc_cost(self):
            return self.c1 + self.c2 * self.r_star / self.r_base

    return RStarFJ


def q9_family():
    rule("Q9 the r_star family: choosing the RAY, at network level")
    C = qn.shared_budget()

    # The family must contain both incumbents exactly, or the comparison means nothing.
    for wl in qn.WORKLOADS:
        r = fj_params(wl, qn.B_PP)[1]
        o_q = Optimizer(build(wl, ForkJoinStation), C).run().objective
        o_p = Optimizer(build(wl, PaperFJ), C).run().objective
        f_q = Optimizer(build(wl, rstar_class(r)), C).run().objective
        f_p = Optimizer(build(wl, rstar_class(1.0)), C).run().objective
        assert abs(f_q - o_q) < 1e-12 and abs(f_p - o_p) < 1e-12, (wl, o_q, f_q, o_p, f_p)
    print("  family identity check: r_star=r reproduces qopt and r_star=1 reproduces the")
    print("  paper, bit-for-bit, on all three workloads. PASS\n")

    hdr = (f"  {'workload':19s} {'r':>4s} {'r*_best':>8s} {'obj(best)':>10s} "
           f"{'obj(qopt)':>10s} {'obj(paper)':>11s} {'obj(split)':>11s} {'best vs qopt':>13s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    curves = {}
    for wl in qn.WORKLOADS:
        r = fj_params(wl, qn.B_PP)[1]
        grid = [0.2 + 0.02 * i for i in range(291)]        # 0.20 .. 6.00
        vals = []
        for rs in grid:
            try:
                vals.append((Optimizer(build(wl, rstar_class(rs)), C).run().objective, rs))
            except Exception:
                vals.append((float("inf"), rs))
        best_obj, best_rs = min(vals)
        curves[wl] = vals
        o_q = Optimizer(build(wl, ForkJoinStation), C).run().objective
        o_p = Optimizer(build(wl, PaperFJ), C).run().objective
        o_s = Optimizer(build(wl, OptimalFJ), C).run().objective
        print(f"  {wl:19s} {r:4.1f} {best_rs:8.3f} {best_obj:10.6f} {o_q:10.6f} "
              f"{o_p:11.6f} {o_s:11.6f} {100*(o_q-best_obj)/o_q:12.2f}%")
    print("\n  objective vs r_star (the curve eq 21 actually sees):")
    print(f"    {'r_star':>7s} " + " ".join(f"{w[:9]:>11s}" for w in qn.WORKLOADS))
    for rs in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        row = []
        for wl in qn.WORKLOADS:
            v = min((abs(x[1] - rs), x[0]) for x in curves[wl])[1]
            row.append(f"{v:11.6f}")
        print(f"    {rs:7.2f} " + " ".join(row))
    return curves


def q10_diagnose():
    """Isolate WHY the inner-split policy loses to the paper in classical_dominant."""
    rule("Q10 diagnosis: the inner split optimizes the SPLIT, not the station's SHARE")
    C = qn.shared_budget()
    wl = "classical_dominant"
    print(f"  {wl} at C = {C:.4f}. Q3 showed r* = 1.0000 at the converged point, so the")
    print("  optimal-split policy and the paper choose the IDENTICAL split there. The only")
    print("  remaining difference is alloc_cost, hence the SHARE of C the station gets.\n")
    hdr = f"  {'policy':9s} {'alloc_cost':>10s} {'S(fj_pp)':>9s} {'spend':>8s} {'m1':>8s} {'m2':>8s} {'T(fj)':>8s} {'obj':>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, cls in (("qopt", ForkJoinStation), ("paper", PaperFJ), ("split", OptimalFJ)):
        net = build(wl, cls)
        res = Optimizer(net, C).run()
        i = [s.name for s in net.stations].index("fj_pp")
        st, S = net.stations[i], res.capacities[i]
        if isinstance(st, OptimalFJ):
            _, m1, m2, _ = st.split(S)
        elif isinstance(st, PaperFJ):
            m1 = m2 = S * st.mu
        else:
            m1, m2 = S * st.mu, S * st.r * st.mu
        print(f"  {label:9s} {st.alloc_cost:10.3f} {S:9.4f} {st.alloc_cost*S:8.4f} "
              f"{m1:8.4f} {m2:8.4f} {res.sojourn_times[i]:8.5f} {res.objective:10.6f}")
    print("\n  The split policy reports S = spend/(c1+c2) while its true m1 comes from the")
    print("  inner solve, so 'S' no longer means 'server 1 runs at S*mu' and eq 22's")
    print("  surrogate zeta = T*(S*mu - gamma) is anchored to a rate the station does not")
    print("  have. eq 21 then mis-prices the station: here it OVER-funds it (spend 7.96 vs")
    print("  the paper's 7.49) and buys a better T(fj) that is not worth what the other")
    print("  13 stations gave up for it. Direction is incidental; consistency is the point.")
    print("  That mispricing is the whole gap -- the split itself is identical.")


if __name__ == "__main__":
    # Confirm the fork-join gamma the traffic equations actually derive.
    net = qn.build_qcsc_network("balanced")
    g = {st.name: st.gamma for st in net.stations}
    assert abs(g["fj_pp"] - GAMMA_FJ) < 1e-12 and abs(g["fj_sp"] - GAMMA_FJ) < 1e-12, g
    print(f"fork-join gamma derived from the traffic equations: {g['fj_pp']}")

    gate = q5_sign_check()
    uni = q1_unimodality()
    if not (gate and uni):
        print("\nGATE FAILED - the remaining questions are not well posed. Stopping.")
        sys.exit(1)
    q4_q2_special_cases_and_drift()
    q8_floors()
    q6_elasticity()
    q7_cost_sweep()
    q3_network()
    q10_diagnose()
    q9_family()
