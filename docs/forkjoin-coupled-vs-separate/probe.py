"""Evidence for docs/forkjoin-coupled-vs-separate/findings.md.

Should each fork-join station's `r*` be solved on its own, given the spend eq 21 hands it,
or should all fork-join stations be solved as one coupled problem under a combined budget?

Self-contained: imports only qopt and the stdlib. Run from the repo root:
    python docs/forkjoin-coupled-vs-separate/probe.py

Sections mirror the findings document:
  1  the tuned ray is scale-free in SLACK space         (findings section 2)
  2  the coupled KKT collapses to the per-station ray   (findings section 3)
  3  (1) nested vs (2) fork-join block vs (3) global    (findings section 4)
  4  why (1) and (3) differ at all: the phi spread      (findings section 5)

The reference-optimum machinery below (VFJ / VGG1 / split / polish) also appears in
docs/slope-calibrated-zeta/probe.py. The duplication is deliberate: the repo's convention is
that each docs/ probe runs standalone from the repo root with no cross-directory imports.

`phi` is the slope-correction factor, named as in docs/slope-calibrated-zeta/. It is NOT the
paper's `kappa` -- see ../paper-map.md.
"""

import math
import warnings

from qopt import ForkJoinStation, GG1Station, Optimizer
from qopt.forkjoin_approx import t_ul
from qopt.forkjoin_policy import _dt_dm1, _min_on_spend_line, optimal_ray

warnings.simplefilter("ignore")


def head(n, title):
    print(f"\n{'='*84}\n{n}. {title}\n{'='*84}")


# ======================================================================================
# per-station value functions and the reference optimum
# ======================================================================================

class VFJ:
    """T*(spend) for a fork-join station, with the ray re-optimized at every spend."""

    def __init__(self, gamma, mu, r, c1, c2, w, name):
        self.gamma, self.mu, self.r, self.c1, self.c2 = gamma, mu, r, c1, c2
        self.w, self.name = w, name
        self.b1, self.b2 = c1 / mu, c2 / (r * mu)
        self.floor = gamma * (self.b1 + self.b2)

    def solve(self, sp):
        return _min_on_spend_line(self.gamma, self.b1, self.b2, sp)

    def ray(self, sp):
        m1, m2 = self.solve(sp)
        return m2 / m1

    def T(self, sp):
        m1, m2 = self.solve(sp)
        return t_ul(self.gamma, m1, m2)

    def marginal(self, sp):
        # Central difference, NOT the envelope formula via `_dt_dm1`: that branch drops the
        # alpha*t_bot term at the m1 == m2 kink, which is where every beta1 == beta2 station
        # and every tight budget sits. Using it here made the EXACT optimum come out worse
        # than the shipped loop -- see ../slope-calibrated-zeta/findings.md section 5.
        h = 1e-6 * (sp - self.floor)
        return -self.w * (self.T(sp + h) - self.T(sp - h)) / (2 * h)


class VGG1:
    def __init__(self, gamma, mu, c, w, name):
        self.gamma, self.mu, self.c, self.w, self.name = gamma, mu, c, w, name
        self.floor = c * gamma / mu

    def ray(self, sp):
        return float("nan")

    def T(self, sp):
        return 1.0 / (sp / self.c * self.mu - self.gamma)

    def marginal(self, sp):
        x = sp / self.c * self.mu - self.gamma
        return self.w * self.mu / (self.c * x * x)


def spend_for(it, nu):
    lo = it.floor * (1 + 1e-14)
    hi = max(it.floor * 2, it.floor + 1.0)
    while it.marginal(hi) > nu:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        lo, hi = (mid, hi) if it.marginal(mid) > nu else (lo, mid)
    return 0.5 * (lo + hi)


def split(items, B):
    """Water-fill: the single nu equalizing w_i*|dT_i/dspend_i|.  This IS the coupled
    optimum over whichever `items` are passed -- all stations for (3), fork-join only for (2)."""
    lo, hi = 1e-16, 1e18
    for _ in range(80):
        nu = math.sqrt(lo * hi)
        lo, hi = (nu, hi) if sum(spend_for(it, nu) for it in items) > B else (lo, nu)
    nu = math.sqrt(lo * hi)
    return nu, [spend_for(it, nu) for it in items]


def polish(items, spends, B):
    """Pairwise local descent on the exact objective, auditing `split` without reusing its
    marginal formula. Relative threshold and a round cap: an absolute one spins forever once
    the objective is ~1e3, where a single ulp is ~1e-13 and rounding noise reads as progress.

    Use `audit_split` rather than calling this directly and comparing: `polish` starts from
    the objective at `spends` and only ever lowers it, so "the result is no worse than
    split's" is true by construction and tests nothing.
    """
    s = list(spends)
    best = sum(it.w * it.T(x) for it, x in zip(items, s))
    step = 1e-3 * B
    for _ in range(200):
        if step <= 1e-13 * B:
            break
        improved = False
        for i in range(len(s)):
            for j in range(len(s)):
                if i == j:
                    continue
                for d in (step, -step):
                    t = list(s)
                    t[i] += d
                    t[j] -= d
                    if t[i] <= items[i].floor or t[j] <= items[j].floor:
                        continue
                    v = sum(it.w * it.T(x) for it, x in zip(items, t))
                    if v < best * (1.0 - 1e-12):
                        best, s, improved = v, t, True
        if not improved:
            step *= 0.5
    return best


def audit_split(items, spends, B, obj):
    """Confirm `split`'s answer is the local optimum. Returns (from_answer, from_perturbed).

    Two descents, because either alone is weak:

    - From split's own answer, descent must find NOTHING better. This is the check that
      matters, but on its own it also passes for a `polish` that does nothing at all --
      which is exactly how the original version of this audit was vacuous.
    - From a PERTURBED start, descent must come BACK to the same objective. That is what
      shows the descent is live, and it is what makes the word "independent" honest.

    Tolerances are measured, not guessed. Over the 36 rows below, descent from split's
    answer improves on it by 0.0 relative in every case, and descent from a 5% perturbation
    stops short by at most 1.0e-12 relative.
    """
    same = polish(items, spends, B)
    assert same >= obj * (1.0 - 1e-12), (
        f"local descent beat split() by {(obj - same) / obj:.3e} relative: split() did not "
        f"find the optimum")
    nudged = list(spends)
    d = 0.05 * (spends[0] - items[0].floor)
    nudged[0] -= d
    nudged[1] += d
    back = polish(items, nudged, B)
    assert back <= obj * (1.0 + 1e-9), (
        f"descent from a perturbed start reached {back} against split()'s {obj}, so it is "
        f"not descending and the check above confirms nothing")
    return same, back


def build(specs):
    """The stations qopt itself optimizes: fork-join on the `tuned` policy, M/M/1 otherwise."""
    out = []
    for spec in specs:
        d = dict(spec)
        kind, name, w = d.pop("kind"), d.pop("name"), d.pop("w")
        if kind == "fj":
            out.append(ForkJoinStation(d["gamma"], d["mu"], w, r=d["r"], c1=d["c1"],
                                       c2=d["c2"], r_star="tuned", name=name))
        else:
            out.append(GG1Station.mm1(d["gamma"], d["mu"], w, c=d["c"], name=name))
    return out


def vals(specs):
    out = []
    for spec in specs:
        d = dict(spec)
        out.append(VFJ(**d) if d.pop("kind") == "fj" else VGG1(**d))
    return out


# --- the four instances ---------------------------------------------------------------
FJ_A = dict(kind="fj", gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, w=1.0, name="FJ-A")

BASELINE = [FJ_A,
            dict(kind="fj", gamma=0.80, mu=2.0, r=2.0, c1=1.0, c2=3.0, w=2.0, name="FJ-B"),
            dict(kind="gg1", gamma=0.60, mu=1.5, c=2.0, w=1.0, name="SS-1"),
            dict(kind="gg1", gamma=1.20, mu=3.0, c=0.5, w=1.5, name="SS-2")]

ADVERSARIAL = [FJ_A,
               dict(kind="fj", gamma=9.00, mu=10.0, r=1.0, c1=0.05, c2=0.05, w=50.0,
                    name="FJ-hot"),
               dict(kind="fj", gamma=0.05, mu=1.0, r=20.0, c1=0.01, c2=9.0, w=0.02,
                    name="FJ-cold"),
               dict(kind="gg1", gamma=0.60, mu=1.5, c=2.0, w=1.0, name="SS-1")]

FJ_ONLY = [FJ_A,
           dict(kind="fj", gamma=0.46, mu=1.0, r=4.0, c1=4.0, c2=1.0, w=1.0, name="FJ-A2")]

ONE_FJ = [FJ_A,
          dict(kind="gg1", gamma=0.60, mu=1.5, c=2.0, w=1.0, name="SS-1"),
          dict(kind="gg1", gamma=1.20, mu=3.0, c=0.5, w=30.0, name="SS-heavy")]

MULTS = (1.001, 1.01, 1.05, 1.2, 1.5, 2, 5, 20, 100)


def main():
    # ==================================================================================
    # 1. the tuned ray is scale-free in slack space
    # ==================================================================================
    head(1, "the tuned ray is scale-free in slack space")
    print("""r_star = m2/m1 moves a great deal with the station's own spend; u = x2/x1 in
slack space barely does, and what little it moves is all `alpha` drift -- alpha is the only
term in t_ul that is not homogeneous of degree -1 in the slacks x_k = m_k - gamma.""")
    G, MU, R, C1, C2 = 0.45, 1.0, 4.0, 4.0, 1.0
    b1, b2 = C1 / MU, C2 / (R * MU)
    sfloor = G * (b1 + b2)
    print(f"\nstation gamma={G} mu={MU} r={R} c1={C1} c2={C2}:  beta1={b1} beta2={b2}  "
          f"price ratio p={b1/b2}  spend floor={sfloor}")
    print(f"\n{'spend/floor':>12} {'r* = m2/m1':>13} {'u = x2/x1':>13} {'alpha':>10} {'rho1':>8}")
    for k in (1.001, 1.01, 1.1, 1.5, 2, 4, 10, 100, 1e4, 1e8):
        m1, m2 = _min_on_spend_line(G, b1, b2, sfloor * k)
        print(f"{k:>12g} {m2/m1:>13.8f} {(m2-G)/(m1-G):>13.8f} "
              f"{(G/m1 + G/m2)/8:>10.6f} {G/m1:>8.4f}")

    def u_star(p, alpha=0.0):
        """Ray condition with alpha held CONSTANT, solved for u = x2/x1.

            p/u^2 - 1/(1-alpha) = (p-1)/(1+u)^2

        With x1 the bottleneck, t_bot = 1/x1, so at frozen alpha
        dT/dx1 = -1/x1^2 + (1-alpha)/D^2  and  dT/dx2 = (1-alpha)(-1/x2^2 + 1/D^2);
        setting the ratio to p, normalizing x1 = 1, x2 = u and dividing by (1-alpha) gives
        the above. alpha = 0 recovers the asymptotic form the spend->inf row converges to.
        """
        def g(u):
            D = 1.0 + u
            return p / u ** 2 - 1.0 / (1.0 - alpha) - (p - 1.0) / D ** 2
        lo, hi = 1e-10, 1e10        # g decreasing in u: +inf as u->0, -inf as u->inf
        for _ in range(300):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
        return math.sqrt(lo * hi)

    print("""
Does the frozen-alpha condition actually PREDICT that drift?  Solve it at each row's own
measured alpha and compare with the measured u.  The alpha -> 0 form is shown alongside: it
is a single number for every row, so it cannot explain any of the drift.""")
    print(f"\n{'spend/floor':>12} {'alpha':>9} {'u measured':>13} {'u frozen-alpha':>15}"
          f" {'rel':>9} {'u at alpha=0':>13}")
    u_inf = u_star(b1 / b2, 0.0)
    worst = 0.0
    for k in (1.001, 1.01, 1.1, 1.5, 2, 4, 10, 100, 1e4, 1e8):
        m1, m2 = _min_on_spend_line(G, b1, b2, sfloor * k)
        alpha = (G / m1 + G / m2) / 8.0
        um = (m2 - G) / (m1 - G)
        uf = u_star(b1 / b2, alpha)
        worst = max(worst, abs(uf / um - 1))
        print(f"{k:>12g} {alpha:>9.6f} {um:>13.8f} {uf:>15.8f} {abs(uf/um-1):>9.1e}"
              f" {u_inf:>13.8f}")
    print(f"\nworst |frozen-alpha prediction / measured - 1| = {worst:.2e}")
    print("""The residual is the next-order effect frozen alpha drops: alpha is not really
constant, so the true condition carries d(alpha)/dx terms, which `_dt_dm1` includes.""")

    print("\nthe alpha -> 0 limit, across price ratios:")
    print(f"{'p = b1/b2':>11} {'u* at alpha=0':>16}")
    for p in (0.25, 1.0, 4.0, 16.0, 100.0):
        print(f"{p:>11g} {u_star(p, 0.0):>16.10f}")
    print(f"\nu*(16) = {u_star(16.0, 0.0):.10f} vs the spend->inf row above: same to 8 digits.")
    print("""
So the budget does not choose the DIRECTION, only how far out along it the station sits, and
it reaches the direction only through alpha. r* = (gamma + x2)/(gamma + x1) therefore runs
from 1 at the stability boundary up to u*(p, 0), and saturates once rho is small.""")

    # ==================================================================================
    # 2. the coupled KKT collapses to the per-station ray condition
    # ==================================================================================
    head(2, "the coupled KKT collapses to the per-station ray condition")
    print("""Solved in FULL 2n-dimensional space by projected gradient descent on the budget
hyperplane -- no per-station reduction assumed, so this cannot beg the question. If the
coupled problem's ray condition really is nu-free, then at its optimum each station's own
derivative ratio must equal its own price ratio, and one scalar nu must serve all of them.""")
    STN = [dict(gamma=0.45, b1=4.0, b2=0.25, w=1.0),
           dict(gamma=0.80, b1=0.5, b2=0.75, w=2.0),
           dict(gamma=2.00, b1=0.02, b2=3.00, w=0.3)]
    B = 3.0 * sum(s["gamma"] * (s["b1"] + s["b2"]) for s in STN)
    beta = [[s["b1"], s["b2"]] for s in STN]
    bn2 = sum(b[0] ** 2 + b[1] ** 2 for b in beta)

    def _obj(ms):
        return sum(s["w"] * t_ul(s["gamma"], m[0], m[1]) for s, m in zip(STN, ms))

    def _spend(ms):
        return sum(s["b1"] * m[0] + s["b2"] * m[1] for s, m in zip(STN, ms))

    ms = [[B / len(STN) / (s["b1"] + s["b2"])] * 2 for s in STN]
    sc = B / _spend(ms)
    ms = [[m[0] * sc, m[1] * sc] for m in ms]
    step, it, gn = 1e-3, 0, math.inf
    for it in range(500_000):
        grad = [[s["w"] * _dt_dm1(s["gamma"], m[0], m[1]),
                 s["w"] * _dt_dm1(s["gamma"], m[1], m[0])] for s, m in zip(STN, ms)]
        dot = sum(g[0] * b[0] + g[1] * b[1] for g, b in zip(grad, beta))
        proj = [[g[0] - dot / bn2 * b[0], g[1] - dot / bn2 * b[1]]
                for g, b in zip(grad, beta)]
        gn = math.sqrt(sum(p[0] ** 2 + p[1] ** 2 for p in proj))
        if gn < 1e-14:
            break
        f0 = _obj(ms)
        while step >= 1e-18:
            trial = [[m[0] - step * p[0], m[1] - step * p[1]] for m, p in zip(ms, proj)]
            if all(t[0] > s["gamma"] and t[1] > s["gamma"] for t, s in zip(trial, STN)) \
                    and _obj(trial) < f0:
                ms, step = trial, step * 1.3
                break
            step *= 0.5
        if step < 1e-18:
            break
    print(f"\niters={it}  |projected grad|={gn:.3e}  budget error={_spend(ms)-B:+.3e}  "
          f"obj={_obj(ms):.14f}")
    print(f"\n{'stn':>4} {'dT/dm1 / dT/dm2':>20} {'beta1/beta2':>14} {'rel err':>10}"
          f" {'w*(-dT/dm1)/beta1':>20}")
    for k, (s, m) in enumerate(zip(STN, ms)):
        d1 = _dt_dm1(s["gamma"], m[0], m[1])
        d2 = _dt_dm1(s["gamma"], m[1], m[0])
        print(f"{k:>4} {d1/d2:>20.12f} {s['b1']/s['b2']:>14.8f} "
              f"{abs((d1/d2)/(s['b1']/s['b2'])-1):>10.2e} {-s['w']*d1/s['b1']:>20.12f}")
    print("""
col 2 == col 3: each ray obeys its OWN price ratio, with no nu, no B and no w in it.
col 5 equal across rows: the entire coupling between stations is one scalar.
So coupling the fork-join stations cannot change any ray except by changing a spend.""")

    # ==================================================================================
    # 3. the three formulations
    # ==================================================================================
    head(3, "(1) nested per-station  vs  (2) fork-join block  vs  (3) global")
    print("""(1) what qopt does: eq 21 splits spend across ALL stations, then each fork-join
    station solves its own ray at the spend it was given.
(2) the proposal: take C_FJ = C - (single-server spend that (1) chose), then solve all
    fork-join stations as one coupled problem under C_FJ.
(3) reference: one coupled problem over every station, water-filled on true marginals and
    audited by pairwise local descent on the exact objective.

All three minimize the same objective under the same total budget C. Negative means better
than (1).""")
    for specs, label in ((BASELINE, "2 fork-join + 2 M/M/1"),
                         (ADVERSARIAL, "3 fork-join + 1 M/M/1, weights 0.02..50, "
                                       "price ratios 16 / 1 / 0.022"),
                         (FJ_ONLY, "2 near-identical fork-join, no single-server station"),
                         (ONE_FJ, "ONE fork-join: (2) is vacuous by construction")):
        fjs = [v for v in vals(specs) if isinstance(v, VFJ)]
        allv = vals(specs)
        nfj = len(fjs)
        print(f"\n--- {label} ---")
        print(f"{'C/floor':>8} {'obj (1)':>15} {'(2)-(1)':>11} {'(3)-(1)':>11}"
              f" {'(3)-(2)':>11} {'max rho':>8} {'r* (1)vs(3)':>12}")
        for mult in MULTS:
            st = build(specs)
            C = mult * sum(s.min_spend for s in st)
            res = Optimizer(st, C).run()
            spend1 = [s.alloc_cost * Si for s, Si in zip(st, res.capacities)]
            obj1 = res.objective
            # (1)'s own r_star must equal the ray our value function picks at (1)'s spend,
            # or the three columns are not comparing the same thing.
            for k in range(nfj):
                assert abs(st[k].r_star - fjs[k].ray(spend1[k])) < 1e-9, st[k].r_star

            _, spend3 = split(allv, C)
            obj3 = sum(v.w * v.T(x) for v, x in zip(allv, spend3))
            audit_split(allv, spend3, C, obj3)

            if nfj > 1:
                B_FJ = C - sum(spend1[nfj:])
                _, s2fj = split(fjs, B_FJ)
            else:
                s2fj = spend1[:nfj]          # nothing to couple: (2) collapses onto (1)
            obj2 = (sum(v.w * v.T(x) for v, x in zip(fjs, s2fj))
                    + sum(v.w * v.T(x) for v, x in zip(allv[nfj:], spend1[nfj:])))

            rho = max(f.gamma / f.solve(sp)[0] for f, sp in zip(fjs, spend1))
            dray = max(abs(f.ray(a) / f.ray(b) - 1)
                       for f, a, b in zip(fjs, spend1, spend3))
            print(f"{mult:>8g} {obj1:>15.7f} {(obj2-obj1)/obj1*100:>10.5f}%"
                  f" {(obj3-obj1)/obj1*100:>10.5f}% {(obj3-obj2)/obj2*100:>10.5f}%"
                  f" {rho:>8.4f} {dray*100:>11.4f}%")
    print("""
The decisive row block is the last one. With a single fork-join station (2) has nothing to
couple, yet (3)-(1) is at its LARGEST -- so the coupling that matters is fork-join against
the single-server stations, which is exactly what (2) freezes when it subtracts their spend.""")

    # ==================================================================================
    # 4. why (1) and (3) differ at all
    # ==================================================================================
    head(4, "why (1) and (3) differ: the phi spread")
    print("""eq 21 splits spend using the surrogate T = zeta/(S*mu-gamma), whose implied
marginal per dollar is w*T*mu/(c*(S*mu-gamma)).  Only RELATIVE marginals decide a split, so
eq 21 is exact iff

    phi_i = [true w_i*|dT_i/dspend_i|] / [surrogate marginal]

is the same for every station.  M/M/1 has zeta == 1 identically, so phi == 1 there and eq 21
is its exact optimum; the fork-join is where phi moves. Measured at (1)'s converged point.""")
    allv = vals(BASELINE)
    print(f"\n{'C/floor':>8} " + " ".join(f"{v.name+' phi':>14}" for v in allv)
          + f" {'spread':>9} {'(3)-(1)':>10}")
    for mult in MULTS:
        st = build(BASELINE)
        C = mult * sum(s.min_spend for s in st)
        res = Optimizer(st, C).run()
        phis = []
        for s, v, Si, T in zip(st, allv, res.capacities, res.sojourn_times):
            x = Si * s.mu - s.gamma
            surrogate = v.w * T * s.mu / (s.alloc_cost * x)
            phis.append(v.marginal(s.alloc_cost * Si) / surrogate)
        _, sp3 = split(allv, C)
        obj3 = sum(v.w * v.T(x) for v, x in zip(allv, sp3))
        print(f"{mult:>8g} " + " ".join(f"{p:>14.9f}" for p in phis)
              + f" {max(phis)/min(phis)-1:>8.3%} {(obj3-res.objective)/res.objective*100:>9.5f}%")
    print("""
An 8.8% spread in the prices costs 0.04% of the objective: the loss is second order, because
the objective is flat near its optimum. That is why (1) is nearly optimal despite mispricing,
and it is also why fixing phi -- see ../slope-calibrated-zeta/ -- is the lever that matters
rather than restructuring which stations are solved together.""")


if __name__ == "__main__":
    main()
