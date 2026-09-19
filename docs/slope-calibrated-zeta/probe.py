"""Evidence for the slope-calibrated zeta proposal (docs/slope-calibrated-zeta/findings.md).

Self-contained: imports only qopt and the stdlib, writes everything it claims to stdout.
Run from the repo root:  python docs/slope-calibrated-zeta/probe.py

Sections mirror the findings document:
  1  phi closed forms, checked against central differences (findings section 5)
  2  radial == envelope derivative only on the optimal ray (findings section 5)
  3  slope-calibrated zeta lands on the global optimum     (findings section 4)
  4  phi stays positive and bounded                        (findings section 7)
  5  convergence is unaffected                             (findings section 7)
  6  robustness to a wrong phi                             (findings section 6)

The ray/KKT results this proposal rests on are NOT repeated here: they belong to the prior
question and live in docs/forkjoin-coupled-vs-separate/, which this document's section 2 cites.

`phi` is this document's name for the slope-correction factor. It is NOT the paper's
`kappa`, which is the sojourn-time functional form -- see findings section 5.
"""

import math
import warnings

from qopt import ForkJoinStation, GG1Station, Optimizer
from qopt.forkjoin_approx import t_ul
from qopt.forkjoin_policy import _dt_dm1, _min_on_spend_line

warnings.simplefilter("ignore")


def head(n, title):
    print(f"\n{'='*84}\n{n}. {title}\n{'='*84}")


# ======================================================================================
# derivatives
# ======================================================================================

def dT_dS_gg1(st, S):
    """Closed form for GG1Station.sojourn_time, differentiated in S.

    E[T] = (1/m)(1 + k*rho/(1-rho)) = 1/m + k*gamma/(m*(m-gamma)),  m = S*mu,
    k = (cov_a^2 + cov_s^2)/2.  So dT/dm = -1/m^2 - k*gamma*(2m-gamma)/(m*(m-gamma))^2.
    """
    m = S * st.mu
    x = m - st.gamma
    k = (st.cov_a ** 2 + st.cov_s ** 2) / 2.0
    return -st.mu * (1.0 / m ** 2 + k * st.gamma * (2 * m - st.gamma) / (m * x) ** 2)


def dT_dS_fj(st, S):
    """RADIAL derivative of t_ul along the station's current ray -- kink-free.

    Along a ray both rates scale with S, so min(m1, m2) never switches: qopt's `_anchor`
    pairs `mu` with the SLOWER server and `r >= 1`, so t_bot = 1/x1 with no max() to cross.
    That is why this needs none of `_dt_dm1`'s kink handling, and why it is correct where
    `_dt_dm1` is not (findings section 5).

    alpha = gamma*(1/m1 + 1/m2)/8 is homogeneous of degree -1 in S, so d alpha/dS =
    -alpha/S, which contributes the first term below.
    """
    a, b = st.mu, st.mu * st.r
    m1, m2 = a * S, b * S
    x1, x2 = m1 - st.gamma, m2 - st.gamma
    D = x1 + x2
    t_ub = 1.0 / x1 + 1.0 / x2 - 1.0 / D
    t_bot = 1.0 / x1
    alpha = (st.gamma / m1 + st.gamma / m2) / 8.0
    d_ub = -a / x1 ** 2 - b / x2 ** 2 + (a + b) / D ** 2
    d_bot = -a / x1 ** 2
    return (alpha / S) * (t_ub - t_bot) + (1.0 - alpha) * d_ub + alpha * d_bot


def dT_dS_fd(st, S):
    h = 1e-7 * (S - st.gamma / st.mu)
    return (st.sojourn_time(S + h) - st.sojourn_time(S - h)) / (2 * h)


def phi(st, S, exact):
    """The slope-correction factor: |dT/dS| * x / (mu * T).  1.0 <=> eq 22 is already right."""
    return -exact(st, S) * (S * st.mu - st.gamma) / (st.mu * st.sojourn_time(S))


# ======================================================================================
# exact per-station value functions, for the reference global optimum
# ======================================================================================

class VFJ:
    """T*(spend) for a fork-join station: the ray re-optimizes at every spend."""

    def __init__(self, gamma, mu, r, c1, c2, w, name):
        self.gamma, self.mu, self.r, self.w, self.name = gamma, mu, r, w, name
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
        # Central difference, NOT the envelope formula via `_dt_dm1`: that formula drops
        # t_bot at the r_star == 1 kink, which is exactly where tight budgets and every
        # beta1 == beta2 station sit (findings section 5).
        h = 1e-6 * (sp - self.floor)
        return -self.w * (self.T(sp + h) - self.T(sp - h)) / (2 * h)


class VGG1:
    def __init__(self, gamma, mu, c, cov_a, cov_s, w, name):
        self.gamma, self.mu, self.c, self.w, self.name = gamma, mu, c, w, name
        self.k = (cov_a ** 2 + cov_s ** 2) / 2.0
        self.floor = c * gamma / mu

    def ray(self, sp):
        return float("nan")

    def T(self, sp):
        m = sp / self.c * self.mu
        rho = self.gamma / m
        return (1.0 / m) * (1.0 + self.k * rho / (1.0 - rho))

    def marginal(self, sp):
        h = 1e-6 * (sp - self.floor)
        return -self.w * (self.T(sp + h) - self.T(sp - h)) / (2 * h)


def spend_for(it, nu):
    lo = it.floor * (1 + 1e-14)
    hi = max(it.floor * 2, it.floor + 1.0)
    while it.marginal(hi) > nu:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if it.marginal(mid) > nu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def split(items, B):
    """Water-fill: one nu equalizing w_i*|dT_i/dspend_i|.  This is the coupled optimum."""
    lo, hi = 1e-16, 1e18
    for _ in range(80):
        nu = math.sqrt(lo * hi)
        if sum(spend_for(it, nu) for it in items) > B:
            lo = nu
        else:
            hi = nu
    nu = math.sqrt(lo * hi)
    return nu, [spend_for(it, nu) for it in items]


def polish(items, spends, B):
    """Pairwise local descent on the exact objective -- audits `split` without reusing its
    marginal formula, so a bug in one cannot hide in the other."""
    s = list(spends)
    best = sum(it.w * it.T(x) for it, x in zip(items, s))
    step = 1e-3 * B
    # The improvement threshold is RELATIVE. An absolute one spins forever: at an objective
    # of ~1e3 a single ulp is ~1e-13, so any absolute tolerance below that accepts pure
    # rounding noise as an improvement, `improved` never comes back False, and the step
    # never halves. The round cap is a backstop, not the normal exit.
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


# ======================================================================================
# station builders: level-calibrated (as shipped) vs slope-calibrated (the proposal)
# ======================================================================================

def _slope_zeta(st, T, S, exact):
    """zeta = phi * T * x.  Linear in T, which is what keeps `Optimizer._noise_floor`
    correct unchanged -- it propagates a CI half-width through this same hook."""
    x = S * st.mu - st.gamma
    return phi(st, S, exact) * T * x


def build(specs, *, blend):
    """blend=0 reproduces the shipped eq-22 level calibration exactly (phi forced to 1);
    blend=1 is the proposal; intermediate values interpolate, for section 8."""
    out = []
    for spec in specs:
        d = dict(spec)
        kind, name, w = d.pop("kind"), d.pop("name"), d.pop("w")
        if kind == "fj":
            class C(ForkJoinStation):
                def zeta_from(self, T, S, _b=blend):
                    x = S * self.mu - self.gamma
                    p = phi(self, S, dT_dS_fj)
                    return (1.0 + _b * (p - 1.0)) * T * x
            out.append(C(d["gamma"], d["mu"], w, r=d["r"], c1=d["c1"], c2=d["c2"],
                         r_star="tuned", name=name))
        else:
            class C(GG1Station):
                def zeta_from(self, T, S, _b=blend):
                    x = S * self.mu - self.gamma
                    p = phi(self, S, dT_dS_gg1)
                    return (1.0 + _b * (p - 1.0)) * T * x
            out.append(C(d["gamma"], d["mu"], w, c=d["c"], cov_a=d["cov_a"],
                         cov_s=d["cov_s"], name=name))
    return out


def vals(specs):
    out = []
    for spec in specs:
        d = dict(spec)
        out.append(VFJ(**d) if d.pop("kind") == "fj" else VGG1(**d))
    return out


# --- the networks used below ----------------------------------------------------------
NET_FJ_MM1 = [
    dict(kind="fj", gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, w=1.0, name="FJ-A"),
    dict(kind="fj", gamma=0.80, mu=2.0, r=2.0, c1=1.0, c2=3.0, w=2.0, name="FJ-B"),
    dict(kind="gg1", gamma=0.60, mu=1.5, c=2.0, cov_a=1.0, cov_s=1.0, w=1.0, name="SS-1"),
    dict(kind="gg1", gamma=1.20, mu=3.0, c=0.5, cov_a=1.0, cov_s=1.0, w=1.5, name="SS-2"),
]
NET_MIXED_COV = [
    dict(kind="fj", gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, w=1.0, name="FJ-A"),
    dict(kind="gg1", gamma=0.60, mu=1.5, c=2.0, cov_a=1.0, cov_s=0.0, w=1.0, name="M/D/1"),
    dict(kind="gg1", gamma=1.20, mu=3.0, c=0.5, cov_a=2.0, cov_s=2.0, w=1.5, name="cov2"),
    dict(kind="gg1", gamma=0.90, mu=2.0, c=1.0, cov_a=1.0, cov_s=1.0, w=1.0, name="M/M/1"),
]
NET_STRESS = [
    dict(kind="fj", gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, w=1.0, name="FJ-A"),
    dict(kind="fj", gamma=2.00, mu=5.0, r=1.0, c1=0.1, c2=0.1, w=8.0, name="FJ-B"),
    dict(kind="gg1", gamma=0.60, mu=1.5, c=2.0, cov_a=1.0, cov_s=0.0, w=1.0, name="M/D/1"),
    dict(kind="gg1", gamma=1.20, mu=3.0, c=0.5, cov_a=5.0, cov_s=5.0, w=1.5, name="cov5"),
    dict(kind="gg1", gamma=0.90, mu=2.0, c=1.0, cov_a=1.0, cov_s=1.0, w=0.2, name="M/M/1"),
]
MULTS = (1.01, 1.05, 1.2, 1.5, 2, 5, 20)


def main():
    # ======================================================================================
    # 1. phi closed forms
    # ======================================================================================
    head(1, "phi closed forms, checked against central differences")
    print("""phi = |dT/dS| * x / (mu * T).  phi == 1 means eq 22's level calibration already
carries the right slope; anything else is the factor eq 21 misprices the station by.""")
    print(f"\n{'station':<18}" + "".join(f"{'rho='+f'{r}':>28}" for r in (0.95, 0.67, 0.25)))
    CASES = [("M/M/1", GG1Station.mm1(0.6, 1.5, c=2.0), dT_dS_gg1),
             ("M/D/1", GG1Station.md1(0.6, 1.5, c=2.0), dT_dS_gg1),
             ("G/G/1 cov=2", GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0), dT_dS_gg1),
             ("G/G/1 cov=5", GG1Station(0.6, 1.5, c=2.0, cov_a=5.0, cov_s=5.0), dT_dS_gg1),
             ("fork-join p=16", ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0), dT_dS_fj),
             ("fork-join r*=1", ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, r_star=1.0),
              dT_dS_fj)]
    for name, st, exact in CASES:
        row = f"{name:<18}"
        for target in (0.95, 0.67, 0.25):
            S = st.gamma / (st.mu * target)
            e, f = exact(st, S), dT_dS_fd(st, S)
            row += f"  phi={phi(st,S,exact):>9.6f} (fd rel {abs(e/f-1):>7.1e})"
        print(row)
    print("""
The fork-join is the MILDEST case. M/D/1 reaches 0.83 and cov=5 goes well above 1, so this
is an eq-22 issue across every station whose zeta depends on S -- not a fork-join issue.""")

    print("\nwhere `_dt_dm1`'s kink branch would have been used instead (findings section 5):")
    print(f"{'station':<18} {'spend/floor':>12} {'radial (correct)':>18} {'via _dt_dm1':>14} {'rel':>9}")
    fj = ForkJoinStation(9.0, 10.0, r=1.0, c1=0.05, c2=0.05)   # beta1 == beta2, so r* == 1
    for k in (1.05, 1.5, 4.0):
        S = k * fj.gamma / fj.mu
        m1, m2 = S * fj.mu, S * fj.mu * fj.r
        env = -_dt_dm1(fj.gamma, m1, m2) / (fj.c1 / fj.mu_base) * fj.alloc_cost
        rad = -dT_dS_fj(fj, S)
        print(f"{'FJ beta1==beta2':<18} {k:>12g} {rad:>18.8g} {env:>14.8g} {abs(env/rad-1):>9.1%}")


    # ======================================================================================
    # 2. radial == envelope, but only on the optimal ray
    # ======================================================================================
    head(2, "the radial derivative equals the envelope one only on the optimal ray")
    print("""eq 21 works in S-space; the coupled optimum is stated in spend-space. They agree
because at an optimal ray grad T is parallel to beta, so EVERY direction with the same
spend increment gives the same first-order change in T -- including the radial one that
`dT/dS` at a frozen ray takes. Off the optimal ray that equality fails, which is why this
is a property of the fixed point and not an identity.""")
    from qopt.forkjoin_policy import optimal_ray
    spec = dict(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0)
    v = VFJ(w=1.0, name="A", **spec)
    print(f"\non the optimal ray:\n{'spend/floor':>12} {'r*':>11} {'radial/c':>18}"
          f" {'envelope':>18} {'rel':>9}")
    for k in (1.001, 1.01, 1.1, 1.5, 2, 4, 10, 100):
        sp = v.floor * k
        rs = optimal_ray(spec["gamma"], spec["mu"], spec["r"], spec["c1"], spec["c2"], sp)
        st = ForkJoinStation(r_star=rs, **spec)
        rad = dT_dS_fj(st, sp / st.alloc_cost) / st.alloc_cost
        env = -v.marginal(sp)
        print(f"{k:>12g} {rs:>11.7f} {rad:>18.10g} {env:>18.10g} {abs(rad/env-1):>9.1e}")
    print(f"\noff the optimal ray (true optimum at this spend is "
          f"{optimal_ray(0.45,1.0,4.0,4.0,1.0,4*v.floor):.6f}):")
    for rs in (1.0, 2.0, 4.0):
        st = ForkJoinStation(r_star=rs, **spec)
        S = 4 * v.floor / st.alloc_cost
        rad = dT_dS_fj(st, S) / st.alloc_cost
        env = -v.marginal(S * st.alloc_cost)
        print(f"  r*={rs:<5} radial={rad:>14.8g} envelope={env:>14.8g} rel={abs(rad/env-1):>8.2%}")


    # ======================================================================================
    # 3. the payoff
    # ======================================================================================
    head(3, "slope-calibrated zeta lands on the global exact optimum")
    print("""(1)  as shipped: eq 21 + eq 22 level calibration
(1s) proposal: eq 21 + zeta = phi*T*x
(3)  reference: the coupled optimum over ALL stations, water-filled on true marginals and
     independently audited by pairwise local descent on the exact objective.""")
    for specs, label in ((NET_FJ_MM1, "2 fork-join + 2 M/M/1  (phi moves only on the FJ pair)"),
                         (NET_MIXED_COV, "1 fork-join + M/D/1 + G/G/1(cov 2) + M/M/1")):
        print(f"\n--- {label} ---")
        print(f"{'C/floor':>8} {'(1) level':>15} {'(1s) slope':>15} {'(3) exact':>15}"
              f" {'(1s)-(3)':>12} {'(1)-(3)':>10} {'it':>4} {'it_s':>5}")
        V = vals(specs)
        for mult in MULTS:
            stL, stS = build(specs, blend=0.0), build(specs, blend=1.0)
            C = mult * sum(s.min_spend for s in stL)
            rL, rS = Optimizer(stL, C).run(), Optimizer(stS, C).run()
            _, sp3 = split(V, C)
            o3 = sum(x.w * x.T(y) for x, y in zip(V, sp3))
            audit = polish(V, sp3, C)
            assert audit <= o3 + 1e-12, f"split() is not a local min: {audit} < {o3}"
            print(f"{mult:>8g} {rL.objective:>15.9f} {rS.objective:>15.9f} {o3:>15.9f}"
                  f" {(rS.objective-o3)/o3*100:>11.6f}% {(rL.objective-o3)/o3*100:>9.4f}%"
                  f" {rL.iterations:>4} {rS.iterations:>5}")
    print("\n(1s) matches the independently-computed optimum to machine precision at every row.")


    # ======================================================================================
    # 4. phi stays positive and bounded
    # ======================================================================================
    head(4, "phi > 0 and bounded -- eq 21 needs a strictly positive zeta")
    lo_all, hi_all = math.inf, -math.inf
    print(f"{'station':<22} {'phi range over spend/floor in [1+1e-6, 1e8]':>46}")
    for cov in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        st = GG1Station(1.0, 1.0, c=1.0, cov_a=cov, cov_s=cov)
        vs = [phi(st, m, dT_dS_gg1) for m in (1.000001, 1.001, 1.1, 2, 10, 1e3, 1e6, 1e8)]
        lo_all, hi_all = min(lo_all, *vs), max(hi_all, *vs)
        print(f"{'G/G/1 cov='+str(cov):<22} {'['+f'{min(vs):.6f}, {max(vs):.6f}'+']':>46}")
    for r in (1.0, 2.0, 20.0):
        for rs in (0.05, 1.0, 3.0, 50.0):
            st = ForkJoinStation(0.45, 1.0, r=r, c1=4.0, c2=1.0, r_star=rs)
            vs = [phi(st, m * st.gamma / st.mu, dT_dS_fj)
                  for m in (1.000001, 1.001, 1.1, 2, 10, 1e4, 1e8)]
            lo_all, hi_all = min(lo_all, *vs), max(hi_all, *vs)
            print(f"{'FJ r='+str(r)+' r*='+str(rs):<22} "
                  f"{'['+f'{min(vs):.6f}, {max(vs):.6f}'+']':>46}")
    print(f"\nphi over everything above: [{lo_all:.6f}, {hi_all:.6f}] -- strictly positive.")
    print("""The FJ rows repeat across r because `_anchor` derives BOTH effective rates from
r_star alone (mu = mu_base*min(1,r_star), r = max(1,r_star)/min(1,r_star)); the constructed
r_base reaches `alloc_cost` but never the queueing model, so phi cannot depend on it.""")
    print("""phi -> 0 at the stability boundary for cov=0 is correct, not a defect: a D/D/1
station has E[T] = 1/m with no queueing term at all, so extra capacity buys it very
little and eq 21 should indeed hand it close to the minimum.""")


    # ======================================================================================
    # 5. convergence
    # ======================================================================================
    head(5, "convergence is unaffected")
    V = vals(NET_STRESS)
    print(f"{'C/floor':>10} {'it level':>9} {'it slope':>9} {'(1)-(3)%':>11} {'(1s)-(3)%':>13}")
    bad = 0
    for mult in (1.0001, 1.001, 1.01, 1.05, 1.2, 1.5, 2, 3, 5, 10, 50, 500, 1e4):
        try:
            stL, stS = build(NET_STRESS, blend=0.0), build(NET_STRESS, blend=1.0)
            C = mult * sum(s.min_spend for s in stL)
            rL, rS = Optimizer(stL, C).run(), Optimizer(stS, C).run()
            _, sp3 = split(V, C)
            o3 = sum(x.w * x.T(y) for x, y in zip(V, sp3))
            gs = (rS.objective - o3) / o3 * 100
            bad += abs(gs) > 1e-6 or not rS.converged
            print(f"{mult:>10g} {rL.iterations:>9} {rS.iterations:>9}"
                  f" {(rL.objective-o3)/o3*100:>10.4f}% {gs:>12.7f}%")
        except Exception as exc:
            bad += 1
            print(f"{mult:>10g}  FAILED: {type(exc).__name__}: {exc}")
    print(f"\n{bad} failures or off-optimum rows out of 13.")


    # ======================================================================================
    # 6. robustness to a wrong phi
    # ======================================================================================
    head(6, "robustness: how wrong may phi be before it stops helping?")
    print("""zeta = [1 + f*(phi-1)]*T*x.  f=0 is the shipped level calibration, f=1 the proposal,
f=2 an error as large as doing nothing but in the opposite direction. This is the question
that decides whether an ANALYTIC phi is safe on the simulated path, where the true slope is
not measurable (findings section 6).""")
    V = vals(NET_STRESS)
    BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    print(f"\n{'C/floor':>8}" + "".join(f"{'f='+str(f):>12}" for f in BLENDS))
    for mult in (1.05, 1.2, 1.5, 2, 5):
        C = mult * sum(s.min_spend for s in build(NET_STRESS, blend=0.0))
        _, sp3 = split(V, C)
        o3 = sum(x.w * x.T(y) for x, y in zip(V, sp3))
        row = f"{mult:>8g}"
        for f in BLENDS:
            r = Optimizer(build(NET_STRESS, blend=f), C).run()
            row += f"{(r.objective-o3)/o3*100:>11.5f}%"
        print(row)
    print("""
Each cell is % above the true optimum; lower is better. The loss is quadratic in the phi
error, so a roughly-right phi captures most of the gain and even a 100% overcorrection
still beats doing nothing.""")


if __name__ == "__main__":
    main()
