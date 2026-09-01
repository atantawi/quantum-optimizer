"""The locally optimal ray for a fork-join station (issue #10 item 3).

A fork-join station has two servers and one capacity variable, so `S_2` is free. In
EFFECTIVE-RATE space the choice is a point on the station's spend line: raising server k's
rate by one unit costs `beta_k = c_k/mu_k`, so the spend is exactly `beta_1*m1 + beta_2*m2`.
`ForkJoinStation` runs on a ray `m2 = r_star*m1` of that line, which is what keeps spend
linear in `S` -- the form eq 21's budget column requires.

This module answers "which ray", by solving the local optimality condition of findings
section 8 at a GIVEN spend. That single scalar solve is the inner half of a nested fixed
point: the spend itself comes from eq 21, whose prices depend on `r_star`, so the outer
loop closes the circle. See docs/forkjoin-s2-policy/findings.md sections 4, 7 and 8.
"""

import math

from qopt.exceptions import InstabilityError


def optimal_ray(gamma, mu_base, r, c1, c2, spend):
    """`r_star` minimizing E[T] over the rays reachable at this station spend.

    Args:
        gamma: arrival rate at the station.
        mu_base, r, c1, c2: the CONSTRUCTED hardware -- server 1 has base rate `mu_base`
            and costs `c1`, server 2 has base rate `mu_base*r` and costs `c2`. These are
            the constructor arguments, not `ForkJoinStation`'s effective `mu`/`r`.
        spend: the station's share of the budget, `S * alloc_cost`.

    Returns `m2*/m1*` in effective rates, which is exactly the `r_star` to run on. Raises
    InstabilityError if `spend` cannot keep both servers stable on any ray.
    """
    b1 = c1 / mu_base
    b2 = c2 / (r * mu_base)
    m1, m2 = _min_on_spend_line(gamma, b1, b2, spend)
    return m2 / m1


def _dt_dm1(lam, m1, m2):
    """d t_ul / d m1 at fixed m2, including the term that comes through alpha.

    `t_ul` is symmetric in its two rates, so the other partial is this with the arguments
    swapped. Undefined at m1 == m2, where `t_bot`'s max() kinks; this takes the branch
    that treats m1 as the non-bottleneck, which only ever matters on a set of measure zero
    inside the bisection below.
    """
    alpha = (lam / m1 + lam / m2) / 8.0
    D = m1 + m2 - 2.0 * lam
    t_ub = 1.0 / (m1 - lam) + 1.0 / (m2 - lam) - 1.0 / D
    t_bot = 1.0 / (min(m1, m2) - lam)
    d_ub = -1.0 / (m1 - lam) ** 2 + 1.0 / (D * D)
    d_bot = -1.0 / (m1 - lam) ** 2 if m1 < m2 else 0.0
    d_alpha = -lam / (8.0 * m1 * m1)
    return d_alpha * (t_bot - t_ub) + (1.0 - alpha) * d_ub + alpha * d_bot


def _min_on_spend_line(gamma, b1, b2, spend):
    """(m1, m2) minimizing `t_ul` subject to `b1*m1 + b2*m2 == spend`.

    Solves the stationarity condition of findings section 8 directly,

        |dT/dm1| / |dT/dm2| = beta_1/beta_2

    by bisection, rather than minimizing `t_ul` along the line. Two reasons, and the first
    is the load-bearing one:

    - **Precision.** Locating a minimum by comparing function values is limited to about
      sqrt(machine epsilon), ~1e-8 relative, because a quadratic minimum is flat. That
      noise lands in `r_star`, hence in `alloc_cost`, hence in the optimizer's step -- and
      it stalls the outer fixed point, which jitters above a 1e-9 tolerance for several
      iterations after the answer has actually been reached. A sign change is not flat, so
      bisection pins m1 to full relative precision and the outer loop stops when it is
      done. m2, recovered from the spend line, is the weaker of the two: it inherits
      ~eps*(beta_1/beta_2) rather than eps, so the returned RATIO does too. That is 1e-13
      or better at the price ratios this model reaches, far below what the outer tolerance
      resolves, and the guard at the end of this function handles the extreme where it is
      not.
    - It is also the rule the finding actually states, and it costs one cheap evaluation
      per halving instead of a scan of the whole interval. The 200-step cap is not a
      tolerance: bisection runs until the bracket is adjacent floats, which is ~52 steps
      at ordinary scale and at most 80 over the whole reachable input range.

    Bisection is valid here without a unimodality assumption beyond a single sign change.
    Let `g(m1) = dT/dm1 - (beta_1/beta_2)*dT/dm2` along the line. As m1 falls to gamma,
    dT/dm1 -> -infinity so g -> -infinity; as m1 rises until m2 falls to gamma, dT/dm2 ->
    -infinity so g -> +infinity. `t_bot`'s kink at m1 == m2 is a jump in g, but an UPWARD
    one -- both terms gain a positive increment as m1 crosses m2 -- so it runs with the
    trend rather than against it. A bracket that straddles the kink therefore converges
    onto it, which is the r_star = 1 answer, with no special case.
    """
    # Both servers need a rate above gamma, which bounds m1 from either side.
    lo, hi = gamma, (spend - b2 * gamma) / b1
    if not hi > lo:
        raise InstabilityError(
            f"fork-join spend {spend} admits no stable ray: needs > "
            f"gamma*(beta_1+beta_2) = {gamma * (b1 + b2)}"
        )
    price = b1 / b2

    def g(m1):
        m2 = (spend - b1 * m1) / b2
        if m2 <= gamma:
            # `hi` is derived by subtraction too, so at price ratios around 1e14 it loses
            # b2 entirely and admits midpoints past the m2 = gamma boundary, where the
            # derivatives divide by zero. Report "too far right" and let bisection retreat.
            return math.inf
        return _dt_dm1(gamma, m1, m2) - price * _dt_dm1(gamma, m2, m1)

    # The endpoints are the stability boundary, where the signs are known and the
    # derivatives diverge, so they are bracketed but never evaluated.
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        if mid <= a or mid >= b:      # adjacent floats: no interval left to halve
            break
        if g(mid) < 0.0:
            a = mid
        else:
            b = mid
    m1 = 0.5 * (a + b)
    m2 = (spend - b1 * m1) / b2
    if not (m1 > gamma and m2 > gamma):
        # Recovering m2 by subtraction destroys it once b1*m1 approaches `spend`: one ulp
        # of `spend` is eps*b1*m1, so m2 carries ~eps*m1*(b1/b2) of absolute error, which
        # swamps the true m2 - gamma when b1/b2 is large AND the spend sits within
        # ~eps*(b1/b2) of the floor. Measured at b1/b2 = 1e5, gamma = 0.45: one ulp of
        # spend is 1e-11 against a true margin of 1e-12, and m2 comes back BELOW gamma --
        # which escapes as an InstabilityError from the middle of the optimizer loop.
        #
        # Every such case has r_star = 1 to full precision, and that is not luck: this
        # only triggers at the stability boundary, and r_star tends to 1 there (findings
        # section 4). So return that ray computed directly. It needs no cancelling
        # subtraction, and the guard at the top of this function makes it strictly stable:
        # spend > gamma*(b1+b2) gives spend/(b1+b2) > gamma.
        m1 = m2 = spend / (b1 + b2)
    return m1, m2


# --------------------------------------------------------------------------------------
# The named policies a caller selects between.
# --------------------------------------------------------------------------------------

R_STAR_INVARIANT_R = "invariant-r"
"""Hold the constructed ratio: `r_star = r`, so both servers receive the same capacity S.

qopt's incumbent policy and the default, priced `c1 + c2`.
"""

R_STAR_EQUAL_RATE = "equal-rate"
"""Equalize the two effective rates: `r_star = 1`, so server 2 receives only `S/r`.

The paper's rule, priced `c1 + c2/r` -- which is the exact cost of its own capacities,
not a fudge factor.
"""

R_STAR_TUNED = "tuned"
"""Solve `r_star` from the local optimality condition at the station's own spend.

The station starts on the ray `r_star = 1` -- the one that minimizes its stability floor,
see `_INITIAL_R_STAR` -- and is retuned by the optimizer each iteration, making `r_star`
the inner variable of a nested fixed point.
"""

R_STAR_FIXED = "fixed"
"""Reported policy for a station constructed with a numeric `r_star` -- some other ray of
the same family, held constant. Not accepted as an input: it carries no value."""

_INITIAL_R_STAR = {R_STAR_INVARIANT_R: None, R_STAR_TUNED: 1.0, R_STAR_EQUAL_RATE: 1.0}
"""Named policy -> the ray to start on, with None meaning "the constructed r".

`tuned` starts at r_star = 1 rather than at the incumbent r, and that is a feasibility
requirement rather than a preference. A station's stability floor over the family is
`gamma*(c1 + c2*r_star/r) / (mu*min(1, r_star))`, which is minimized at exactly
r_star = 1, where it equals the spend line's own floor `gamma*(beta_1+beta_2)`.

That makes the entry here the ray the whole feasibility story is told at:
`ForkJoinStation.min_spend` prices the station's floor on this ray precisely because it is
the ray a run starts from, so a `tuned` entry of r would advertise the incumbent's floor
and refuse budgets the station can in fact serve -- measured: every C in (1.80, 3.15] that
`equal-rate` completes, and the station's own answer at that boundary is r_star = 1 anyway.
Starting on the floor-minimizing ray is also the better initial guess, since r_star tends
to 1 as the budget tightens.
"""


def resolve_r_star(r_star, r):
    """Normalize a `ForkJoinStation` r_star argument to `(policy_name, initial r_star)`.

    Accepts None (the default, equivalent to R_STAR_INVARIANT_R), one of the named
    policies, or a strictly-positive finite float naming a fixed ray.
    """
    if r_star is None:
        r_star = R_STAR_INVARIANT_R
    if isinstance(r_star, str):
        if r_star not in _INITIAL_R_STAR:
            raise ValueError(
                f"r_star must be a number > 0 or one of "
                f"{sorted(_INITIAL_R_STAR)}, got {r_star!r}"
            )
        initial = _INITIAL_R_STAR[r_star]
        return r_star, r if initial is None else initial
    if not math.isfinite(r_star) or r_star <= 0:
        raise ValueError(f"r_star must be a finite number > 0, got {r_star}")
    return R_STAR_FIXED, float(r_star)
