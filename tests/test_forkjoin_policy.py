"""The locally-optimal fork-join ray (issue #10 item 3).

Reference figures come from docs/forkjoin-s2-policy/findings.md, which swept the same
quantity with an independent throwaway probe.
"""

import math

import pytest

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul
from qopt.forkjoin_policy import (
    R_STAR_TUNED,
    _dt_dm1,
    _min_on_spend_line,
    optimal_ray,
)


def test_equal_prices_give_the_papers_ray_by_symmetry():
    """beta_1 == beta_2 makes t_ul symmetric in the two rates, so the optimum is m1 == m2.

    beta_k = c_k/mu_k, so equal prices means c2 == c1*r. This is findings section 8's
    "beta_1 = beta_2 recovers the paper's r -> 1 exactly by symmetry".
    """
    r_star = optimal_ray(gamma=0.45, mu_base=1.0, r=4.0, c1=1.0, c2=4.0, spend=3.6)
    assert r_star == pytest.approx(1.0, rel=1e-9)


# The QCSC fork-join hardware, per workload: (mu_base, r, c1, c2). Server 1 is whichever
# of the two units is slower, and costs follow the SERVER -- so `classical_dominant` puts
# the expensive QPU on the fast server. gamma = 0.45 at every fork-join phase.
GAMMA_FJ = 0.45
QCSC_FJ = {
    "balanced":           (1.0, 1.0, 4.0, 1.0),   # beta_1/beta_2 = 4
    "quantum_dominant":   (1.0, 4.0, 4.0, 1.0),   # beta_1/beta_2 = 16
    "classical_dominant": (1.0, 4.0, 1.0, 4.0),   # beta_1/beta_2 = 1
}

# findings section 4 / probe-output.txt lines 27-61: r* against the station's spend as a
# multiple of its own optimal stability floor gamma*(beta_1+beta_2). Published to 3dp.
FINDINGS_SECTION_4 = [
    ("balanced",           1.20, 1.079), ("balanced",           2.00, 1.276),
    ("balanced",           4.00, 1.461), ("balanced",           8.00, 1.566),
    ("balanced",          20.00, 1.633),
    ("quantum_dominant",   1.20, 1.242), ("quantum_dominant",   2.00, 1.797),
    ("quantum_dominant",   4.00, 2.274), ("quantum_dominant",   8.00, 2.532),
    ("quantum_dominant",  20.00, 2.693),
    ("classical_dominant", 1.20, 1.000), ("classical_dominant", 2.00, 1.000),
    ("classical_dominant", 4.00, 1.000), ("classical_dominant", 8.00, 1.000),
    ("classical_dominant",20.00, 1.000),
]


@pytest.mark.parametrize("workload,floor_multiple,expected", FINDINGS_SECTION_4)
def test_reproduces_the_swept_r_star_table_from_findings(workload, floor_multiple,
                                                         expected):
    """Cross-check against a table produced by an independent throwaway probe."""
    mu_base, r, c1, c2 = QCSC_FJ[workload]
    b1, b2 = c1 / mu_base, c2 / (r * mu_base)
    spend = GAMMA_FJ * (b1 + b2) * floor_multiple
    r_star = optimal_ray(GAMMA_FJ, mu_base, r, c1, c2, spend)
    assert r_star == pytest.approx(expected, abs=5e-4)


def test_r_star_tends_to_one_at_the_stability_boundary():
    """The cheapest stable point has both servers just above gamma, which IS homogeneous,
    so the paper's ray is exact in the tight-budget limit (findings section 4)."""
    mu_base, r, c1, c2 = QCSC_FJ["quantum_dominant"]
    floor = GAMMA_FJ * (c1 / mu_base + c2 / (r * mu_base))
    assert optimal_ray(GAMMA_FJ, mu_base, r, c1, c2, floor * 1.000001) == \
        pytest.approx(1.0, abs=1e-3)


# findings section 4 / section 5 (probe Q7): r* against the price ratio beta_1/beta_2,
# every point measured at 6x that case's own optimal floor. Keyed by c_qpu (c_gpu = 1),
# which enters the two workloads' price ratio inversely -- `quantum_dominant` puts the QPU
# on server 1, `classical_dominant` on server 2.
FINDINGS_Q7 = {
    "quantum_dominant": [(0.25, 1.0000), (0.50, 1.2223), (1.00, 1.5298), (2.00, 1.9286),
                         (3.00, 2.2145), (4.00, 2.4447), (6.00, 2.8124), (8.00, 3.1074),
                         (16.0, 3.9527), (32.0, 5.0255)],
    "classical_dominant": [(0.25, 2.4447), (0.50, 1.9286), (1.00, 1.5298), (2.00, 1.2223),
                           (3.00, 1.0751), (4.00, 1.0000), (6.00, 0.8963), (8.00, 0.8182),
                           (16.0, 0.6537), (32.0, 0.5185)],
}


@pytest.mark.parametrize("workload,c_qpu,expected", [
    (wl, c_qpu, expected)
    for wl, rows in FINDINGS_Q7.items() for c_qpu, expected in rows
])
def test_reproduces_the_price_ratio_sweep_from_findings(workload, c_qpu, expected):
    """r* is driven by the price ratio, and r* < 1 is reachable.

    The two workloads share one curve in beta_1/beta_2 despite opposite hardware, which is
    findings section 4's "structure enters only through beta". `classical_dominant` past
    c_qpu/c_gpu = 4 is where r* crosses below 1: the nominally FASTER server is bought
    down below the slower one because it is priced above its speed advantage.
    """
    mu_base, r = 1.0, 4.0
    c1, c2 = (c_qpu, 1.0) if workload == "quantum_dominant" else (1.0, c_qpu)
    b1, b2 = c1 / mu_base, c2 / (r * mu_base)
    spend = 6.0 * GAMMA_FJ * (b1 + b2)
    assert optimal_ray(GAMMA_FJ, mu_base, r, c1, c2, spend) == \
        pytest.approx(expected, abs=5e-5)


def test_spend_below_the_optimal_floor_is_unstable_on_every_ray():
    mu_base, r, c1, c2 = QCSC_FJ["balanced"]
    floor = GAMMA_FJ * (c1 / mu_base + c2 / (r * mu_base))
    with pytest.raises(InstabilityError):
        optimal_ray(GAMMA_FJ, mu_base, r, c1, c2, spend=floor)


def test_r_star_satisfies_the_local_condition_to_full_precision():
    """r* must be pinned far tighter than a function-value minimizer can manage.

    A quadratic minimum is flat, so locating it by comparing values of `t_ul` is limited to
    about sqrt(machine epsilon) -- ~1e-8 relative. That much noise in r* lands in
    `alloc_cost` and stalls the optimizer's outer fixed point, which jitters above a 1e-9
    tolerance for several iterations after the answer has been reached.

    Asserted as the RESIDUAL of the condition being solved, not as the stability of r*
    under a nudged spend: bisection is a deterministic map, so a nudge moves its midpoint
    sequence by ~1e-13 no matter how far that sequence has converged, and a truncated
    bisection with 0.5% error in r* passes such a test easily. The residual does not --
    measured 1.8e-15 converged, 9.9e-10 at 30 halvings, 8.7e-3 at 8.

    Uses a workload whose optimum is INTERIOR. At r* = 1 the condition holds only in the
    subgradient sense, `t_bot`'s kink leaving a jump across the optimum rather than a zero.
    """
    mu_base, r, c1, c2 = QCSC_FJ["quantum_dominant"]
    b1, b2 = c1 / mu_base, c2 / (r * mu_base)
    spend = 6.0 * GAMMA_FJ * (b1 + b2)
    r_star = optimal_ray(GAMMA_FJ, mu_base, r, c1, c2, spend)
    assert r_star > 1.0 + 1e-6, "fixture must have an interior optimum, not the kink"
    # Recover the two rates from the ray and the spend line, then evaluate the condition
    # dT/dm1 = (beta_1/beta_2) * dT/dm2 that the solve claims to have satisfied.
    m1 = spend / (b1 + b2 * r_star)
    m2 = r_star * m1
    d1 = _dt_dm1(GAMMA_FJ, m1, m2)
    d2 = _dt_dm1(GAMMA_FJ, m2, m1)
    scale = abs(d1) + (b1 / b2) * abs(d2)
    assert abs(d1 - (b1 / b2) * d2) / scale < 1e-13


# --------------------------------------------------------------------------------------
# Float-arithmetic corners of the spend line. Both need a price ratio far beyond anything
# this model reaches, and both used to escape as errors from inside the optimizer loop.
# --------------------------------------------------------------------------------------

def test_extreme_price_ratio_at_the_floor_still_returns_a_stable_ray():
    """`m2` is recovered as `(spend - b1*m1)/b2`, which loses it to cancellation once
    `b1*m1` approaches `spend`: the absolute error is ~eps*m1*(b1/b2). At b1/b2 = 1e5 and
    gamma = 0.45 one ulp of `spend` is 1e-11 against a true `m2 - gamma` of 1e-12, so `m2`
    comes back BELOW gamma -- an unstable ray on a spend that admits stable ones.
    """
    gamma, mu_base, r, c1, c2 = 0.45, 1.0, 1.0, 1e5, 1.0
    b1, b2 = c1 / mu_base, c2 / (r * mu_base)
    floor = gamma * (b1 + b2)
    # `nextafter` rather than a relative factor for the tightest case. This list began
    # `1e-16`, which is below machine epsilon: `floor * (1 + 1e-16) == floor`, so that case
    # was really testing the floor ITSELF, which admits no stable ray and is now refused.
    # It passed only because the rescue ray rounded up off the boundary by one ulp.
    spends = [math.nextafter(floor, math.inf)]
    spends += [floor * (1.0 + f) for f in (1e-15, 3e-14, 1e-13, 1e-12)]
    for spend in spends:
        m1, m2 = _min_on_spend_line(gamma, b1, b2, spend)
        assert m1 > gamma and m2 > gamma, (spend, m1 - gamma, m2 - gamma)


def test_extreme_price_ratio_at_the_floor_does_not_escape_the_optimizer():
    """The same corner, through the public path: it surfaced as an InstabilityError raised
    mid-loop, from a budget the incumbent policy serves without complaint."""
    from qopt import GG1Station, Optimizer, min_feasible_budget
    from qopt.station import ForkJoinStation

    def stations():
        return [ForkJoinStation(gamma=0.45, mu=1.0, r=1.0, c1=1e5, c2=1.0,
                                r_star=R_STAR_TUNED, name="fj"),
                GG1Station.mm1(gamma=0.9, mu=1.0, c=1.0, name="q")]

    result = Optimizer(stations(), (1.0 + 1e-14) * min_feasible_budget(stations())).run()
    assert result.converged
    assert result.capacities[0] > 0.0


def test_a_price_ratio_that_collapses_the_bracket_raises_rather_than_dividing_by_zero():
    """At b1/b2 around 1e14 the interval's own endpoint is computed by subtraction and
    `b1 + b2` loses `b2`, so midpoints land past the m2 = gamma boundary where the
    derivatives divide by zero. That has to come out as this module's own error."""
    for gamma in (1e3, 1e6):
        b1, b2 = 1e16, 1.0
        try:
            m1, m2 = _min_on_spend_line(gamma, b1, b2, gamma * (b1 + b2) * 1.0000001)
        except InstabilityError:
            continue
        assert m1 > gamma and m2 > gamma, (gamma, m1, m2)


# --------------------------------------------------------------------------------------
# Input validation. `optimal_ray` is exported from the package root, so its arguments get
# the same treatment `ForkJoinStation` gives the constructor arguments they mirror --
# without it, `spend = inf` returned nan, a zero rate or cost raised a raw
# ZeroDivisionError from inside the bisection, and `gamma = nan` surfaced as an
# InstabilityError quoting a nan floor.
# --------------------------------------------------------------------------------------

GOOD = dict(gamma=0.45, mu_base=1.0, r=4.0, c1=4.0, c2=1.0, spend=10.0)

NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("arg,bad", [
    ("gamma", 0.0), ("gamma", -1.0), ("gamma", NAN), ("gamma", INF),
    ("mu_base", 0.0), ("mu_base", -1.0), ("mu_base", NAN), ("mu_base", INF),
    ("r", 0.999), ("r", 0.0), ("r", -1.0), ("r", NAN), ("r", INF),
    ("c1", 0.0), ("c1", -1.0), ("c1", NAN), ("c1", INF),
    ("c2", 0.0), ("c2", -1.0), ("c2", NAN), ("c2", INF),
    ("spend", 0.0), ("spend", -1.0), ("spend", NAN), ("spend", INF),
])
def test_optimal_ray_rejects_invalid_inputs(arg, bad):
    kwargs = dict(GOOD, **{arg: bad})
    with pytest.raises(ValueError, match=arg):
        optimal_ray(**kwargs)


def test_optimal_ray_accepts_the_boundary_values_it_documents():
    """r == 1 is legal -- identical hardware, which `ForkJoinStation` also accepts -- and so
    is a spend that only just clears the stability floor. Validation must not narrow the
    domain the solver actually handles.

    At r == 1 the two servers are identical but not equally PRICED here (beta_1 = 4 against
    beta_2 = 1), so the answer is a ray above 1, not the symmetric one -- which is the point
    of admitting r == 1 rather than assuming it degenerates.
    """
    assert optimal_ray(**dict(GOOD, r=1.0)) == pytest.approx(1.480955619083298, rel=1e-9)
    b1, b2 = GOOD["c1"] / GOOD["mu_base"], GOOD["c2"] / (GOOD["r"] * GOOD["mu_base"])
    floor = GOOD["gamma"] * (b1 + b2)
    # At the floor the ray tends to 1 (findings section 4), reached here to 1.4e-12.
    assert optimal_ray(**dict(GOOD, spend=floor * (1.0 + 1e-12))) == pytest.approx(
        1.0, rel=1e-11)


def test_optimal_ray_still_raises_instability_below_the_floor():
    """A spend that is valid but too small stays an InstabilityError, not a ValueError:
    the inputs are well formed, the station simply cannot be stabilized for that money."""
    b1, b2 = GOOD["c1"] / GOOD["mu_base"], GOOD["c2"] / (GOOD["r"] * GOOD["mu_base"])
    with pytest.raises(InstabilityError):
        optimal_ray(**dict(GOOD, spend=GOOD["gamma"] * (b1 + b2)))


def test_a_scale_the_derivative_cannot_evaluate_raises_instead_of_guessing():
    """`_dt_dm1` squares rate differences, which `t_ul` itself never does, so it runs out
    of exponent range long before the rest of the library does.

    Three failure modes, and they reach the guard by two different routes -- so all three
    are listed here rather than one standing in for the others:

    - above ~1.3e154 the square OVERFLOWS, raising OverflowError,
    - below ~1.5e-162 it underflows to zero and the reciprocal raises ZeroDivisionError,
    - between those the expression evaluates to NaN with no exception at all, and that is
      the dangerous one: bisection tests `g(mid) < 0.0`, which is False for NaN, so the
      bracket narrowed the wrong way and returned a confidently wrong ray. Verified to
      produce NaN and not an exception: `_dt_dm1(1e-155, 3e-155, 6e-155)`.
    """
    cases = [
        dict(gamma=1e155 / 3.0, mu_base=1.0, r=2.0, c1=1.0, c2=1.0, spend=1e155),
        dict(gamma=1e-160, mu_base=1.0, r=2.0, c1=1.0, c2=1.0, spend=3e-160),
        dict(gamma=1e-155, mu_base=1.0, r=1.0, c1=1.0, c2=1.0, spend=9e-155),
        # This one is the reason the NaN branch is a raise and not a shrug. With the
        # `isfinite` check removed it RETURNS, quietly, 32.9% off: 1.9117666863801601
        # against the 1.438914358249078 the identical case gives when every rate and
        # gamma is scaled up by 1e120 -- and `t_ul` is invariant under that scaling, so
        # the scaled answer is the right one. The other cases here raise either way,
        # because a NaN bracket walks down into the underflow.
        dict(gamma=1e-154, mu_base=1.0, r=4.0, c1=4.0, c2=1.0, spend=6e-154),
    ]
    assert _dt_dm1(1e-155, 3e-155, 6e-155) != _dt_dm1(1e-155, 3e-155, 6e-155)   # NaN
    for kwargs in cases:
        with pytest.raises(ValueError, match="scale"):
            optimal_ray(**kwargs)


def test_a_tuned_run_at_a_scale_the_solver_cannot_reach_fails_as_a_value_error():
    """Through the public path: the incumbent policy serves this budget, so the tuned
    policy must refuse it coherently rather than with a raw OverflowError."""
    from qopt.optimizer import Optimizer
    from qopt.station import ForkJoinStation

    kw = dict(gamma=1.0, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    assert Optimizer([ForkJoinStation(**kw, name="fj")], 1e155).run().converged
    with pytest.raises(ValueError, match="scale"):
        Optimizer([ForkJoinStation(**kw, r_star=R_STAR_TUNED, name="fj")], 1e155).run()


def test_the_unaffordable_spend_message_agrees_with_the_test_that_refused():
    """The error quotes `gamma*(beta_1+beta_2)` as the threshold, so it must be that exact
    quantity that decides. It was not: the test was `hi = (spend - b2*gamma)/b1 > gamma`,
    which rounds differently, so the message could refuse a spend strictly GREATER than the
    number it told the caller to exceed -- `0.39400156892415794` against a quoted
    `0.3940015689241579`. A caller obeying the message still got the error.
    """
    kw = dict(gamma=0.149, mu_base=2.254, r=4.236, c1=4.969, c2=4.199)
    b1 = kw["c1"] / kw["mu_base"]
    b2 = kw["c2"] / (kw["r"] * kw["mu_base"])
    floor = kw["gamma"] * (b1 + b2)
    assert optimal_ray(**kw, spend=math.nextafter(floor, math.inf)) > 0.0
    with pytest.raises(InstabilityError):
        optimal_ray(**kw, spend=floor)


def test_the_boundary_fallback_is_a_stability_rescue_not_an_optimum():
    """The `m1 = m2 = spend/(b1+b2)` fallback returns a STABLE ray, not the best one, and
    the difference is large enough that it must not be described as the optimum.

    At `b1/b2 = 1e16` the spend line is almost vertical: server 1's rate is pinned near
    gamma and the whole remaining budget buys server 2's rate, so the optimum is a ray in
    the hundreds of thousands. Collapsing to `r_star = 1` costs 37% of E[T] here. Measured
    against the best point on the same line, found in exact arithmetic.
    """
    gamma, b1, b2 = 1e3, 1e16, 1.0
    m1, m2 = _min_on_spend_line(gamma, b1, b2, gamma * (b1 + b2) * 1.0000001)
    assert m1 > gamma and m2 > gamma          # the property it does guarantee
    assert m2 / m1 == 1.0                     # and the ray it returns
    assert t_ul(gamma, m1, m2) == pytest.approx(13750.0, rel=1e-6)
    # The best ray on that same spend line, and what collapsing costs.
    assert t_ul(gamma, 1000.0000999750063, 249937766.57889298) == pytest.approx(
        10002.5, rel=1e-6)
