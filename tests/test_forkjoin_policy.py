"""The locally-optimal fork-join ray (issue #10 item 3).

Reference figures come from docs/forkjoin-s2-policy/findings.md, which swept the same
quantity with an independent throwaway probe.
"""

import pytest

from qopt.exceptions import InstabilityError
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
    for factor in (1e-16, 1e-15, 3e-14, 1e-13, 1e-12):
        m1, m2 = _min_on_spend_line(gamma, b1, b2, floor * (1.0 + factor))
        assert m1 > gamma and m2 > gamma, (factor, m1 - gamma, m2 - gamma)


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
