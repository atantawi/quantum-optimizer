import pytest

from qopt.allocator import allocate, min_feasible_budget, noise_floor
from qopt.exceptions import InfeasibleBudgetError
from qopt.station import ForkJoinStation, GG1Station


def test_single_station_spends_whole_budget():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    (S,) = allocate([st], C=4.0, zeta_vec=[1.0])
    assert S == pytest.approx(4.0 / 2.0, rel=1e-12)  # S = C / c = 2.0


def test_budget_fully_spent_identity():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),
        GG1Station.md1(gamma=0.3, mu=2.0, c=1.0),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),
    ]
    C = 20.0
    zeta_vec = [1.0, 0.9, 1.5]
    S = allocate(stations, C, zeta_vec)
    spent = sum(st.alloc_cost * Si for st, Si in zip(stations, S))
    assert spent == pytest.approx(C, rel=1e-12)


def test_min_feasible_budget():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),   # 2 * 0.6/1.0 = 1.2
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),  # 2 * 0.5/1.0 = 1.0
    ]
    assert min_feasible_budget(stations) == pytest.approx(1.2 + 1.0, rel=1e-12)


def test_allocate_refuses_a_budget_its_stations_cannot_support():
    """`allocate` must not return an unstable capacity. Eq 21's slack term goes negative
    when C is below the stations' CURRENT floors, and the base term `gamma/mu` is then
    reduced rather than added to, so every returned capacity is below the stability
    boundary -- silently, because eq 21 itself has no stability test.

    This is reachable through the composition of two root-exported functions: a tuned
    fork-join left on a previous run's ray prices above `min_feasible_budget`, which
    reports the floor at the ray a RUN starts from (see Station.min_spend). The helper's
    guarantee is kept by making the failure loud rather than by weakening the helper.
    """
    from qopt.exceptions import InfeasibleBudgetError
    from qopt.forkjoin_policy import R_STAR_TUNED
    from qopt.optimizer import Optimizer

    st = ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, r_star=R_STAR_TUNED)
    policy_floor = min_feasible_budget([st])
    Optimizer([st], 20.0 * policy_floor).run()
    assert st.alloc_cost * (st.gamma / st.mu) > policy_floor      # the ray moved above it

    with pytest.raises(InfeasibleBudgetError, match="2.10291"):
        allocate([st], 2.0, [st.default_zeta])

    # Exactly at the current-ray floor too: slack 0 leaves every station AT the boundary,
    # where `S*mu == gamma` and the sojourn time diverges.
    with pytest.raises(InfeasibleBudgetError):
        allocate([st], st.alloc_cost * (st.gamma / st.mu), [st.default_zeta])


def test_a_budget_above_the_reported_floor_allocates_stably():
    """The contract `min_feasible_budget` states, exercised on a mixed network: any budget
    strictly above it makes eq 21's slack positive and every capacity stable. It holds for
    every station on its starting ray, which is every station a run ever allocates for.
    """
    from qopt.forkjoin_policy import R_STAR_EQUAL_RATE, R_STAR_TUNED

    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),
        ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, r_star=R_STAR_TUNED),
        ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0,
                        r_star=R_STAR_EQUAL_RATE),
        ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=1.0, c2=20.0, r_star=0.4),
    ]
    floor = min_feasible_budget(stations)
    for mult in (1.000000001, 1.001, 1.5, 10.0):
        S = allocate(stations, mult * floor, [st.default_zeta for st in stations])
        for st, Si in zip(stations, S):
            st.check_stable(Si)          # raises InstabilityError if S*mu <= gamma


@pytest.mark.parametrize("zeta,match", [
    ([1.0, 1.0], "length"),                 # short: zip() silently dropped a station
    ([1.0, 1.0, 1.0, 1.0], "length"),       # long
    ([1.0, 1.0, 0.0], "strictly positive"),
    ([1.0, 1.0, -1.0], "strictly positive"),
    ([1.0, 1.0, float("nan")], "strictly positive"),
    ([1.0, 1.0, float("inf")], "strictly positive"),
])
def test_allocate_rejects_a_zeta_vector_it_cannot_use(zeta, match):
    """The other half of the guarantee `min_feasible_budget` states, and the half that was
    left open: a bad zeta produced silently wrong capacities rather than an error.

    A SHORT vector was the worst of them -- `zip` truncates, so `allocate` returned fewer
    capacities than there were stations and renormalized the budget across the survivors.
    A zero left its station at exactly `S*mu == gamma` with the budget far above the floor,
    which is precisely the unstable-in-silence outcome the ray guards were added to prevent.
    """
    stations = [GG1Station.mm1(gamma=0.5, mu=1.0, c=c, name=n)
                for c, n in ((1.0, "a"), (2.0, "b"), (3.0, "c"))]
    assert min_feasible_budget(stations) == pytest.approx(3.0, rel=1e-12)
    with pytest.raises(ValueError, match=match):
        allocate(stations, 10.0, zeta)


def test_allocate_rejects_a_non_finite_budget():
    """`allocate` is root-exported, so it cannot rely on the Optimizer's budget guard. The
    slack test is written `not slack > 0.0` rather than `slack <= 0.0` for exactly this: a
    NaN budget passes every ordering comparison, and used to yield NaN capacities.
    """
    stations = [GG1Station.mm1(gamma=0.5, mu=1.0, c=1.0, name="a")]
    for bad in (float("nan"), float("-inf")):
        with pytest.raises(InfeasibleBudgetError):
            allocate(stations, bad, [1.0])


def test_noise_floor_rejects_a_dzeta_of_the_wrong_length():
    """It indexes `dzeta` positionally 2n times; a short one raised a raw IndexError."""
    stations = [GG1Station.mm1(gamma=0.5, mu=1.0, c=c, name=n)
                for c, n in ((1.0, "a"), (2.0, "b"), (3.0, "c"))]
    with pytest.raises(ValueError, match="length"):
        noise_floor(stations, 10.0, [1.0, 1.0, 1.0], [0.1, 0.1])


def test_the_reported_floor_is_bit_for_bit_the_one_allocate_prices():
    """`min_feasible_budget` and eq 21's slack term must agree to the LAST BIT, or the two
    disagree over a budget one ulp wide.

    Eq 21 needs `base_i = gamma_i/mu_i` for the capacity formula and so prices the floor as
    `sum(alloc_cost_i * base_i)`; the helper is a sum of `Station.min_spend`. Written
    `alloc_cost * gamma / mu` that is `(a*g)/m` against eq 21's `a*(g/m)` -- equal for most
    inputs, one ulp apart when `mu` is not a power of two, which `r_star < 1` produces by
    scaling `mu` by the ray. The helper came out LOWER there, so a budget between the two
    passed the Optimizer's guard and then hit a non-positive slack: before `allocate`
    checked, that returned capacities below the stability boundary in silence.

    So `min_spend` carries eq 21's parenthesization deliberately, and this pins it: the
    smallest representable budget above the reported floor must allocate.
    """
    import math

    from qopt.forkjoin_policy import R_STAR_TUNED

    for stations in (
        # A single-server station is enough to expose it -- (0.7*0.1)/0.3 and 0.7*(0.1/0.3)
        # are adjacent floats -- so the base-class `min_spend` is pinned here too, not only
        # the fork-join override.
        [GG1Station.mm1(gamma=0.1, mu=0.3, c=0.7)],
        [GG1Station.mm1(gamma=0.1, mu=0.6, c=1.3),
         GG1Station.mm1(gamma=0.45, mu=1.7, c=2.5)],
        [ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=1.0, c2=20.0, r_star=0.3)],
        [ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=1.0, c2=20.0, r_star=0.7),
         GG1Station.mm1(gamma=0.9, mu=1.0, c=1.0)],
        [ForkJoinStation(gamma=0.2, mu=0.5, r=2.0, c1=1.0, c2=5.0, r_star=R_STAR_TUNED),
         GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)],
    ):
        floor = min_feasible_budget(stations)
        base = [st.gamma / st.mu for st in stations]
        assert floor == sum(st.alloc_cost * b for st, b in zip(stations, base))
        # Both sides, and this matters: comparing the helper only against a hand-copy of
        # eq 21's expression pins one side of a two-sided agreement, and regrouping
        # `allocate`'s OWN floor then goes unnoticed. These two bracket it -- the smallest
        # budget above the floor must allocate, and the floor itself must not.
        S = allocate(stations, math.nextafter(floor, math.inf),
                     [st.default_zeta for st in stations])
        assert all(Si > 0.0 for Si in S)
        with pytest.raises(InfeasibleBudgetError):
            allocate(stations, floor, [st.default_zeta for st in stations])


def test_weight_scales_allocation_above_base():
    # Two stations identical except weight. Since S_i = base_i + slack * num_i / denom
    # with num_i = sqrt(w_i * z / (c * mu)) and base/slack/denom shared, the capacity
    # *above base* scales as sqrt(w). weights 1 and 4 => the second gets exactly 2x.
    a = GG1Station.mm1(gamma=0.5, mu=1.0, weight=1.0, c=1.0)
    b = GG1Station.mm1(gamma=0.5, mu=1.0, weight=4.0, c=1.0)
    Sa, Sb = allocate([a, b], C=6.0, zeta_vec=[1.0, 1.0])
    base = 0.5  # gamma/mu, same for both
    assert Sb > Sa  # higher weight -> more capacity
    assert (Sb - base) == pytest.approx(2.0 * (Sa - base), rel=1e-12)


def test_objective_reflects_weights():
    # A non-unit weight must multiply through the objective, not be dropped.
    from qopt.optimizer import Optimizer

    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, weight=3.0, c=2.0),
        GG1Station.mm1(gamma=0.3, mu=2.0, weight=1.0, c=1.0),
    ]
    res = Optimizer(stations, budget=5 * min_feasible_budget(stations)).run()
    expected = sum(w * t for w, t in zip((3.0, 1.0), res.sojourn_times))
    assert res.objective == pytest.approx(expected, rel=1e-12)
    # and it differs from the unweighted sum, proving the weight is live
    assert res.objective != pytest.approx(sum(res.sojourn_times), rel=1e-6)


def test_all_stations_stable_under_feasible_budget():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),
        GG1Station.md1(gamma=0.3, mu=2.0, c=1.0),
    ]
    C = 3 * min_feasible_budget(stations)
    S = allocate(stations, C, [st.default_zeta for st in stations])
    for st, Si in zip(stations, S):
        assert Si * st.mu > st.gamma


def test_an_extreme_weight_ratio_raises_rather_than_returning_a_boundary_capacity():
    """A lopsided weight vector rounds the light station's share away at budgets FAR above
    the floor, and for a station that cannot be priced at rho == 1 the contract is that this
    is an error, not a number.

    `min_feasible_budget` once described this as happening only "within a few ulps of the
    floor". It is really set by the share-to-base ratio, so weight skew reaches it at any
    budget. TWO INDEPENDENT mechanisms get there, and an earlier version of this test
    conflated them: it measured its whole grid on a cov = 0 light station and reported the
    result as eq 21's rounding, which only the first of these is.

    EQ 21's ROUNDING is calibration-independent, and the clean way to see that is at cov = 1,
    where the two modes are bit-identical -- phi == 1, so slope's `phi*T*x` IS level's `T*x`.
    Both then need a weight ratio of 1e30 AND a budget at 1.01x the floor; from 2x upwards the
    share survives, down to x = 2.0e-15. Two calibrations failing on exactly the same cell is
    what rules the calibration out as the cause. This is the arm that pins the FAILS-LOUDLY
    contract: an M/M/1 at `S*mu == gamma` has no sojourn time, so the run names the station it
    could not keep stable instead of reporting an objective computed there.

    The cov = 0 collapse used to raise here too and no longer does, deliberately: such a
    station's E[T] is finite at rho == 1 and the boundary is a capacity it can be priced at,
    so the same collapse is a legitimate answer rather than a failure. That half moved to
    test_a_deterministic_station_may_sit_exactly_on_its_boundary, which also covers why the
    collapse reaches so much further under slope calibration. What stays here is the check
    that opening the domain for k = 0 did NOT open it for k > 0.
    """
    from qopt.exceptions import InstabilityError
    from qopt.optimizer import Optimizer
    from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE

    def run(cov, weight_ratio, multiple, mode):
        light = GG1Station(gamma=1.0, mu=2.0, weight=1.0, c=1.0, cov_a=cov, cov_s=cov,
                           name="light", zeta_mode=mode)
        heavy = GG1Station.mm1(gamma=1.0, mu=2.0, weight=weight_ratio, c=1.0, name="heavy",
                               zeta_mode=mode)
        stations = [light, heavy]
        C = multiple * min_feasible_budget(stations)
        return stations, Optimizer(stations, C).run()

    for mode in (ZETA_SLOPE, ZETA_LEVEL):
        # Eq 21's rounding at cov = 1, where the two calibrations are the same arithmetic.
        # k = 1, so `admits_full_utilization` is False and the boundary is still refused.
        with pytest.raises(InstabilityError, match="light"):
            run(1.0, 1e30, 1.01, mode)
        # One doubling of the budget is enough to save the share -- barely.
        stations, res = run(1.0, 1e30, 2.0, mode)
        x = res.capacities[0] * stations[0].mu - stations[0].gamma
        assert 0.0 < x < 1e-14, x
        # A ratio of 1e20 does not reach it at the three budgets checked here -- the point
        # being that the ratio, not the budget, is what this arm is about.
        for multiple in (1.01, 100.0, 10000.0):
            stations, res = run(1.0, 1e20, multiple, mode)
            assert res.capacities[0] * stations[0].mu > stations[0].gamma

    # The same networks with a sane weight ratio are fine under both calibrations, so the
    # test is about the ratio and not about these stations being unservable.
    for mode in (ZETA_SLOPE, ZETA_LEVEL):
        for cov in (0.0, 1.0):
            stations, res = run(cov, 10.0, 100.0, mode)
            for st, Si in zip(stations, res.capacities):
                st.check_stable(Si)


def test_a_deterministic_station_may_sit_exactly_on_its_boundary():
    """`S*mu == gamma` is a point of a cov = 0 station's domain, not its boundary, and eq 21
    is allowed to put it there.

    A deterministic station has `E[T] = 1/(S*mu)` exactly -- k multiplies the whole
    Allen-Cunneen congestion term, so at k = 0 there is no `1/(1-rho)` left to diverge. E[T] is
    finite and smooth at rho == 1 with a bounded derivative. The stability guard used to refuse
    the point anyway, and that refusal cost an OPTIMUM rather than a corner case: E[T] is
    strictly decreasing with no asymptote, so a weighted objective's infimum over a budget
    simplex can lie exactly there, and slope-calibrated zeta converges onto it.

    Reference network, from the review that found this: a deterministic station at
    gamma=0.6/mu=1.0/weight=1 against an M/M/1 at gamma=1.2/mu=3.0/weight=3e5/c=0.5, budget
    1.01x the floor. The infimum is 6250001.666666654, attained AT the boundary. Slope
    calibration returns exactly that. Level calibration stops 2.5e-9 short of the boundary and
    reports 6250003.607; the gap is 3e-7 relative here and worth 1.9% further out
    (test_slope_beats_level_on_a_deterministic_station).

    Slope gets there and level does not because of the ORDER in which zeta vanishes: phi is
    exactly `1 - rho` at k = 0, so `zeta_slope = phi*T*x` goes as x^2 while
    `zeta_level = T*x` goes as x. Shares go as sqrt(zeta), so slope's share map is linear in x
    -- a contraction whose fixed point IS the boundary -- and level's goes as sqrt(x), whose
    fixed point is a small positive x. For any k > 0 neither collapses, because zeta tends to
    `k*rho` instead of to zero
    (test_zeta_is_bounded_away_from_zero_at_the_boundary_unless_k_is_zero).

    Zero is also what `zeta_from` reports there, and that is load-bearing: it makes x = 0 an
    exact fixed point of the loop. Clamping to ZETA_FLOOR instead was measured and rejected --
    it buys 3.6e-11 of spare capacity whose own zeta clamps again, and the map enters a
    period-3 cycle (0 -> 3.6e-11 -> 2.1e-15 -> 0) that the loop stops on by tolerance, at
    whichever point it happens to reach.
    """
    import math

    from qopt.optimizer import Optimizer
    from qopt.station import Station
    from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE

    dd = GG1Station(0.6, 1.0, 1.0, c=1.0, cov_a=0.0, cov_s=0.0, name="dd")
    mm = GG1Station.mm1(1.2, 3.0, 3e5, c=0.5, name="mm")
    assert dd.admits_full_utilization is True
    assert mm.admits_full_utilization is False
    # Not a GG1-only opt-in: the base class refuses by default, and the fork-join inherits
    # that refusal because both its branches are M/M/1.
    assert Station.admits_full_utilization.fget(mm) is False
    assert ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0,
                           c2=1.0).admits_full_utilization is False

    # The boundary is priceable, and priced by the closed form rather than by a limit.
    boundary = dd.gamma / dd.mu
    dd.check_stable(boundary)                       # does not raise
    assert dd.sojourn_time(boundary) == 1.0 / (boundary * dd.mu)
    assert dd.dT_dS(boundary) == -dd.mu / (boundary * dd.mu) ** 2
    assert dd.phi(boundary) == 0.0                  # phi carries the factor x
    # ... and still refused one ulp BELOW it, where E[T] would price a capacity the station
    # cannot serve. Opening the domain must not open it past rho == 1.
    from qopt.exceptions import InstabilityError
    with pytest.raises(InstabilityError, match="dd"):
        dd.check_stable(math.nextafter(boundary, 0.0))

    def net(mode):
        return [GG1Station(0.6, 1.0, 1.0, c=1.0, cov_a=0.0, cov_s=0.0, zeta_mode=mode,
                           name="dd"),
                GG1Station.mm1(1.2, 3.0, 3e5, c=0.5, zeta_mode=mode, name="mm")]

    stations = net(ZETA_SLOPE)
    C = 1.01 * min_feasible_budget(stations)
    res = Optimizer(stations, C).run()
    assert res.converged
    assert res.capacities[0] * stations[0].mu == stations[0].gamma      # exactly on it
    assert res.zeta[0] == 0.0
    assert res.zeta_phi[0] == 0.0
    assert res.sojourn_times[0] == pytest.approx(1.0 / 0.6, rel=1e-15)
    assert res.objective == pytest.approx(6250001.666666654, rel=1e-12)

    # Level reaches a strictly interior point, and a worse one. Both halves matter: the
    # sqrt(x) share map is why it does not collapse, and the objective is why that is not
    # something to prefer.
    lvl_stations = net(ZETA_LEVEL)
    lvl = Optimizer(lvl_stations, C).run()
    assert lvl.capacities[0] * lvl_stations[0].mu > lvl_stations[0].gamma
    assert lvl.objective > res.objective

    # Zero zeta is accepted from this station and from no other, so the silent-instability
    # hole `allocate`'s check closed stays closed.
    from qopt.allocator import allocate
    S = allocate(stations, C, [0.0, 1.0])
    assert S[0] == stations[0].gamma / stations[0].mu
    with pytest.raises(ValueError, match="strictly positive"):
        allocate(stations, C, [1.0, 0.0])           # the M/M/1 in position 1

    # x = 0 is an exact FIXED POINT of the loop map, not a cycle the tolerance hides.
    z = [0.0, 1.0]
    for _ in range(5):
        S = allocate(stations, C, z)
        assert S[0] * stations[0].mu == stations[0].gamma
        z = [st.zeta_from(st.sojourn_time(Si), Si) for st, Si in zip(stations, S)]
        assert z[0] == 0.0


def test_slope_beats_level_on_a_deterministic_station():
    """Why the domain was opened rather than slope mode restricted: slope is the ACCURATE one.

    The review that found the boundary collapse read it as a slope-mode regression, on the
    grounds that level mode succeeds where slope raises. It does, but by stopping short of the
    optimum -- the collapse was slope converging onto a true infimum that the guard forbade.
    Brute-forced against a 400k-point grid on S_dd over the budget line, on the review's own
    network at three budgets:

        C = 1.01x floor, w = 3e5:  infimum 6250001.667   slope 6250001.667   level 6250003.607
        C = 3x    floor, w = 1e3:  infimum     105.8333  slope     105.8333  level     107.3484
        C = 10x   floor, w = 1e3:  infimum      24.81481 slope      24.81481 level      25.29338

    Level's excess is 1.43% and 1.93% on the last two, and it gets there by over-allocating the
    zero-variability station -- 0.912 against the optimal 0.600 at 10x -- because eq 22's level
    zeta overstates how much a deterministic station gains from capacity. That is the whole
    point of calibrating on the slope instead.

    Slope does NOT attain the infimum exactly at every budget, and the assertions below say so:
    across the six cells its relative excess runs from -5.8e-15 to 1.1e-10, the largest miss
    being the one cell where it stops 6.3e-09 of capacity short of the boundary rather than on
    it. Level's smallest excess over the same cells is 3.1e-07, so the two are separated by
    more than three orders of magnitude even at their closest.
    """
    from qopt.optimizer import Optimizer
    from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE

    def net(mode, w):
        return [GG1Station(0.6, 1.0, 1.0, c=1.0, cov_a=0.0, cov_s=0.0, zeta_mode=mode,
                           name="dd"),
                GG1Station.mm1(1.2, 3.0, w, c=0.5, zeta_mode=mode, name="mm")]

    def infimum(C, w):
        """The objective at S_dd = gamma/mu, which is where E[T]_dd is minimised on the line."""
        S1 = 0.6
        S2 = (C - S1) / 0.5
        m = 3.0 * S2
        rho = 1.2 / m
        return 1.0 / S1 + w * (1.0 / m) * (1.0 + rho / (1.0 - rho))

    cells = (
        # multiple, weight, level's measured relative excess over the infimum
        (1.01, 3e5, 3.1042e-07),
        (1.01, 1e3, 8.0020e-05),
        (3.0, 3e5, 5.3313e-05),
        (3.0, 1e3, 1.4316e-02),
        (10.0, 3e5, 2.3851e-04),
        (10.0, 1e3, 1.9285e-02),
    )
    for mult, w, level_excess in cells:
        C = mult * min_feasible_budget(net(ZETA_LEVEL, w))
        best = infimum(C, w)
        slope = Optimizer(net(ZETA_SLOPE, w), C).run()
        level = Optimizer(net(ZETA_LEVEL, w), C).run()
        # Slope reaches the infimum to floating-point noise. Two-sided and NOT exact: the
        # largest measured miss is 1.1e-10 (at 10x/1e3, where it stops 6.3e-09 short of the
        # boundary) and the smallest is -5.8e-15, the infimum being recomputed here by a
        # different expression than the optimizer uses. 1e-09 gives one decade of headroom
        # over the worst cell while sitting two and a half decades below level's smallest
        # excess (3.1e-07), so no cell can pass both this assertion and the next.
        assert abs(slope.objective / best - 1.0) < 1e-09, (mult, w, slope.objective, best)
        # Level overshoots, by the margin measured for that cell.
        assert level.objective / best - 1.0 == pytest.approx(level_excess, rel=0.02), (
            mult, w, level.objective / best - 1.0
        )
        assert level.capacities[0] > slope.capacities[0]


def test_a_collapsed_share_can_land_on_a_technically_stable_capacity():
    """The same eq 21 collapse, on a station where it does NOT raise -- the silent half.

    test_an_extreme_weight_ratio_raises_rather_than_returning_a_boundary_capacity pins the
    loud outcome: `b * mu == gamma` exactly, station unstable, `sojourn_time` refuses. That
    depends on how `gamma/mu` rounds, and for gamma=0.1, mu=0.39 it rounds the other way --
    `b * mu` lands 1.4e-17 ABOVE gamma. The station is then technically stable, so nothing
    refuses: E[T] comes back finite and eq 22 calibrates a zeta from it.

    Pinned because `min_feasible_budget` used to promise "an error rather than a wrong
    number" for this collapse without qualification, and this is the counterexample. It is
    an accepted limitation, not a latent fix: the capacity is what the caller's budget and
    weights bought, and both candidate remedies are ruled out -- nudging S up spends budget
    that was not allocated, and raising on collapse contradicts
    test_the_reported_floor_is_bit_for_bit_the_one_allocate_prices, which requires the
    smallest representable budget above the floor to allocate. If a future change makes this
    raise, that is an improvement and this test should be rewritten to say so, not deleted.
    """
    import math

    light = GG1Station.mm1(gamma=0.1, mu=0.39, c=1.0, weight=1e-30, name="light")
    heavy = GG1Station.mm1(gamma=0.1, mu=0.39, c=1.0, weight=1.0, name="heavy")
    stations = [light, heavy]
    base = 0.1 / 0.39

    # The rounding that makes this the silent case rather than the loud one.
    assert base * light.mu > light.gamma
    assert base * light.mu - light.gamma == pytest.approx(1.39e-17, rel=1e-2)

    C = 1.01 * min_feasible_budget(stations)
    S_light, _ = allocate(stations, C, [st.default_zeta for st in stations])
    assert S_light == base                      # the share rounded away entirely
    assert S_light - base < math.ulp(base)      # and it was not merely small

    # No exception, and a finite number an unsuspecting caller would use.
    T = light.sojourn_time(S_light)
    assert math.isfinite(T) and T > 1e16
    assert math.isfinite(light.phi(S_light))
