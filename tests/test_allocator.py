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
