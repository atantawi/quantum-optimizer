import pytest

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul
from qopt.forkjoin_policy import (
    R_STAR_EQUAL_RATE,
    R_STAR_INVARIANT_R,
    R_STAR_TUNED,
    optimal_ray,
)
from qopt.allocator import min_feasible_budget
from qopt.station import (
    ForkJoinStation,
    GG1Station,
    Station,
)


def test_alloc_cost_is_sum_of_both_servers():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=3.0)
    assert st.alloc_cost == pytest.approx(4.0)
    assert st.default_zeta == pytest.approx(1.5)
    assert isinstance(st, Station)


def test_sojourn_uses_t_ul_with_both_effective_rates():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    # m1 = S*mu = 1.0 (slower), m2 = S*r*mu = 2.0 (faster)
    assert st.sojourn_time(1.0) == pytest.approx(t_ul(0.6, 1.0, 2.0), rel=1e-12)


def test_homogeneous_forkjoin_matches_nelson_tantawi():
    st = ForkJoinStation(gamma=0.5, mu=1.0, r=1.0, c1=1.0, c2=1.0)
    rho = 0.5
    expected = (12 - rho) / (8 * (1.0 - 0.5))  # 2.875
    assert st.sojourn_time(1.0) == pytest.approx(expected, rel=1e-12)


def test_zeta_uses_slower_server_rate():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    expected = st.sojourn_time(1.0) * (1.0 * 1.0 - 0.6)
    assert st.zeta(1.0) == pytest.approx(expected, rel=1e-12)


def test_unstable_raises_on_slower_server():
    st = ForkJoinStation(gamma=1.0, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    with pytest.raises(InstabilityError):
        st.sojourn_time(1.0)  # S*mu = 1.0 == gamma (slower server binds)


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0.6, mu=1.0, r=0.9, c1=1.0, c2=1.0),   # r < 1
    dict(gamma=0.6, mu=1.0, r=2.0, c1=0.0, c2=1.0),   # c1 <= 0
    dict(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=-1.0),  # c2 <= 0
])
def test_forkjoin_validation(kwargs):
    with pytest.raises(ValueError):
        ForkJoinStation(**kwargs)


# --------------------------------------------------------------------------------------
# The r_star family (issue #10). The two effective rates lie on a ray m2 = r_star*m1;
# both incumbent policies are members, at r_star = r (qopt) and r_star = 1 (the paper).
# --------------------------------------------------------------------------------------

def test_r_star_defaults_to_r_which_is_the_incumbent_qopt_policy():
    base = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=3.0)
    explicit = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=3.0, r_star=2.0)
    assert base.r_star == 2.0
    assert (base.mu, base.r, base.alloc_cost) == (explicit.mu, explicit.r,
                                                  explicit.alloc_cost)
    assert base.alloc_cost == pytest.approx(4.0)          # c1 + c2*r_star/r = 1 + 3*2/2
    # Equal capacity on both servers is what r_star = r MEANS, not a general property.
    assert base.server_capacities(3.0) == pytest.approx((3.0, 3.0))


def test_r_star_one_recovers_the_papers_rule():
    """S_2 = S_1/r: both servers land on the same effective rate, priced c1 + c2/r."""
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=4.0, c1=1.0, c2=3.0, r_star=1.0)
    assert st.alloc_cost == pytest.approx(1.0 + 3.0 / 4.0)
    assert st.sojourn_time(1.0) == pytest.approx(t_ul(0.6, 1.0, 1.0), rel=1e-12)
    assert st.server_capacities(1.0) == pytest.approx((1.0, 0.25))


def test_alloc_cost_prices_the_chosen_ray():
    st = ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, r_star=2.32)
    assert st.alloc_cost == pytest.approx(4.0 + 1.0 * 2.32 / 4.0)
    S = 2.0
    assert st.sojourn_time(S) == pytest.approx(t_ul(0.45, S, 2.32 * S), rel=1e-12)


def test_server_capacities_are_the_capacities_behind_the_effective_rates():
    """Ties the money space to the rate space: S_1, S_2 must be the per-server capacities
    that produce sojourn_time's two rates through the CONSTRUCTED hardware params."""
    mu_1, r, gamma = 1.3, 4.0, 0.45
    for r_star in (0.5, 1.0, 2.32, 4.0):
        st = ForkJoinStation(gamma=gamma, mu=mu_1, r=r, c1=4.0, c2=1.0, r_star=r_star)
        S = 1.7
        S1, S2 = st.server_capacities(S)
        m1, m2 = S1 * mu_1, S2 * r * mu_1
        assert m2 == pytest.approx(r_star * m1, rel=1e-12), r_star
        assert st.sojourn_time(S) == pytest.approx(t_ul(gamma, m1, m2), rel=1e-12), r_star
        # And the budget column must charge exactly what those two capacities cost.
        assert st.alloc_cost * S == pytest.approx(st.c1 * S1 + st.c2 * S2, rel=1e-12)


def test_r_star_below_one_anchors_mu_on_the_binding_server():
    """r_star < 1 makes server 2 the effectively slower one. `mu` is documented as the
    binding rate and the allocator reads it directly, so it must follow the swap."""
    st = ForkJoinStation(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0, r_star=0.5)
    assert st.mu == pytest.approx(0.5)     # 1.0 * min(1, r_star)
    assert st.r == pytest.approx(2.0)      # effective faster/slower ratio
    assert st.r_base == 4.0                # the constructed hardware ratio survives
    assert st.sojourn_time(1.0) == pytest.approx(t_ul(0.45, 1.0, 0.5), rel=1e-12)
    assert st.server_capacities(1.0) == pytest.approx((1.0, 0.125))


def test_r_star_below_one_binds_stability_on_server_two():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=4.0, c1=1.0, c2=1.0, r_star=0.5)
    # At S = 1.0 server 1 runs at m1 = 1.0 > gamma but server 2 at m2 = 0.5 < gamma.
    # An unanchored mu would test 1.0 > 0.6 and wave this through.
    with pytest.raises(InstabilityError):
        st.check_stable(1.0)
    with pytest.raises(InstabilityError):
        st.sojourn_time(1.0)
    st.check_stable(1.5)          # m2 = 0.75 > 0.6: both servers stable
    assert st.sojourn_time(1.5) == pytest.approx(t_ul(0.6, 1.5, 0.75), rel=1e-12)


def test_min_feasible_budget_follows_the_ray():
    from qopt.allocator import min_feasible_budget

    kw = dict(gamma=0.6, mu=1.0, r=4.0, c1=1.0, c2=1.0)
    # r_star = r: alloc_cost 2.0, mu 1.0 -> 2.0 * 0.6/1.0
    assert min_feasible_budget([ForkJoinStation(**kw)]) == pytest.approx(1.2)
    # r_star = 0.5: alloc_cost 1.125, mu 0.5 -> 1.125 * 0.6/0.5
    assert min_feasible_budget(
        [ForkJoinStation(**kw, r_star=0.5)]) == pytest.approx(1.35)


def test_sim_node_emits_the_ray_it_runs():
    def rates_of(st, S):
        node = st.sim_node(S, "job")
        return [b["service"]["job"]["distribution"]["rate"] for b in node["branches"]]

    kw = dict(gamma=0.45, mu=1.0, r=4.0, c1=1.0, c2=1.0, name="fj")
    # r_star >= 1 keeps server 1 slower: branches are (m1, m2) = (S*mu, r_star*S*mu).
    assert rates_of(ForkJoinStation(**kw, r_star=2.0), 2.0) == pytest.approx([2.0, 4.0])
    # r_star < 1 swaps them; branches stay ordered slower-first, so the ray {2.0, 1.0}
    # is emitted as [1.0, 2.0].
    assert rates_of(ForkJoinStation(**kw, r_star=0.5), 2.0) == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize("r_star", [0.0, -1.0, float("nan"), float("inf")])
def test_r_star_validation(r_star):
    with pytest.raises(ValueError):
        ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=1.0, r_star=r_star)


# --------------------------------------------------------------------------------------
# Named r_star policies and the retune hook (issue #10 item 3). `r_star` carries either a
# fixed ray (a float) or one of three named policies.
# --------------------------------------------------------------------------------------

FJ = dict(gamma=0.45, mu=1.0, r=4.0, c1=4.0, c2=1.0)


def test_invariant_r_is_the_default_and_names_the_incumbent_policy():
    named = ForkJoinStation(**FJ, r_star=R_STAR_INVARIANT_R)
    assert (named.r_star, named.mu, named.r, named.alloc_cost) == (4.0, 1.0, 4.0, 5.0)
    assert named.policy == R_STAR_INVARIANT_R
    assert ForkJoinStation(**FJ).policy == R_STAR_INVARIANT_R


def test_equal_rate_names_the_papers_policy():
    st = ForkJoinStation(**FJ, r_star=R_STAR_EQUAL_RATE)
    assert st.r_star == 1.0
    assert st.policy == R_STAR_EQUAL_RATE
    assert st.alloc_cost == pytest.approx(4.0 + 1.0 / 4.0)   # c1 + c2/r
    assert st.server_capacities(2.0) == pytest.approx((2.0, 0.5))


def test_a_float_r_star_is_a_fixed_ray():
    st = ForkJoinStation(**FJ, r_star=2.32)
    assert st.policy == "fixed"


def test_a_tuned_station_starts_on_the_floor_minimizing_ray():
    """`tuned` must start at r_star = 1, not at the incumbent r.

    Not a preference. A station's stability floor over the family,
    `gamma*(c1 + c2*r_star/r) / (mu*min(1, r_star))`, is minimized at exactly r_star = 1,
    and the Optimizer evaluates `min_feasible_budget` ONCE, before any retune. Starting at
    r would therefore advertise the incumbent's floor for a station that can reach any ray.
    """
    tuned = ForkJoinStation(**FJ, r_star=R_STAR_TUNED)
    assert tuned.policy == R_STAR_TUNED
    assert tuned.r_star == 1.0
    paper = ForkJoinStation(**FJ, r_star=R_STAR_EQUAL_RATE)
    assert (tuned.mu, tuned.r, tuned.alloc_cost) == \
        (paper.mu, paper.r, paper.alloc_cost)


def test_tuned_floor_is_the_minimum_over_every_ray():
    """The floor a tuned station advertises must be no larger than any fixed ray's."""
    tuned = min_feasible_budget([ForkJoinStation(**FJ, r_star=R_STAR_TUNED)])
    for r_star in (0.25, 0.5, 1.0, 1.7, 2.32, 4.0, 9.0):
        assert tuned <= min_feasible_budget([ForkJoinStation(**FJ, r_star=r_star)]) + 0.0
    assert tuned == pytest.approx(
        min_feasible_budget([ForkJoinStation(**FJ, r_star=1.0)]), rel=1e-15)


# A tuned station whose optimum lies BELOW 1: the expensive server is server 2, priced
# above its speed advantage (c2/c1 > r). Every other tuned fixture in the suite lands
# above 1, where min(1, r_star) == 1.0 makes the re-anchoring of `mu` a no-op -- so
# without this the branch that keeps `_check_stable` guarding the binding server, and the
# failure it exists to prevent, are never exercised.
FJ_LOW = dict(gamma=0.45, mu=1.0, r=4.0, c1=1.0, c2=20.0)


def test_retune_below_one_re_anchors_mu_onto_the_newly_binding_server():
    st = ForkJoinStation(**FJ_LOW, r_star=R_STAR_TUNED)
    S_new = st.retune(3.0)
    assert st.r_star < 1.0
    # `mu` must follow the ray onto server 2, which the ray now leaves slower.
    assert st.mu == pytest.approx(st.mu_base * st.r_star, rel=1e-12)
    assert st.mu < st.mu_base
    assert st.r == pytest.approx(1.0 / st.r_star, rel=1e-12)
    # And the guard must now bind on that server: S*mu is the SMALLER of the two rates.
    m_slow, m_fast = S_new * st.mu, S_new * st.mu * st.r
    assert m_slow < m_fast
    assert m_slow > st.gamma
    st.check_stable(S_new)


def test_a_tuned_station_can_cross_below_one_during_an_optimizer_run():
    """The r_star < 1 branch has to be reachable through the public path, not just by
    calling retune by hand."""
    from qopt.allocator import min_feasible_budget as _floor
    from qopt.optimizer import Optimizer

    stations = [ForkJoinStation(**FJ_LOW, r_star=R_STAR_TUNED, name="fj"),
                GG1Station.mm1(gamma=0.9, mu=1.0, c=1.0, name="q")]
    result = Optimizer(stations, 4.0 * _floor(stations)).run()
    assert result.converged
    assert stations[0].r_star < 1.0
    assert stations[0].mu < stations[0].mu_base


@pytest.mark.parametrize("bad", ["", "auto", "tune", "invariant_r", "EQUAL-RATE"])
def test_unknown_policy_name_is_rejected(bad):
    with pytest.raises(ValueError, match="r_star"):
        ForkJoinStation(**FJ, r_star=bad)


def test_retune_is_a_no_op_on_every_policy_but_tuned():
    for r_star in (None, R_STAR_INVARIANT_R, R_STAR_EQUAL_RATE, 2.32):
        st = ForkJoinStation(**FJ, r_star=r_star)
        before = (st.r_star, st.mu, st.r, st.alloc_cost)
        assert st.retune(3.0) == 3.0
        assert (st.r_star, st.mu, st.r, st.alloc_cost) == before


def test_retune_moves_a_tuned_station_onto_the_locally_optimal_ray():
    st = ForkJoinStation(**FJ, r_star=R_STAR_TUNED)
    S = 3.0
    spend = S * st.alloc_cost
    S_new = st.retune(S)
    assert st.r_star == pytest.approx(
        optimal_ray(0.45, 1.0, 4.0, 4.0, 1.0, spend), rel=1e-12)
    # The ray changed, so mu and r must be re-anchored on the newly binding server.
    k = min(1.0, st.r_star)
    assert st.mu == pytest.approx(st.mu_base * k, rel=1e-12)
    assert st.r == pytest.approx(max(1.0, st.r_star) / k, rel=1e-12)
    assert st.policy == R_STAR_TUNED          # retuning does not consume the policy


def test_retune_preserves_the_stations_spend_exactly():
    """Eq 21 hands out a capacity priced at the OLD alloc_cost. Retuning reprices the
    station, so `retune` returns the capacity that buys the same spend at the new price --
    that is what keeps the budget satisfied and keeps S meaning "server 1 runs at S*mu".
    """
    for S in (2.0, 3.0, 7.5):
        st = ForkJoinStation(**FJ, r_star=R_STAR_TUNED)
        spend = S * st.alloc_cost
        assert st.retune(S) * st.alloc_cost == pytest.approx(spend, rel=1e-12)


def test_retune_below_the_optimal_floor_raises():
    st = ForkJoinStation(**FJ, r_star=R_STAR_TUNED)
    with pytest.raises(InstabilityError):
        st.retune(0.45 * (4.0 + 0.25) / st.alloc_cost)   # exactly the optimal floor


def test_mu_base_is_the_constructed_server_one_rate():
    """Tuning has to recompute `mu` from the hardware, and `mu` itself is already the
    anchored (effective) rate -- so the constructed rate has to survive separately, the
    way `r_base` does."""
    assert ForkJoinStation(**FJ, r_star=0.5).mu_base == 1.0
