import pytest

from qopt.allocator import allocate, min_feasible_budget
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
