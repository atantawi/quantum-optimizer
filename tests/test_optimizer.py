import pytest

from qopt.allocator import min_feasible_budget
from qopt.exceptions import InfeasibleBudgetError
from qopt.optimizer import Optimizer, Result
from qopt.station import ForkJoinStation, GG1Station


def _mm1_network():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="a"),
        GG1Station.mm1(gamma=0.3, mu=2.0, c=1.0, name="b"),
    ]


def test_mm1_network_converges_with_unit_zeta():
    stations = _mm1_network()
    opt = Optimizer(stations, budget=5 * min_feasible_budget(stations))
    res = opt.run()
    assert res.converged
    assert all(z == pytest.approx(1.0, rel=1e-9) for z in res.zeta)
    # budget fully spent
    spent = sum(st.alloc_cost * S for st, S in zip(stations, res.capacities))
    assert spent == pytest.approx(opt.budget, rel=1e-9)


def test_objective_matches_weighted_sojourn_sum():
    stations = _mm1_network()
    opt = Optimizer(stations, budget=5 * min_feasible_budget(stations))
    res = opt.run()
    expected = sum(
        st.weight * st.sojourn_time(S) for st, S in zip(stations, res.capacities)
    )
    assert res.objective == pytest.approx(expected, rel=1e-12)
    assert res.sojourn_times == pytest.approx(
        [st.sojourn_time(S) for st, S in zip(stations, res.capacities)], rel=1e-12
    )


def test_mixed_network_with_md1_and_forkjoin_converges():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0),      # load-dependent zeta
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),
    ]
    opt = Optimizer(stations, budget=6 * min_feasible_budget(stations))
    res = opt.run()
    assert res.converged
    for st, S in zip(stations, res.capacities):
        assert S * st.mu > st.gamma  # stable


def test_infeasible_budget_raises():
    stations = _mm1_network()
    with pytest.raises(InfeasibleBudgetError):
        Optimizer(stations, budget=min_feasible_budget(stations)).run()


def test_nonpositive_initial_zeta_raises():
    stations = _mm1_network()
    with pytest.raises(ValueError):
        Optimizer(
            stations,
            budget=5 * min_feasible_budget(stations),
            initial_zeta=[1.0, 0.0],
        ).run()


def test_max_iter_guard_returns_not_converged():
    # tol = 0 is never satisfied by "< tol", so the loop runs to max_iter and reports False.
    stations = _mm1_network()
    opt = Optimizer(
        stations, budget=5 * min_feasible_budget(stations), tol=0.0, max_iter=3
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = opt.run()
    assert res.converged is False
    assert res.iterations == 3
    assert res.residual >= 0.0  # a real residual was recorded, not left at inf
    assert isinstance(res, Result)
    assert res.zeta == pytest.approx(
        [st.zeta(S) for st, S in zip(stations, res.capacities)], rel=1e-12
    )


def test_converged_result_reports_small_residual():
    stations = _mm1_network()
    opt = Optimizer(stations, budget=5 * min_feasible_budget(stations))
    res = opt.run()
    assert res.converged
    assert res.residual < opt.tol


def test_valid_custom_initial_zeta_is_honored_and_converges():
    stations = _mm1_network()
    opt = Optimizer(
        stations, budget=5 * min_feasible_budget(stations), initial_zeta=[1.3, 0.7]
    )
    res = opt.run()
    assert res.converged
    # M/M/1 zeta is identically 1 regardless of the (positive) starting guess.
    assert all(z == pytest.approx(1.0, rel=1e-9) for z in res.zeta)


def test_initial_zeta_wrong_length_raises():
    stations = _mm1_network()  # two stations
    with pytest.raises(ValueError, match="length"):
        Optimizer(
            stations,
            budget=5 * min_feasible_budget(stations),
            initial_zeta=[1.0],
        ).run()


@pytest.mark.parametrize("bad_budget", [float("nan"), float("inf")])
def test_non_finite_budget_raises(bad_budget):
    stations = _mm1_network()
    with pytest.raises(ValueError, match="finite"):
        Optimizer(stations, budget=bad_budget).run()


def test_nan_initial_zeta_raises():
    # NaN passes `z <= 0`, so the guard must reject it via isfinite.
    stations = _mm1_network()
    with pytest.raises(ValueError, match="finite"):
        Optimizer(
            stations,
            budget=5 * min_feasible_budget(stations),
            initial_zeta=[1.0, float("nan")],
        ).run()
