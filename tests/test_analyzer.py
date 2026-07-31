import pytest

from qopt.analyzer import AnalyticAnalyzer, Analyzer, Evaluation
from qopt.station import ForkJoinStation, GG1Station


def _stations():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


def test_evaluation_defaults_are_independent_per_instance():
    a = Evaluation(sojourn_times=[1.0])
    b = Evaluation(sojourn_times=[2.0])
    a.degraded.append("x")
    a.extras["k"] = 1
    assert b.degraded == []
    assert b.extras == {}
    assert a.ci is None


def test_analyzer_is_abstract():
    assert getattr(Analyzer.evaluate, "__isabstractmethod__", False)
    with pytest.raises(TypeError):
        Analyzer()  # type: ignore[abstract]


def test_analytic_analyzer_is_not_stochastic():
    assert AnalyticAnalyzer.is_stochastic is False
    assert AnalyticAnalyzer().is_stochastic is False
    assert isinstance(AnalyticAnalyzer(), Analyzer)


def test_analytic_analyzer_mirrors_sojourn_time_bitwise():
    stations = _stations()
    S = [2.5, 3.5, 3.0]
    ev = AnalyticAnalyzer().evaluate(stations, S)
    assert ev.sojourn_times == [st.sojourn_time(Si) for st, Si in zip(stations, S)]
    assert ev.ci is None
    assert ev.degraded == []
    assert ev.extras == {}


def test_analytic_analyzer_ignores_fresh_seed():
    stations = _stations()
    S = [2.5, 3.5, 3.0]
    assert (
        AnalyticAnalyzer().evaluate(stations, S, fresh_seed=True).sojourn_times
        == AnalyticAnalyzer().evaluate(stations, S).sojourn_times
    )


def test_analytic_analyzer_propagates_instability():
    from qopt.exceptions import InstabilityError

    stations = [GG1Station.mm1(gamma=1.0, mu=1.0, c=1.0, name="a")]
    with pytest.raises(InstabilityError):
        AnalyticAnalyzer().evaluate(stations, [1.0])
