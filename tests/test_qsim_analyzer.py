import pytest

from conftest import FakeTransport
from qopt.exceptions import InstabilityError, SimulationQualityError
from qopt.network import Network, Route
from qopt.qsim.analyzer import FRESH_SEED_OFFSET, SimulationAnalyzer
from qopt.qsim.client import QsimClient
from qopt.station import ForkJoinStation, GG1Station


def _network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def _healthy(sim_response, **kwargs):
    """A response whose throughput brackets the derived gammas (0.6, 0.4, 0.5)."""
    return sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.6, "md1": 0.4, "fj": 0.5},
        system=1.16,
        **kwargs,
    )


def _analyzer(network, response, **kwargs):
    transport = FakeTransport((200, response))
    client = QsimClient("http://qsim.test", transport=transport)
    return SimulationAnalyzer(network, client, **kwargs), transport


S_OK = [3.0, 4.0, 5.0]


def test_is_stochastic():
    assert SimulationAnalyzer.is_stochastic is True


def test_evaluate_returns_sojourn_times_ci_and_extras(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.sojourn_times == [0.42, 0.29, 0.45]
    # Compared pairwise via pytest.approx, not one list == [...] literal: 0.29 - 0.01
    # is 0.27999999999999997 in float64, one ULP off the 0.28 literal (see the same
    # workaround in tests/test_qsim_measures.py).
    assert len(ev.ci) == 3
    assert ev.ci[0] == pytest.approx((0.41, 0.43))
    assert ev.ci[1] == pytest.approx((0.28, 0.30))
    assert ev.ci[2] == pytest.approx((0.44, 0.46))
    assert ev.degraded == []
    assert ev.extras["system_response_time"] == (1.16, (1.15, 1.17))
    assert ev.extras["seed"] == 20260729
    assert ev.extras["wallClockSeconds"] == 8.3
    assert len(transport.requests) == 1


def test_evaluate_sends_the_model_at_the_given_capacities(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    analyzer.evaluate(network.stations, S_OK)
    request = transport.requests[0]
    assert request["model"] == network.to_model_dict(S_OK)
    assert request["measures"] == [
        "response-time", "system-response-time", "throughput"
    ]
    assert request["stopping"]["maxWallClockSeconds"] == 120


def test_instability_is_caught_before_the_post(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    # mm1 needs S*mu > 0.6; 0.5 saturates it.
    with pytest.raises(InstabilityError):
        analyzer.evaluate(network.stations, [0.5, 4.0, 5.0])
    assert transport.requests == []          # no simulation time was spent


def test_fixed_seed_policy_repeats_one_seed(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response), seed=11)
    for _ in range(3):
        analyzer.evaluate(network.stations, S_OK)
    assert [r["seed"] for r in transport.requests] == [11, 11, 11]


def test_vary_seed_policy_advances_per_iteration(sim_response):
    network = _network()
    analyzer, transport = _analyzer(
        network, _healthy(sim_response), seed=11, seed_policy="vary"
    )
    for _ in range(3):
        analyzer.evaluate(network.stations, S_OK)
    assert [r["seed"] for r in transport.requests] == [11, 12, 13]


def test_none_seed_policy_omits_the_seed(sim_response):
    network = _network()
    analyzer, transport = _analyzer(
        network, _healthy(sim_response), seed_policy=None
    )
    analyzer.evaluate(network.stations, S_OK)
    assert "seed" not in transport.requests[0]


def test_fresh_seed_is_offset_and_does_not_advance_the_counter(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response), seed=11,
                                    seed_policy="vary")
    analyzer.evaluate(network.stations, S_OK)              # seed 11, iteration -> 1
    analyzer.evaluate(network.stations, S_OK, fresh_seed=True)
    analyzer.evaluate(network.stations, S_OK)              # seed 12, not 13
    assert [r["seed"] for r in transport.requests] == [
        11, 11 + FRESH_SEED_OFFSET, 12
    ]


def test_invalid_seed_policy_rejected():
    network = _network()
    client = QsimClient("http://qsim.test", transport=FakeTransport((200, {})))
    with pytest.raises(ValueError, match="seed_policy"):
        SimulationAnalyzer(network, client, seed_policy="random")


def test_stations_must_be_the_networks_stations(sim_response):
    network = _network()
    analyzer, _ = _analyzer(network, _healthy(sim_response))
    other = [GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1")]
    with pytest.raises(ValueError, match="network stations"):
        analyzer.evaluate(other, [3.0])


# --- the gamma-conservation check (spec 6.8) --------------------------------

def test_conservation_miss_warns_and_records(sim_response):
    network = _network()
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.9, "md1": 0.4, "fj": 0.5},      # 0.9 CI excludes 0.6
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response)
    with pytest.warns(RuntimeWarning, match="excludes derived gamma"):
        ev = analyzer.evaluate(network.stations, S_OK)
    assert any("mm1" in d and "excludes derived gamma" in d for d in ev.degraded)
    assert ev.sojourn_times == [0.42, 0.29, 0.45]            # the run still proceeds


def test_conservation_miss_raises_under_strict(sim_response):
    network = _network()
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.9, "md1": 0.4, "fj": 0.5},
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response, strict=True)
    with pytest.warns(RuntimeWarning, match="excludes derived gamma"):
        with pytest.raises(SimulationQualityError, match="excludes derived gamma"):
            analyzer.evaluate(network.stations, S_OK)


def test_forkjoin_throughput_never_flags_whatever_its_value(sim_response):
    network = _network()
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.6, "md1": 0.4, "fj": 99.0},     # nonsense at fj
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.degraded == []


def test_conservation_bracket_is_inclusive(sim_response):
    network = _network()
    # gamma sits exactly on the CI edge: mean 0.61, half-width 0.01 -> (0.60, 0.62).
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.61, "md1": 0.4, "fj": 0.5},
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.degraded == []


def test_missing_throughput_bounds_are_treated_as_a_miss(sim_response):
    network = _network()
    response = _healthy(sim_response)
    for m in response["measures"]:
        if m["station"] == "mm1" and m["type"] == "throughput":
            m["lower"] = None
            m["upper"] = None
    analyzer, _ = _analyzer(network, response)
    with pytest.warns(RuntimeWarning, match="mm1"):
        ev = analyzer.evaluate(network.stations, S_OK)
    assert any("mm1" in d for d in ev.degraded)


def test_strict_also_raises_on_a_degraded_measure(sim_response):
    network = _network()
    analyzer, _ = _analyzer(network, _healthy(sim_response, completed=False),
                            strict=True)
    with pytest.warns(RuntimeWarning, match="completed=false"):
        with pytest.raises(SimulationQualityError, match="completed=false"):
            analyzer.evaluate(network.stations, S_OK)
