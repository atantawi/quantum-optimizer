import pytest

from conftest import FakeTransport
from qopt.allocator import min_feasible_budget
from qopt.analyzer import AnalyticAnalyzer
from qopt.exceptions import InstabilityError, SimulationQualityError
from qopt.network import Network, Route
from qopt.optimizer import Optimizer
from qopt.qsim.analyzer import FRESH_SEED_OFFSET, SimulationAnalyzer
from qopt.qsim.client import QsimClient
from qopt.station import ForkJoinStation, GG1Station
from qopt.zeta import ZETA_SLOPE


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


def test_the_simulation_preflight_refuses_the_boundary_a_deterministic_station_admits(
    sim_response,
):
    """`admits_full_utilization` is a statement about the ANALYTIC model, and stops there.

    A `cov_a == cov_s == 0` G/G/1 has `E[T] = 1/(S*mu)`, finite at rho == 1, so
    `Station.check_stable` prices `S*mu == gamma` for it. The simulator is given no such
    station: `Network.to_model_dict` builds the arrival distribution from
    `Network.arrival_scv` and the routing, and only `cov_s` reaches the service node, so
    `cov_a` is never emitted at all. With the default `arrival_scv == 1.0` the model that
    would go over the wire at that capacity is a SATURATED M/D/1 -- exponential arrivals at
    0.6 against deterministic service with mean 1/0.6 -- which is exactly the shape spec 7.3
    added this guard to refuse before spending minutes on it.

    So `evaluate` passes `strict=True` and the boundary is refused here while the analytic
    path keeps it. The message says which of the two domains rejected the point, because the
    same station at the same capacity is legal one call away.

    Relaxing this would take showing the EMITTED arrival process deterministic at that
    station, which is a property of `arrival_scv` and the routing together and is not
    currently derived anywhere in qopt.
    """
    import math

    dd = GG1Station(mu=1.0, c=1.0, cov_a=0.0, cov_s=0.0, name="dd")
    network = Network(
        [dd],
        [Route(Network.SOURCE, "dd", 1.0), Route("dd", Network.SINK, 1.0)],
        arrival_rate=0.6,
        name="dd-net",
    )
    assert dd.gamma == 0.6
    boundary = dd.gamma / dd.mu

    # The analytic side admits it, and is unchanged by this test's subject.
    assert dd.admits_full_utilization is True
    dd.check_stable(boundary)
    assert dd.sojourn_time(boundary) == 1.0 / (boundary * dd.mu)

    # What the simulator would be sent there: arrivals from arrival_scv, not from cov_a.
    assert network.arrival_scv == 1.0
    nodes = network.to_model_dict([boundary])["nodes"]
    assert nodes[0]["arrivals"]["jobs"]["distribution"] == {
        "type": "exponential", "rate": 0.6
    }
    assert nodes[1]["service"]["jobs"]["distribution"] == {
        "type": "deterministic", "value": 1.0 / 0.6
    }

    response = sim_response(
        sojourn={"dd": 1.9}, throughput={"dd": 0.6}, system=1.9, model_name="dd-net"
    )
    analyzer, transport = _analyzer(network, response)
    with pytest.raises(InstabilityError) as exc:
        analyzer.evaluate(network.stations, [boundary])
    assert "ANALYTIC" in str(exc.value)          # names which domain refused it
    assert transport.requests == []              # no simulation time was spent

    # Strict means strict only AT the boundary: one ulp above it still runs.
    analyzer.evaluate(network.stations, [math.nextafter(boundary, 1.0)])
    assert len(transport.requests) == 1

    # And the relaxation is still available to a caller that asks for the analytic domain.
    assert dd.check_stable(boundary, strict=False) is None
    with pytest.raises(InstabilityError):
        dd.check_stable(boundary, strict=True)


def test_the_strict_preflight_is_declared_as_this_analyzers_domain(sim_response):
    """The preflight's strictness is a class attribute, not a literal inside `evaluate`.

    `Optimizer.run` has to know where this analyzer's domain ends -- its iterates come from
    eq 21 against the ANALYTIC domain, which is wider -- so both read one flag. Dropping the
    attribute would silently relax the preflight, which is why `evaluate` is checked here
    through the flag and not only for its own behaviour.
    """
    assert SimulationAnalyzer.requires_strict_stability is True
    assert AnalyticAnalyzer.requires_strict_stability is False

    dd = GG1Station(mu=1.0, c=1.0, cov_a=0.0, cov_s=0.0, name="dd")
    network = Network(
        [dd],
        [Route(Network.SOURCE, "dd", 1.0), Route("dd", Network.SINK, 1.0)],
        arrival_rate=0.6,
        name="dd-net",
    )
    response = sim_response(
        sojourn={"dd": 1.9}, throughput={"dd": 0.6}, system=1.9, model_name="dd-net"
    )
    analyzer, transport = _analyzer(network, response)
    with pytest.raises(InstabilityError):
        analyzer.evaluate(network.stations, [dd.gamma / dd.mu])
    assert transport.requests == []

    # The analytic analyzer prices that same capacity, which is what the flag distinguishes.
    assert AnalyticAnalyzer().evaluate(network.stations, [dd.gamma / dd.mu]).sojourn_times \
        == [1.0 / dd.gamma]


# The canned E[T] is not dd's model at cov_a = 1, so the realistic measurement below draws a
# shape flag, and the loop over a constant response stops just short of tol. Neither is what
# this test is about; both are filtered by message so any other warning still surfaces.
@pytest.mark.filterwarnings("ignore:station 'dd'.*disagrees with its analytic model")
@pytest.mark.filterwarnings("ignore:Optimizer did not converge")
def test_a_default_simulated_run_reaches_the_post_on_a_boundary_optimum(sim_response):
    """The reported failure, end to end: `Optimizer(..., analyzer=SimulationAnalyzer(...))`.

    With `warm_start=True` -- the default -- the analytic pre-solve ran first and handed its
    capacities straight to `evaluate`. On this network that answer is S_dd = gamma/mu
    exactly, so a normal run raised InstabilityError after ZERO POSTs. The warm start is now
    declined when it falls outside the analyzer's domain and the loop starts from eq 21 on
    the initial zeta instead, so the run reaches the simulator.

    The first request is asserted to carry the COLD capacity, not the warm one: the emitted
    service rate is `S*mu`, and at the boundary that would be exactly gamma.
    """
    dd = GG1Station(gamma=None, mu=1.0, weight=1.0, c=1.0, cov_a=0.0, cov_s=0.0,
                    zeta_mode=ZETA_SLOPE, name="dd")
    mm = GG1Station.mm1(gamma=None, mu=3.0, weight=3e5, c=1.0, name="mm")
    network = Network(
        [dd, mm],
        [Route(Network.SOURCE, "dd", 1 / 3), Route(Network.SOURCE, "mm", 2 / 3),
         Route("dd", Network.SINK, 1.0), Route("mm", Network.SINK, 1.0)],
        arrival_rate=1.8,
        name="boundary-net",
    )
    assert [st.gamma for st in network.stations] == [0.6, 1.2]
    C = 1.01 * min_feasible_budget(network.stations)

    # The warm start is the boundary, to the bit -- the premise of the whole test.
    assert Optimizer(network.stations, C).run().capacities[0] == 0.6

    response = sim_response(
        sojourn={"dd": 1.7, "mm": 0.9}, throughput={"dd": 0.6, "mm": 1.2},
        system=2.6, model_name="boundary-net",
    )
    # dd is zeta_mode=ZETA_SLOPE, so it now requests interarrival-time too. The network
    # feeds dd a Bernoulli split of a Poisson source, so its arrivals are Poisson: mean
    # 1/gamma (gamma=0.6) with variance mean**2, i.e. a measured cov_a of 1. That differs
    # from dd's constructor cov_a=0.0, which moves phi and so the later iterates, but not
    # what this test asserts: the first POST is sent before any measurement exists.
    response = _with_interarrival(response, dd=(1 / 0.6, (1 / 0.6) ** 2))
    analyzer, transport = _analyzer(network, response)
    with pytest.warns(RuntimeWarning, match="warm start"):
        result = Optimizer(network.stations, C, analyzer=analyzer).run()

    assert transport.requests                      # it got to the simulator at all
    rate = transport.requests[0]["model"]["nodes"][1]["service"]["jobs"]["distribution"]
    assert rate["value"] == pytest.approx(1.0 / 0.6000315230918326, rel=1e-15)
    assert rate["value"] != 1.0 / 0.6              # not the warm start's saturated rate
    assert result.warm_start_iterations == 0
    assert result.sim_calls == len(transport.requests) > 0


# --- measured arrival cov_a (spec 2026-09-25 §3.3) ------------------------------

from qopt.exceptions import SimulationRequestError
from qopt.qsim.spec import COV_A_MEASURE, MEASURES
from qopt.zeta import ZETA_LEVEL


def _slope_network(md1_mode=ZETA_SLOPE, fj_mode=ZETA_LEVEL):
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1", zeta_mode=md1_mode),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj", zeta_mode=fj_mode),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def _ia_entry(station, mean, variance):
    return {"station": station, "class": "jobs", "type": COV_A_MEASURE,
             "mean": mean, "lower": None, "upper": None, "success": True,
             "variance": variance, "stdDev": variance ** 0.5}


def _with_interarrival(response, **by_station):
    response = dict(response)
    response["measures"] = response["measures"] + [
        _ia_entry(name, mean, var) for name, (mean, var) in by_station.items()
    ]
    return response


def test_a_network_with_no_slope_gg1_sends_todays_request(sim_response):
    network = _network()                                  # every station level
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    ev = analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES)
    assert "secondMoments" not in transport.requests[0]
    assert ev.arrival_cov is None


def test_a_slope_fork_join_alone_does_not_trigger_the_measure(sim_response):
    network = _slope_network(md1_mode=ZETA_LEVEL, fj_mode=ZETA_SLOPE)
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES)


def test_a_slope_gg1_adds_interarrival_time_and_nothing_else(sim_response):
    network = _slope_network()
    response = _with_interarrival(_healthy(sim_response), md1=(2.5, 1.5625))
    analyzer, transport = _analyzer(network, response)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES) + [COV_A_MEASURE]
    assert "secondMoments" not in transport.requests[0]
    assert ev.arrival_cov == [None, 0.5, None]            # sqrt(1.5625 / 2.5**2)
    assert ev.degraded == []


def test_measure_cov_a_false_restores_todays_request(sim_response):
    network = _slope_network()
    analyzer, transport = _analyzer(network, _healthy(sim_response), measure_cov_a=False)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES)
    assert ev.arrival_cov is None


def test_a_missing_measurement_degrades_and_strict_raises(sim_response):
    network = _slope_network()
    analyzer, _ = _analyzer(network, _healthy(sim_response))      # no interarrival entry
    with pytest.warns(RuntimeWarning, match="md1"):
        ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.arrival_cov == [None, None, None]
    assert any("md1" in d and "constructor cov_a" in d for d in ev.degraded)

    strict, _ = _analyzer(network, _healthy(sim_response), strict=True)
    with pytest.warns(RuntimeWarning), pytest.raises(SimulationQualityError, match="md1"):
        strict.evaluate(network.stations, S_OK)


_OLD_SERVICE = (400, {"error": "invalid request", "details": [
    "unsupported measure type: 'interarrival-time'; supported: [response-time, throughput]"
]})


def test_an_old_service_names_the_opt_out():
    network = _slope_network()
    client = QsimClient("http://qsim.test", transport=FakeTransport(_OLD_SERVICE))
    with pytest.raises(SimulationRequestError, match=r"measure_cov_a=False"):
        SimulationAnalyzer(network, client).evaluate(network.stations, S_OK)


def test_an_unrelated_request_error_gets_no_version_hint():
    network = _slope_network()
    other = (400, {"error": "invalid request", "details": ["node name 'x y' is unsafe"]})
    client = QsimClient("http://qsim.test", transport=FakeTransport(other))
    with pytest.raises(SimulationRequestError) as info:
        SimulationAnalyzer(network, client).evaluate(network.stations, S_OK)
    assert "measure_cov_a" not in str(info.value)
