import math

import pytest

from qopt.exceptions import TopologyError
from qopt.network import Network, Route
from qopt.optimizer import Optimizer
from qopt.allocator import min_feasible_budget
from qopt.station import ForkJoinStation, GG1Station

SRC = Network.SOURCE
SNK = Network.SINK


def _stations():
    return [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


def _routes():
    return [
        Route(SRC, "mm1", 0.6), Route(SRC, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", SNK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", SNK, 0.5),
        Route("fj", SNK, 1.0),
    ]


def _network():
    return Network(_stations(), _routes(), arrival_rate=1.0, name="qopt-mixed-network")


# --- Route -------------------------------------------------------------------

def test_route_defaults_to_probability_one():
    assert Route("a", "b").probability == 1.0


def test_route_is_frozen():
    import dataclasses

    r = Route("a", "b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.probability = 0.5


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_route_validates_probability(bad):
    with pytest.raises(ValueError):
        Route("a", "b", bad)


# --- gamma derivation --------------------------------------------------------

def test_network_derives_gammas_and_binds_them():
    net = _network()
    assert net.gammas == {"mm1": 0.6, "md1": 0.4, "fj": 0.5}
    assert [st.gamma for st in net.stations] == [0.6, 0.4, 0.5]
    assert net.traffic_iterations == 3


def test_network_is_iterable_and_sized():
    net = _network()
    assert len(net) == 3
    assert [st.name for st in net] == ["mm1", "md1", "fj"]


def test_explicit_gamma_in_a_network_is_rejected():
    stations = _stations()
    stations[0] = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1")
    with pytest.raises(ValueError, match="explicit gamma"):
        Network(stations, _routes(), arrival_rate=1.0)


# --- 4.2 structural validation, one test per row -----------------------------

def test_empty_station_name_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="")]
    with pytest.raises(TopologyError, match="non-empty"):
        Network(stations, [Route(SRC, ""), Route("", SNK)], arrival_rate=1.0)


def test_duplicate_station_names_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    with pytest.raises(TopologyError, match="unique"):
        Network(stations, [Route(SRC, "a"), Route("a", SNK)], arrival_rate=1.0)


def test_double_underscore_in_name_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a__join")]
    with pytest.raises(TopologyError, match="__"):
        Network(stations, [Route(SRC, "a__join"), Route("a__join", SNK)], arrival_rate=1.0)


@pytest.mark.parametrize("reserved", [Network.SOURCE, Network.SINK])
def test_reserved_endpoint_names_rejected(reserved):
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name=reserved)]
    with pytest.raises(TopologyError, match="reserved"):
        Network(stations, [Route(SRC, reserved), Route(reserved, SNK)], arrival_rate=1.0)


def test_dangling_route_endpoint_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    with pytest.raises(TopologyError, match="not a station name"):
        Network(stations, [Route(SRC, "a"), Route("a", "typo")], arrival_rate=1.0)


def test_source_with_an_in_edge_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(SRC, "a"), Route("a", SRC, 0.5), Route("a", SNK, 0.5)]
    with pytest.raises(TopologyError, match="no in-edges"):
        Network(stations, routes, arrival_rate=1.0)


def test_sink_with_an_out_edge_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(SRC, "a"), Route("a", SNK), Route(SNK, "a")]
    with pytest.raises(TopologyError, match="no out-edges"):
        Network(stations, routes, arrival_rate=1.0)


def test_out_edge_probabilities_must_sum_to_one():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="b")]
    routes = [Route(SRC, "a", 0.5), Route(SRC, "b", 0.4),
              Route("a", SNK), Route("b", SNK)]
    with pytest.raises(TopologyError, match="sum to"):
        Network(stations, routes, arrival_rate=1.0)


def test_station_with_no_out_edge_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    with pytest.raises(TopologyError, match="no out-edge"):
        Network(stations, [Route(SRC, "a")], arrival_rate=1.0)


def test_unreachable_station_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="orphan")]
    routes = [Route(SRC, "a"), Route("a", SNK), Route("orphan", SNK)]
    with pytest.raises(TopologyError, match="unreachable from"):
        Network(stations, routes, arrival_rate=1.0)


def test_flow_black_hole_rejected():
    # 'hole' can be reached but can never reach the sink.
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="hole")]
    routes = [Route(SRC, "a"), Route("a", "hole", 0.5), Route("a", SNK, 0.5),
              Route("hole", "hole")]
    with pytest.raises(TopologyError, match="unreachable from stations"):
        Network(stations, routes, arrival_rate=1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_arrival_rate_validated(bad):
    with pytest.raises(ValueError, match="arrival_rate"):
        Network(_stations(), _routes(), arrival_rate=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_arrival_scv_validated(bad):
    with pytest.raises(ValueError, match="arrival_scv"):
        Network(_stations(), _routes(), arrival_rate=1.0, arrival_scv=bad)


# --- the regression that matters --------------------------------------------

# Captured from the pre-change analytic run of examples/mixed_network.py at budget
# 6 * min_feasible_budget = 15.600000000000001. Compared bitwise on purpose: deriving
# gamma from the topology must not perturb a single float.
LEGACY_BUDGET = 15.600000000000001
LEGACY_S = [2.9601176145885644, 3.644844988735743, 3.017459891043565]
LEGACY_T = [0.4237076973701281, 0.2912706108409073, 0.45195507506074634]
LEGACY_ZETA = [1.0, 0.9451279819531168, 1.1377787740190126]
LEGACY_OBJECTIVE = 1.1669333832717816


def test_derived_gamma_reproduces_the_legacy_result_bitwise():
    net = _network()
    budget = 6 * min_feasible_budget(net.stations)
    assert budget == LEGACY_BUDGET
    assert min_feasible_budget(net.stations) == 2.6

    result = Optimizer(net.stations, budget=budget).run()
    assert result.converged
    assert result.iterations == 6
    assert result.capacities == LEGACY_S
    assert result.sojourn_times == LEGACY_T
    assert result.zeta == LEGACY_ZETA
    assert result.objective == LEGACY_OBJECTIVE


def test_example_build_network_returns_a_network_with_the_same_numbers():
    from examples.mixed_network import build_network, main

    net = build_network()
    assert isinstance(net, Network)
    assert [st.gamma for st in net] == [0.6, 0.4, 0.5]
    result = main()
    assert result.capacities == LEGACY_S
    assert result.objective == LEGACY_OBJECTIVE
