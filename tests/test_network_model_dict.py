import json
import pathlib

import pytest

from qopt.network import Network, Route
from qopt.station import ForkJoinStation, GG1Station, Station, distribution_dict

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "qopt_mixed_network_request.json"


def _mixed_network():
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


# --- distribution emission (spec 5.2) ---------------------------------------

def test_distribution_exponential_when_scv_is_one():
    assert distribution_dict(3.0, 1.0) == {"type": "exponential", "rate": 3.0}


def test_distribution_deterministic_when_scv_is_zero():
    assert distribution_dict(4.0, 0.0) == {"type": "deterministic", "value": 0.25}


def test_distribution_moment_form_otherwise():
    assert distribution_dict(2.0, 1.5) == {"mean": 0.5, "scv": 1.5}


# --- station node fragments -------------------------------------------------

def test_measure_type_is_one_constant_for_every_station_type():
    assert Station.SIM_MEASURE_TYPE == "response-time"
    assert GG1Station.mm1(mu=1.0, c=1.0, name="a").SIM_MEASURE_TYPE == "response-time"
    fj = ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj")
    assert fj.SIM_MEASURE_TYPE == "response-time"


def test_forkjoin_is_exempt_from_the_conservation_check():
    # qsim-service#8: a fork-join node's throughput is the internal join station's number.
    assert GG1Station.mm1(mu=1.0, c=1.0, name="a").sim_conservation_checked is True
    assert ForkJoinStation(
        mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"
    ).sim_conservation_checked is False


def test_gg1_sim_node_emits_a_queue():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1")
    assert st.sim_node(3.0, "jobs") == {
        "name": "mm1", "type": "queue", "servers": 1, "scheduling": "fcfs",
        "capacity": None,
        "service": {"jobs": {"distribution": {"type": "exponential", "rate": 3.0}}},
    }


def test_md1_sim_node_emits_a_deterministic_service():
    st = GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="md1")
    node = st.sim_node(4.0, "jobs")
    assert node["service"]["jobs"]["distribution"] == {
        "type": "deterministic", "value": 0.25
    }


def test_forkjoin_sim_node_emits_both_branches_and_join_all():
    st = ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj")
    assert st.sim_node(5.0, "jobs") == {
        "name": "fj", "type": "fork-join",
        "branches": [
            {"service": {"jobs": {"distribution": {"type": "exponential", "rate": 5.0}}}},
            {"service": {"jobs": {"distribution": {"type": "exponential", "rate": 10.0}}}},
        ],
        "join": "all",
    }


def test_station_sim_node_is_abstract():
    from qopt.station import SingleServerStation

    assert getattr(Station.sim_node, "__isabstractmethod__", False)
    with pytest.raises(TypeError):
        SingleServerStation(gamma=0.5, mu=1.0, c=1.0)  # type: ignore[abstract]


# --- to_model_dict ----------------------------------------------------------

def test_to_model_dict_matches_the_golden_fixture_byte_for_byte():
    model = _mixed_network().to_model_dict([3.0, 4.0, 5.0])
    assert json.dumps(model, indent=2) + "\n" == FIXTURE.read_text()


def test_to_model_dict_node_order_is_source_stations_sink():
    model = _mixed_network().to_model_dict([3.0, 4.0, 5.0])
    assert [n["name"] for n in model["nodes"]] == ["src", "mm1", "md1", "fj", "snk"]


def test_to_model_dict_rejects_a_mismatched_capacity_vector():
    with pytest.raises(ValueError, match="length"):
        _mixed_network().to_model_dict([3.0, 4.0])


def test_to_model_dict_arrival_distribution_comes_from_the_network():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(Network.SOURCE, "a"), Route("a", Network.SINK)]
    net = Network(stations, routes, arrival_rate=2.0, arrival_scv=0.0)
    source = net.to_model_dict([3.0])["nodes"][0]
    # arrival_scv = 0 => deterministic inter-arrival times of 1/2.0, never a station's cov_a.
    assert source["arrivals"]["jobs"]["distribution"] == {
        "type": "deterministic", "value": 0.5
    }


def test_to_model_dict_uses_the_configured_job_class():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(Network.SOURCE, "a"), Route("a", Network.SINK)]
    net = Network(stations, routes, arrival_rate=1.0, job_class="web")
    model = net.to_model_dict([3.0])
    assert model["classes"] == [{"name": "web", "type": "open"}]
    assert "web" in model["routing"]
    assert "web" in model["nodes"][1]["service"]


def test_network_has_no_from_model_dict():
    # S is not recoverable from the emitted S*mu product, so a round trip is undefined.
    assert not hasattr(Network, "from_model_dict")


# --- to_dot -----------------------------------------------------------------

def test_to_dot_emits_every_node_and_edge():
    dot = _mixed_network().to_dot()
    assert dot.startswith('digraph "qopt-mixed-network" {')
    assert dot.rstrip().endswith("}")
    for name in ("src", "snk", "mm1", "md1", "fj"):
        assert f'"{name}"' in dot
    assert '"src" -> "mm1"' in dot
    assert '"fj" -> "snk"' in dot
    assert dot.count("->") == 7
    assert "box3d" in dot          # the fork-join station's shape
