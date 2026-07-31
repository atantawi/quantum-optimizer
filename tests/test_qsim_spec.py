import pytest

from qopt.network import Network, Route
from qopt.qsim.client import DEFAULT_STOPPING
from qopt.qsim.spec import MEASURES, build_request
from qopt.station import GG1Station


def _network():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(Network.SOURCE, "a"), Route("a", Network.SINK)]
    return Network(stations, routes, arrival_rate=1.0, name="one")


def test_measure_list_is_the_closed_three():
    # Pins spec 5.4: qsim substitutes DEFAULTS (two of which are join-station numbers
    # at a fork-join node) whenever `measures` is null or empty.
    assert MEASURES == ("response-time", "system-response-time", "throughput")


def test_build_request_always_sends_the_exact_measure_list():
    request = build_request(_network(), [3.0], seed=7, stopping=DEFAULT_STOPPING)
    assert request["measures"] == [
        "response-time", "system-response-time", "throughput"
    ]
    assert request["measures"]  # never empty


def test_build_request_wraps_the_model_block():
    network = _network()
    request = build_request(network, [3.0], seed=7, stopping=DEFAULT_STOPPING)
    assert request["model"] == network.to_model_dict([3.0])
    assert request["seed"] == 7
    assert request["stopping"] == DEFAULT_STOPPING
    assert request["stopping"] is not DEFAULT_STOPPING     # copied, not aliased
    assert set(request) == {"model", "stopping", "measures", "seed"}


def test_build_request_omits_seed_when_none():
    request = build_request(_network(), [3.0], seed=None, stopping=DEFAULT_STOPPING)
    assert "seed" not in request


def test_build_request_rejects_an_empty_measure_list():
    with pytest.raises(ValueError, match="non-empty"):
        build_request(_network(), [3.0], seed=1, stopping=DEFAULT_STOPPING, measures=())
