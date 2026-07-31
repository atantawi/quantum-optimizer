import json

import pytest

from conftest import FakeTransport
from qopt.exceptions import (
    SimulationEngineError,
    SimulationRequestError,
    SimulationTransportError,
)
from qopt.qsim.client import DEFAULT_STOPPING, TIMEOUT_MARGIN_SECONDS, QsimClient

OK_BODY = {"modelName": "m", "completed": True, "measures": []}


def _client(transport, **kwargs):
    return QsimClient("http://qsim.test/", transport=transport, **kwargs)


def test_default_stopping_and_timeout():
    client = _client(FakeTransport((200, OK_BODY)))
    assert client.stopping == DEFAULT_STOPPING
    assert client.stopping is not DEFAULT_STOPPING          # defensively copied
    wall = DEFAULT_STOPPING["maxWallClockSeconds"]
    assert client.timeout == wall + 2 * TIMEOUT_MARGIN_SECONDS


def test_base_url_trailing_slash_stripped():
    client = _client(FakeTransport((200, OK_BODY)))
    client.post_simulate({"model": {}})
    assert client.transport.calls[0][0] == "http://qsim.test/simulate"


def test_timeout_must_clear_the_wall_clock_plus_margin():
    with pytest.raises(ValueError, match="must exceed maxWallClockSeconds"):
        _client(FakeTransport((200, OK_BODY)),
                stopping={"maxWallClockSeconds": 120}, timeout=120.0)


def test_timeout_just_above_the_margin_is_accepted():
    client = _client(FakeTransport((200, OK_BODY)),
                     stopping={"maxWallClockSeconds": 120}, timeout=131.0)
    assert client.timeout == 131.0


def test_stopping_without_a_wall_clock_is_rejected():
    with pytest.raises(ValueError, match="maxWallClockSeconds"):
        _client(FakeTransport((200, OK_BODY)), stopping={"alpha": 0.05})


def test_post_simulate_returns_the_parsed_body_and_sends_json():
    transport = FakeTransport((200, OK_BODY))
    client = _client(transport)
    assert client.post_simulate({"model": {"name": "m"}}) == OK_BODY
    url, request, timeout = transport.calls[0]
    assert url == "http://qsim.test/simulate"
    assert request == {"model": {"name": "m"}}
    assert timeout == client.timeout


@pytest.mark.parametrize("status", [400, 405, 413, 422])
def test_client_errors_map_to_simulation_request_error(status):
    body = {"error": "unprocessable model", "details": ["probabilities do not sum to 1"]}
    client = _client(FakeTransport((status, body)))
    with pytest.raises(SimulationRequestError, match="probabilities do not sum to 1"):
        client.post_simulate({"model": {}})


def test_unsupported_measure_type_maps_to_request_error():
    # Requesting the literal 'fork-join-response-time' is a 400 (spec 5.3); qopt must
    # never emit it, and if it ever does the failure must be this exception.
    body = {"error": "invalid request",
            "details": ["unsupported measure type: fork-join-response-time"]}
    client = _client(FakeTransport((400, body)))
    with pytest.raises(SimulationRequestError, match="fork-join-response-time"):
        client.post_simulate({"measures": ["fork-join-response-time"]})


def test_server_error_maps_to_engine_error():
    body = {"error": "simulation engine error", "details": ["correlationId=abc"]}
    client = _client(FakeTransport((500, body)))
    with pytest.raises(SimulationEngineError, match="correlationId=abc"):
        client.post_simulate({"model": {}})


def test_unexpected_status_maps_to_transport_error():
    client = _client(FakeTransport((302, b"")))
    with pytest.raises(SimulationTransportError, match="unexpected HTTP 302"):
        client.post_simulate({"model": {}})


def test_unreadable_success_body_maps_to_engine_error():
    client = _client(FakeTransport((200, b"<html>not json</html>")))
    with pytest.raises(SimulationEngineError, match="unreadable response body"):
        client.post_simulate({"model": {}})


def test_non_json_error_body_still_produces_a_message():
    client = _client(FakeTransport((422, b"plain text failure")))
    with pytest.raises(SimulationRequestError, match="plain text failure"):
        client.post_simulate({"model": {}})


def test_health_is_a_get():
    transport = FakeTransport((200, OK_BODY))
    client = _client(transport)
    assert client.health() == {"status": "ok"}
    url, request, _ = transport.calls[0]
    assert url == "http://qsim.test/health"
    assert request is None


def test_health_failure_is_a_transport_error():
    transport = FakeTransport((200, OK_BODY), health=(503, {"error": "down"}))
    client = _client(transport)
    with pytest.raises(SimulationTransportError, match="503"):
        client.health()


def test_preflight_calls_health_at_construction():
    transport = FakeTransport((200, OK_BODY))
    _client(transport, preflight=True)
    assert transport.calls[0][0] == "http://qsim.test/health"


def test_preflight_failure_surfaces_immediately():
    transport = FakeTransport((200, OK_BODY), health=(503, {"error": "down"}))
    with pytest.raises(SimulationTransportError):
        _client(transport, preflight=True)


def test_transport_exceptions_are_not_swallowed():
    def broken(url, body, timeout):
        raise SimulationTransportError("connection refused")

    client = _client(broken)
    with pytest.raises(SimulationTransportError, match="connection refused"):
        client.post_simulate({"model": {}})
