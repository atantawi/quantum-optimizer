"""Integration tests against a live qsim-service.

Gated on QOPT_QSIM_URL and skipped by default, because they need the GPL service
running (typically its Docker image) and each takes seconds to minutes.

    QOPT_QSIM_URL=http://localhost:8080 .venv/bin/python -m pytest \
        tests/test_integration_qsim.py -v
"""

import os

import pytest

from qopt.network import Network, Route
from qopt.qsim.analyzer import SimulationAnalyzer
from qopt.qsim.client import QsimClient
from qopt.station import ForkJoinStation, GG1Station

QSIM_URL = os.environ.get("QOPT_QSIM_URL")

pytestmark = pytest.mark.skipif(
    not QSIM_URL, reason="set QOPT_QSIM_URL to run live qsim-service tests"
)

# Tight enough to be discriminating, loose enough to finish in seconds.
STOPPING = {
    "alpha": 0.05,
    "precision": 0.02,
    "minSamples": 50000,
    "maxSamples": 4000000,
    "maxWallClockSeconds": 180,
}


@pytest.fixture
def client():
    return QsimClient(QSIM_URL, stopping=STOPPING, preflight=True)


def test_health_responds(client):
    assert client.health()["status"] == "ok"


def test_mm1_simulated_ci_brackets_the_analytic_sojourn_time(client):
    """Spec 11 criterion 5: the actual validation of the idea, at one station."""
    station = GG1Station.mm1(mu=1.0, c=1.0, name="mm1")
    network = Network(
        [station],
        [Route(Network.SOURCE, "mm1"), Route("mm1", Network.SINK)],
        arrival_rate=1.0,
        name="mm1-bracket",
    )
    assert station.gamma == 1.0

    S = [2.0]                      # S*mu = 2.0, rho = 0.5
    analytic = station.sojourn_time(S[0])
    assert analytic == pytest.approx(1.0, rel=1e-12)      # 1/(S*mu - gamma)

    evaluation = SimulationAnalyzer(network, client).evaluate(network.stations, S)
    lower, upper = evaluation.ci[0]
    assert lower <= analytic <= upper, (
        f"simulated CI ({lower}, {upper}) does not bracket analytic {analytic}"
    )


def test_system_measure_key_inference_holds(client):
    """Settles spec 5.3 gotcha 2: is a system measure keyed on station ""?

    If this fails, change SYSTEM_STATION in qopt/qsim/measures.py to whatever the
    response actually carries and re-run.
    """
    station = GG1Station.mm1(mu=1.0, c=1.0, name="mm1")
    network = Network(
        [station],
        [Route(Network.SOURCE, "mm1"), Route("mm1", Network.SINK)],
        arrival_rate=1.0,
        name="system-measure-probe",
    )
    evaluation = SimulationAnalyzer(network, client).evaluate(network.stations, [2.0])
    assert evaluation.extras["system_response_time"] is not None, (
        "system-response-time did not come back under station '' — see "
        "qopt/qsim/measures.py::SYSTEM_STATION"
    )


def _mixed_network():
    """The spec 4.1.1 branching topology, whose derived gammas are (0.6, 0.4, 0.5)."""
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


def test_gamma_conservation_holds_on_the_branching_network(client):
    """Spec 11 criterion 5b: solve_traffic and to_model_dict describe the same network.

    The branching topology is what makes this meaningful — a tandem chain would pass
    even with the source split wrong.
    """
    network = _mixed_network()
    assert network.gammas == {"mm1": 0.6, "md1": 0.4, "fj": 0.5}

    evaluation = SimulationAnalyzer(network, client).evaluate(
        network.stations, [3.0, 4.0, 5.0]
    )
    throughput = evaluation.extras["throughput"]
    for name, expected in (("mm1", 0.6), ("md1", 0.4)):
        mean, (lower, upper) = throughput[name]
        assert lower <= expected <= upper, (
            f"{name}: simulated throughput {mean} CI ({lower}, {upper}) "
            f"excludes derived gamma {expected}"
        )
    # No conservation degradation was recorded for the checked stations.
    assert not [d for d in evaluation.degraded if "excludes derived gamma" in d]


def test_optimizer_runs_against_the_live_service(client):
    """The whole loop, end to end: warm start, damped iterations, final fresh-seed run."""
    from qopt.allocator import min_feasible_budget
    from qopt.optimizer import Optimizer

    network = _mixed_network()
    budget = 6 * min_feasible_budget(network.stations)
    analyzer = SimulationAnalyzer(network, client)
    result = Optimizer(network, budget=budget, analyzer=analyzer, max_iter=6).run()

    assert result.sim_calls == result.iterations + 1
    assert result.warm_start_iterations > 0
    assert len(result.sojourn_ci) == 3
    assert result.stop_reason in ("tol", "noise-floor", "max_iter")
    for st, S in zip(network.stations, result.capacities):
        assert S * st.mu > st.gamma
    spent = sum(
        st.alloc_cost * S for st, S in zip(network.stations, result.capacities)
    )
    assert spent == pytest.approx(budget, rel=1e-9)


# --- fork-join oracles (spec 8.2, 11 criterion 5a) ---------------------------

FJ_STOPPING = dict(STOPPING, precision=0.01, minSamples=200000, maxWallClockSeconds=600)


@pytest.fixture
def fj_client():
    """A tighter, longer-running client: the fork-join oracles need sharper CIs."""
    return QsimClient(QSIM_URL, stopping=FJ_STOPPING, preflight=True)


def _fork_join_only_network(*, r, mu=1.0, arrival_rate=1.0, name="fj-only"):
    """src -> fj -> snk, where the fork-join station is the only service in the network."""
    station = ForkJoinStation(mu=mu, r=r, c1=1.0, c2=1.0, name="fj")
    return Network(
        [station],
        [Route(Network.SOURCE, "fj"), Route("fj", Network.SINK)],
        arrival_rate=arrival_rate,
        name=name,
    )


def _branch_lower_bound(station, S):
    """The slower branch's own M/M/1 mean — a rigorous lower bound on the FJ sojourn."""
    rates = (S * station.mu, S * station.r * station.mu)
    assert min(rates) > station.gamma, "branch saturated"
    return max(1.0 / (rate - station.gamma) for rate in rates)


def test_forkjoin_response_time_equals_system_response_time(fj_client):
    """Criterion 5a(i): an identity, because both come from the same sample path.

    Also the sharpest guard against a regression to join-anchoring, which measured
    0.0987 on a network where the identity gives 0.2885.
    """
    network = _fork_join_only_network(r=2.0, name="fj-identity")
    station = network.stations[0]
    assert station.gamma == 1.0

    S = [5.0]                       # branches at 5.0 and 10.0
    evaluation = SimulationAnalyzer(network, fj_client).evaluate(network.stations, S)
    fj_response_time = evaluation.sojourn_times[0]
    system, _ = evaluation.extras["system_response_time"]

    assert system == pytest.approx(fj_response_time, abs=1e-9), (
        f"fork-join response-time {fj_response_time} != system-response-time {system}; "
        f"the measure is probably anchored on the join station again"
    )
    assert fj_response_time >= _branch_lower_bound(station, S[0])


def test_symmetric_forkjoin_ci_brackets_t_ul(fj_client):
    """Criterion 5a(ii): r = 1 is where t_ul is exact, so bracketing is the right shape."""
    from qopt.forkjoin_approx import t_ul

    network = _fork_join_only_network(r=1.0, name="fj-symmetric")
    station = network.stations[0]

    S = [4.0]                       # both branches at 4.0, rho = 0.25
    expected = t_ul(station.gamma, S[0] * station.mu, S[0] * station.r * station.mu)
    # t_ul is exact for equal rates: (12 - rho) / (8 * (mu - lambda)).
    rho = station.gamma / (S[0] * station.mu)
    assert expected == pytest.approx(
        (12.0 - rho) / (8.0 * (S[0] * station.mu - station.gamma)), rel=1e-12
    )

    evaluation = SimulationAnalyzer(network, fj_client).evaluate(network.stations, S)
    simulated = evaluation.sojourn_times[0]
    lower, upper = evaluation.ci[0]
    assert lower <= expected <= upper, (
        f"simulated CI ({lower}, {upper}) does not bracket the exact t_ul {expected}"
    )
    assert simulated >= _branch_lower_bound(station, S[0])


def test_unsupported_measure_literal_is_rejected_by_the_live_service(client):
    """Pins spec 5.3: 'fork-join-response-time' is not a type, and qopt never emits it."""
    from qopt.exceptions import SimulationRequestError
    from qopt.qsim.spec import build_request

    network = _fork_join_only_network(r=2.0, name="fj-bad-measure")
    request = build_request(
        network, [5.0], seed=1, stopping=STOPPING,
        measures=("fork-join-response-time",),
    )
    with pytest.raises(SimulationRequestError):
        client.post_simulate(request)
