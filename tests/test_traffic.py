import pytest

from qopt.exceptions import TopologyError
from qopt.traffic import solve_traffic

SRC = "src"
SNK = "snk"


def test_tandem_lambda_is_equal_throughout():
    lam, iterations = solve_traffic(
        ["a", "b"],
        [(SRC, "a", 1.0), ("a", "b", 1.0), ("b", SNK, 1.0)],
        2.0, SRC, SNK,
    )
    assert lam == {"a": 2.0, "b": 2.0}
    assert iterations >= 1


def test_branch_splits_lambda_by_probability():
    lam, _ = solve_traffic(
        ["a", "b"],
        [(SRC, "a", 0.3), (SRC, "b", 0.7), ("a", SNK, 1.0), ("b", SNK, 1.0)],
        10.0, SRC, SNK,
    )
    assert lam["a"] == pytest.approx(3.0, rel=1e-12)
    assert lam["b"] == pytest.approx(7.0, rel=1e-12)


def test_feedback_loop_amplifies_lambda():
    # lambda_a = lambda_0 + p * lambda_a  =>  lambda_a = lambda_0 / (1 - p)
    lam, _ = solve_traffic(
        ["a"],
        [(SRC, "a", 1.0), ("a", "a", 0.25), ("a", SNK, 0.75)],
        1.0, SRC, SNK,
    )
    assert lam["a"] == pytest.approx(1.0 / (1.0 - 0.25), rel=1e-9)


def test_mixed_network_topology_derives_the_documented_gammas():
    # Spec 4.1.1: the topology behind examples/mixed_network.py's hand-supplied gammas.
    lam, iterations = solve_traffic(
        ["mm1", "md1", "fj"],
        [
            (SRC, "mm1", 0.6), (SRC, "md1", 0.4),
            ("mm1", "fj", 0.5), ("mm1", SNK, 0.5),
            ("md1", "fj", 0.5), ("md1", SNK, 0.5),
            ("fj", SNK, 1.0),
        ],
        1.0, SRC, SNK,
    )
    # Bitwise, not approx: Task 3's regression test depends on these being exact.
    assert lam == {"mm1": 0.6, "md1": 0.4, "fj": 0.5}
    assert iterations == 3


def test_closed_subnetwork_hits_the_cap():
    # a -> b -> a with p = 1 each way and external inflow: lambda diverges.
    with pytest.raises(TopologyError, match="closed subnetwork"):
        solve_traffic(
            ["a", "b"],
            [(SRC, "a", 1.0), ("a", "b", 1.0), ("b", "a", 1.0)],
            1.0, SRC, SNK, max_iter=50,
        )


def test_no_stations_is_trivially_solved():
    lam, iterations = solve_traffic([], [], 1.0, SRC, SNK)
    assert lam == {}
    assert iterations == 1
