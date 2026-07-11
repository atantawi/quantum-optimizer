from examples.mixed_network import build_network, main


def test_example_runs_and_converges():
    res = main()
    assert res.converged
    stations = build_network()
    assert len(res.capacities) == len(stations)
    for st, S in zip(stations, res.capacities):
        assert S * st.mu > st.gamma
    assert res.objective > 0
