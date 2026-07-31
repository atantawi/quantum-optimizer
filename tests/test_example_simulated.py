def test_simulated_tandem_runs_analytically_without_a_service(monkeypatch):
    """The example must be runnable offline: it prints the analytic table and stops."""
    monkeypatch.delenv("QOPT_QSIM_URL", raising=False)
    from examples.simulated_tandem import build_network, main

    network = build_network()
    assert [st.name for st in network] == ["shape", "serve"]
    # A tandem chain carries lambda_0 through unchanged.
    assert [st.gamma for st in network] == [1.0, 1.0]

    result = main()
    assert result is not None
    assert result.sim_calls == 0            # no service, so the analytic path ran
    for st, S in zip(network, result.capacities):
        assert S * st.mu > st.gamma


def test_simulated_mixed_network_runs_analytically_without_a_service(monkeypatch):
    monkeypatch.delenv("QOPT_QSIM_URL", raising=False)
    from examples.simulated_mixed_network import build_network, main

    network = build_network()
    assert [st.name for st in network] == ["mm1", "md1", "fj"]
    assert [st.gamma for st in network] == [0.6, 0.4, 0.5]

    result = main()
    assert result is not None
    assert result.sim_calls == 0
    # Same analytic answer as examples/mixed_network.py, bitwise.
    assert result.capacities == [
        2.9601176145885644, 3.644844988735743, 3.017459891043565
    ]
    assert result.objective == 1.1669333832717816
