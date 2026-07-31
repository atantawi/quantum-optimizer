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


def test_mixed_network_table_prints_a_system_measure_with_no_ci(capsys):
    """A system E[T] whose CI is absent must print, not raise.

    measures.extract reports a missing system-level CI as (mean, (None, None)) — the same
    shape throughput uses — so every consumer that formats those bounds has to handle the
    Nones, exactly as the per-station column already does.
    """
    from examples.simulated_mixed_network import _print_table, build_network
    from qopt.optimizer import Result

    network = build_network()
    result = Result(
        capacities=[3.0, 3.6, 3.0], sojourn_times=[0.5, 0.4, 0.3], zeta=[1.0, 1.0, 1.0],
        objective=1.2, iterations=3, residual=1e-7, converged=True, stop_reason="tol",
        system_response_time=(1.15, (None, None)),
    )
    _print_table("check", network, result)
    out = capsys.readouterr().out
    assert "system response time = 1.150000" in out
    assert "CI (None" not in out


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
