def test_package_imports():
    import qopt
    from qopt.exceptions import InfeasibleBudgetError, InstabilityError

    assert issubclass(InfeasibleBudgetError, qopt.QOptError)
    assert issubclass(InstabilityError, qopt.QOptError)


def test_root_exports_the_whole_exception_hierarchy():
    # Spec 7.1 presents this hierarchy as the public error contract: a caller acting on
    # it (retry on SimulationTransportError, don't on SimulationRequestError, escalate on
    # SimulationEngineError) must be able to reach every name from the package root,
    # not just `except qopt.QOptError` (finding 2).
    import qopt

    for name in (
        "QOptError",
        "InfeasibleBudgetError",
        "InstabilityError",
        "TopologyError",
        "SimulationError",
        "SimulationTransportError",
        "SimulationRequestError",
        "SimulationEngineError",
        "SimulationQualityError",
        "MeasureMissingError",
    ):
        assert hasattr(qopt, name), f"qopt.{name} is not exported from the root"
        assert name in qopt.__all__

    assert issubclass(qopt.InfeasibleBudgetError, qopt.QOptError)
    assert issubclass(qopt.InstabilityError, qopt.QOptError)
    assert issubclass(qopt.TopologyError, qopt.QOptError)
    assert issubclass(qopt.SimulationError, qopt.QOptError)
    for subclass in (
        qopt.SimulationTransportError,
        qopt.SimulationRequestError,
        qopt.SimulationEngineError,
        qopt.SimulationQualityError,
        qopt.MeasureMissingError,
    ):
        assert issubclass(subclass, qopt.SimulationError)
        assert issubclass(subclass, qopt.QOptError)
