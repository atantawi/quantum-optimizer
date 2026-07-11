def test_package_imports():
    import qopt
    from qopt.exceptions import InfeasibleBudgetError, InstabilityError

    assert issubclass(InfeasibleBudgetError, qopt.QOptError)
    assert issubclass(InstabilityError, qopt.QOptError)
