"""Exceptions raised by the qopt optimizer."""


class QOptError(Exception):
    """Base class for all qopt errors."""


class InfeasibleBudgetError(QOptError):
    """Raised when the budget cannot satisfy the network's stability constraints."""


class InstabilityError(QOptError):
    """Raised when a capacity leaves a station unstable (S*mu <= gamma)."""
