"""Exceptions raised by the qopt optimizer."""


class QOptError(Exception):
    """Base class for all qopt errors."""


class InfeasibleBudgetError(QOptError):
    """Raised when the budget cannot satisfy the network's stability constraints."""


class InstabilityError(QOptError):
    """Raised when a capacity leaves a station unstable (S*mu <= gamma)."""


class TopologyError(QOptError):
    """Raised when a Network's structure is not a well-formed open chain (spec 4.2)."""


class SimulationError(QOptError):
    """Base class for failures of the simulation-backed evaluation path."""


class SimulationTransportError(SimulationError):
    """The simulation service was unreachable: refused, timed out, or DNS failure."""


class SimulationRequestError(SimulationError):
    """The service rejected our request (HTTP 400/422): a spec.py bug or an invalid network."""


class SimulationEngineError(SimulationError):
    """The service failed while simulating (HTTP 500), or returned an unreadable body."""


class SimulationQualityError(SimulationError):
    """A simulation result was degraded and strict mode was requested (spec 7.2)."""


class MeasureMissingError(SimulationError):
    """The response lacked a station response-time that eq 22 requires (spec 7.1)."""
