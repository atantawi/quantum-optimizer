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
    """We never got a usable answer out of the service.

    Either it was unreachable — refused, timed out, DNS failure — or it answered in a way
    that carries no simulation outcome to interpret: a /health that was not 200, or a
    /simulate status outside the request and 5xx families qsim-service documents.
    """


class SimulationRequestError(SimulationError):
    """The service rejected our request (HTTP 400/405/413/422): a spec.py bug or an invalid network."""


class SimulationEngineError(SimulationError):
    """The service failed while simulating (HTTP 500), or returned an unreadable body."""


class SimulationQualityError(SimulationError):
    """A simulation result was degraded and strict mode was requested (spec 7.2)."""


class MeasureMissingError(SimulationError):
    """The response lacked a station response-time that eq 22 requires (spec 7.1)."""
