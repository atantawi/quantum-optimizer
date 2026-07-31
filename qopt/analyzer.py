"""The evaluation seam: network-level E[T], analytic or simulated (spec 2.1, 6.1)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Evaluation:
    """One network-level evaluation of E[T] at a given capacity vector.

    Fields:
        sojourn_times: E[T] per station, aligned to the station order.
        ci: (lower, upper) per station, or None on a deterministic path.
        degraded: audit strings — weak measures, gamma-conservation misses (spec 6.8).
        extras: diagnostics — system_response_time, throughput, seed, wallClockSeconds.
    """

    sojourn_times: list
    ci: list | None = None
    degraded: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


class Analyzer(ABC):
    """Supplies E[T] for every station from one capacity vector."""

    is_stochastic = False
    """Drives the optimizer's warm-start and damping defaults (spec 6.2)."""

    @abstractmethod
    def evaluate(self, stations, S, *, fresh_seed=False):
        """Return an Evaluation for `stations` at capacities `S`.

        `fresh_seed` asks a stochastic analyzer for an independently seeded run; a
        deterministic one ignores it. One call shape serves both (spec 6.5).
        """


class AnalyticAnalyzer(Analyzer):
    """Delegates to each station's own closed-form sojourn_time. No confidence intervals."""

    is_stochastic = False

    def evaluate(self, stations, S, *, fresh_seed=False):
        return Evaluation(
            sojourn_times=[st.sojourn_time(Si) for st, Si in zip(stations, S)],
            ci=None,
        )
