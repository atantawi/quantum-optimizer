"""Station hierarchy: each station owns its queueing math."""

import math
from abc import ABC, abstractmethod

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul


def distribution_dict(rate, scv):
    """qsim distribution fragment for a given rate (1/mean) and squared CV (spec 5.2).

    The same three-branch rule serves both service and inter-arrival distributions.
    Takes a rate rather than a mean so the exponential form stays bit-exact: the
    emitted `rate` is the caller's S*mu, not a value round-tripped through 1/mean.
    """
    if scv == 1.0:
        return {"type": "exponential", "rate": rate}
    if scv == 0.0:
        return {"type": "deterministic", "value": 1.0 / rate}
    return {"mean": 1.0 / rate, "scv": scv}


class Station(ABC):
    """A node in the queueing network.

    Fields (used directly by the allocator and eqs 21/22):
        gamma: arrival rate. Optional at construction: a Network derives it from the
            traffic equations and binds it via bind_gamma().
        mu: base service rate (for a fork-join station, the slower server's rate).
        weight: sojourn-time weight (omega).
        name: optional label for reporting.
    """

    # --- qsim facts a station carries at class level (spec 5.2) ---
    SIM_MEASURE_TYPE = "response-time"
    """Which qsim measure supplies E[T] for eq 22.

    Deliberately a constant rather than an abstract property: post qsim-service#7 a
    fork-join node's `response-time` *is* the fork-to-join sojourn, so no station type
    varies it. A hook every subclass implements identically is dead abstraction.
    """

    sim_conservation_checked = True
    """Is simulated throughput a valid independent witness on this station's gamma?"""

    DOT_SHAPE = "box"
    """Graphviz node shape used by Network.to_dot()."""

    def __init__(self, gamma=None, mu=None, weight=1.0, *, name=None):
        # `isfinite` first: NaN passes every ordering comparison, so `nan <= 0` is False.
        if gamma is not None and (not math.isfinite(gamma) or gamma <= 0):
            raise ValueError(f"gamma must be a finite number > 0, got {gamma}")
        if mu is None:
            raise ValueError("mu is required")
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError(f"mu must be a finite number > 0, got {mu}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight must be a finite number > 0, got {weight}")
        self._gamma = gamma
        self._gamma_explicit = gamma is not None
        self.mu = mu
        self.weight = weight
        self.name = name

    @property
    def gamma(self):
        """Arrival rate. Either passed explicitly or derived by a Network (spec 4.1)."""
        if self._gamma is None:
            raise ValueError(
                f"station {self.name!r} has no gamma: pass gamma=... explicitly, or add "
                f"the station to a Network, which derives it from the traffic equations"
            )
        return self._gamma

    def bind_gamma(self, value):
        """Attach a Network-derived gamma. Idempotent for an identical value.

        gamma is derived-only for stations in a Network: there is no silent override of an
        explicitly constructed value, and no rebinding to a second network (spec 4.1).
        """
        if self._gamma_explicit:
            raise ValueError(
                f"station {self.name!r} was constructed with an explicit gamma="
                f"{self._gamma}; gamma is derived-only for stations in a Network"
            )
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"derived gamma must be a finite number > 0, got {value}")
        if self._gamma is not None and self._gamma != value:
            raise ValueError(
                f"station {self.name!r} is already bound to gamma={self._gamma}, "
                f"cannot rebind to {value}"
            )
        self._gamma = value

    @abstractmethod
    def sojourn_time(self, S):
        """Expected sojourn time E[T] under capacity S.

        Raises InstabilityError if S*mu <= gamma. Raises ValueError first if gamma is
        unbound (no explicit gamma at construction and not yet bound by a Network) —
        the gamma property itself raises before the stability check can run.
        """

    @abstractmethod
    def sim_node(self, S, job_class):
        """This station's qsim node dict under capacity S (spec 5.2)."""

    @property
    @abstractmethod
    def alloc_cost(self):
        """Cost coefficient used in the budget constraint and eq 21."""

    @property
    @abstractmethod
    def default_zeta(self):
        """Strictly-positive starting guess for zeta."""

    def zeta_from(self, T, S):
        """Invert the functional form (eq 22) for an externally supplied E[T].

        Pure station arithmetic, independent of where E[T] came from — the analytic
        sojourn time or a simulation run.
        """
        return T * (S * self.mu - self.gamma)

    def zeta(self, S):
        """Eq 22 evaluated at this station's own analytic sojourn time."""
        return self.zeta_from(self.sojourn_time(S), S)

    def check_stable(self, S):
        """Raise InstabilityError if capacity S leaves this station unstable.

        Public counterpart to `_check_stable`, which takes the already-computed
        effective rate. Lets a caller fail fast before spending an expensive
        evaluation (spec 7.3) without reimplementing the check or its message.
        """
        self._check_stable(S * self.mu)

    def _check_stable(self, mu_eff):
        if mu_eff <= self.gamma:
            raise InstabilityError(
                f"station {self.name!r} unstable: S*mu={mu_eff} <= gamma={self.gamma}"
            )


class SingleServerStation(Station):
    """Abstract base for one-server queues. Concrete subclasses supply sojourn_time."""

    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, name=None):
        super().__init__(gamma, mu, weight, name=name)
        if not math.isfinite(c) or c <= 0:
            raise ValueError(f"c must be a finite number > 0, got {c}")
        self.c = c

    @property
    def alloc_cost(self):
        return self.c

    @property
    def default_zeta(self):
        return 1.0


class GG1Station(SingleServerStation):
    """G/G/1 queue via the Kingman / Allen-Cunneen mean-value approximation.

        E[T] = (1/mu_eff) * [1 + ((cov_a^2 + cov_s^2)/2) * rho/(1-rho)]

    with mu_eff = S*mu and rho = gamma/mu_eff. Exact for any M/G/1 (cov_a == 1).
    """

    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, cov_a, cov_s, name=None):
        super().__init__(gamma, mu, weight, c=c, name=name)
        if not math.isfinite(cov_a) or cov_a < 0:
            raise ValueError(f"cov_a must be a finite number >= 0, got {cov_a}")
        if not math.isfinite(cov_s) or cov_s < 0:
            raise ValueError(f"cov_s must be a finite number >= 0, got {cov_s}")
        self.cov_a = cov_a
        self.cov_s = cov_s

    def sojourn_time(self, S):
        mu_eff = S * self.mu
        self._check_stable(mu_eff)
        rho = self.gamma / mu_eff
        k = (self.cov_a ** 2 + self.cov_s ** 2) / 2.0
        return (1.0 / mu_eff) * (1.0 + k * rho / (1.0 - rho))

    @classmethod
    def mm1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None):
        """M/M/1 preset (cov_a = cov_s = 1); zeta is identically 1."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=1.0, name=name)

    @classmethod
    def md1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None):
        """M/D/1 preset (cov_a = 1, cov_s = 0); zeta = 1 - rho/2."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=0.0, name=name)

    def sim_node(self, S, job_class):
        return {
            "name": self.name,
            "type": "queue",
            "servers": 1,
            "scheduling": "fcfs",
            "capacity": None,
            "service": {
                job_class: {"distribution": distribution_dict(S * self.mu, self.cov_s ** 2)}
            },
        }


class ForkJoinStation(Station):
    """Fork-join station: two parallel servers sharing one capacity S.

    Both servers receive capacity S, so effective rates are m1 = S*mu (slower) and
    m2 = S*(r*mu) (faster), preserving the ratio r for all S. mu is the slower server's
    rate; the faster server's rate is r*mu (r >= 1). Cost coefficient is c1 + c2.
    """

    sim_conservation_checked = False   # qsim-service#8; delete this line when it lands
    DOT_SHAPE = "box3d"

    def __init__(self, gamma=None, mu=None, weight=1.0, *, r, c1, c2, name=None):
        super().__init__(gamma, mu, weight, name=name)
        if not math.isfinite(r) or r < 1:
            raise ValueError(f"r must be a finite number >= 1, got {r}")
        if not math.isfinite(c1) or c1 <= 0:
            raise ValueError(f"c1 must be a finite number > 0, got {c1}")
        if not math.isfinite(c2) or c2 <= 0:
            raise ValueError(f"c2 must be a finite number > 0, got {c2}")
        self.r = r
        self.c1 = c1
        self.c2 = c2

    @property
    def alloc_cost(self):
        return self.c1 + self.c2

    @property
    def default_zeta(self):
        return 1.5

    def sojourn_time(self, S):
        m1 = S * self.mu          # slower server (binds stability)
        m2 = S * self.r * self.mu  # faster server
        self._check_stable(m1)
        return t_ul(self.gamma, m1, m2)

    def sim_node(self, S, job_class):
        """Two branches at S*mu and S*r*mu joined on "all" — the shared-capacity semantics."""
        return {
            "name": self.name,
            "type": "fork-join",
            "branches": [
                {"service": {job_class: {
                    "distribution": distribution_dict(S * self.mu, 1.0)}}},
                {"service": {job_class: {
                    "distribution": distribution_dict(S * self.r * self.mu, 1.0)}}},
            ],
            "join": "all",
        }
