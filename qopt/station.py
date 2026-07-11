"""Station hierarchy: each station owns its queueing math."""

import math
from abc import ABC, abstractmethod

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul


class Station(ABC):
    """A node in the queueing network.

    Fields (used directly by the allocator and eqs 21/22):
        gamma: fixed arrival rate.
        mu: base service rate (for a fork-join station, the slower server's rate).
        weight: sojourn-time weight (omega).
        name: optional label for reporting.
    """

    def __init__(self, gamma, mu, weight=1.0, *, name=None):
        # `isfinite` first: NaN passes every ordering comparison, so `nan <= 0` is False.
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gamma must be a finite number > 0, got {gamma}")
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError(f"mu must be a finite number > 0, got {mu}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight must be a finite number > 0, got {weight}")
        self.gamma = gamma
        self.mu = mu
        self.weight = weight
        self.name = name

    @abstractmethod
    def sojourn_time(self, S):
        """Expected sojourn time E[T] under capacity S. Raises InstabilityError if S*mu <= gamma."""

    @property
    @abstractmethod
    def alloc_cost(self):
        """Cost coefficient used in the budget constraint and eq 21."""

    @property
    @abstractmethod
    def default_zeta(self):
        """Strictly-positive starting guess for zeta."""

    def zeta(self, S):
        """Invert the functional form (eq 22): zeta = E[T] * (S*mu - gamma)."""
        return self.sojourn_time(S) * (S * self.mu - self.gamma)

    def _check_stable(self, mu_eff):
        if mu_eff <= self.gamma:
            raise InstabilityError(
                f"station {self.name!r} unstable: S*mu={mu_eff} <= gamma={self.gamma}"
            )


class SingleServerStation(Station):
    """Abstract base for one-server queues. Concrete subclasses supply sojourn_time."""

    def __init__(self, gamma, mu, weight=1.0, *, c, name=None):
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

    def __init__(self, gamma, mu, weight=1.0, *, c, cov_a, cov_s, name=None):
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
    def mm1(cls, gamma, mu, weight=1.0, *, c, name=None):
        """M/M/1 preset (cov_a = cov_s = 1); zeta is identically 1."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=1.0, name=name)

    @classmethod
    def md1(cls, gamma, mu, weight=1.0, *, c, name=None):
        """M/D/1 preset (cov_a = 1, cov_s = 0); zeta = 1 - rho/2."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=0.0, name=name)


class ForkJoinStation(Station):
    """Fork-join station: two parallel servers sharing one capacity S.

    Both servers receive capacity S, so effective rates are m1 = S*mu (slower) and
    m2 = S*(r*mu) (faster), preserving the ratio r for all S. mu is the slower server's
    rate; the faster server's rate is r*mu (r >= 1). Cost coefficient is c1 + c2.
    """

    def __init__(self, gamma, mu, weight=1.0, *, r, c1, c2, name=None):
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
