"""Station hierarchy: each station owns its queueing math."""

import math
from abc import ABC, abstractmethod

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul
from qopt.forkjoin_policy import R_STAR_TUNED, optimal_ray, resolve_r_star


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
        mu: base service rate (for a fork-join station, the rate of whichever
            server its ray leaves effectively slower -- see ForkJoinStation).
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
        explicitly constructed value, and no rebinding to a conflicting one (spec 4.1).
        What is rejected is a *disagreement* about the arrival rate, not reuse — putting
        these stations in a second Network that derives the same gamma succeeds, which is
        what makes the two-run examples' fresh-Network choice a matter of isolating mutable
        state rather than a necessity.
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

    def retune(self, S):
        """Adapt any free internal policy parameter to capacity S, returning the capacity
        that buys the same SPEND under the station's possibly-changed `alloc_cost`.

        Called by the Optimizer once per iteration, after eq 21 has allocated. A station
        with no free parameter has nothing to adapt and nothing to reprice, so the default
        returns S untouched -- this hook costs every other station type exactly nothing.

        Returning a capacity rather than mutating S is what keeps the budget satisfied
        across a repricing, and what keeps S meaning what the station says it means.
        """
        return S

    def reset_policy(self):
        """Restore any free internal policy parameter to its constructed value.

        `retune` mutates, so the Optimizer calls this once at the start of every run to
        undo whatever a previous run left behind. That is what keeps a run a pure function
        of (stations-as-constructed, budget) even when the same objects are reused, and it
        matters beyond reproducibility: the retuned parameter can move the station's own
        stability floor, and the Optimizer's feasibility check reads that floor BEFORE the
        first retune gets a chance to correct it.

        A station with no free parameter has nothing to restore, so the default does
        nothing -- this hook costs every other station type exactly nothing.
        """

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
    """Fork-join station: two parallel servers, one capacity variable S.

    Construction describes the HARDWARE: server 1 has base rate `mu` and costs `c1`,
    server 2 has base rate `mu*r` (r >= 1) and costs `c2`. `r_star` then chooses the
    POLICY -- which ray of the effective-rate plane the station runs on:

        m2 = r_star * m1,   with m1 = S*mu the rate bought for server 1

    so server 2 buys capacity S*r_star/r and the station spends S*(c1 + c2*r_star/r).
    A ray is what keeps spend linear in S, which is what eq 21's budget column requires.

    `r_star` takes either a positive float -- some fixed ray of the family -- or one of
    three named policies, and both established rules are members of the family:

        R_STAR_INVARIANT_R  r_star = r, the default: both servers get capacity S, cost
                            c1 + c2. qopt's incumbent.
        R_STAR_EQUAL_RATE   r_star = 1: server 2 gets S/r, cost c1 + c2/r. The paper's
                            rule, and c1 + c2/r is the exact cost of its own capacities.
        R_STAR_TUNED        r_star solved from the local optimality condition at this
                            station's own spend, once per optimizer iteration. Neither
                            incumbent dominates the other -- the paper's rule wins
                            `classical_dominant` by 24.55% and loses `quantum_dominant`
                            by 5.47% -- and this finds the better ray in both.

    A tuned station starts on the ray r_star = 1, which is the one that minimizes its
    stability floor -- the Optimizer checks feasibility once, before any retune, so
    starting at the incumbent would refuse budgets a tuned station can actually serve. It
    is then MUTATED by `retune` during a run, so its final `r_star` is readable off the
    station afterwards. See
    docs/forkjoin-s2-policy/ and qopt.forkjoin_policy.

    Fields the allocator and eqs 21/22 read, which are EFFECTIVE and not the constructor
    arguments:
        mu: the binding (effectively slower) server's base rate, mu*min(1, r_star).
        r:  the effective faster/slower ratio, max(r_star, 1/r_star) >= 1.

    So `r_star < 1` swaps which server binds, and `c1` then pairs with `r`'s slot rather
    than with `mu`'s. Only `alloc_cost` and `server_capacities` need that pairing, and
    both take it from `r_base` -- the constructed r -- rather than from `r`.
    """

    sim_conservation_checked = False   # qsim-service#8; delete this line when it lands
    DOT_SHAPE = "box3d"

    def __init__(self, gamma=None, mu=None, weight=1.0, *, r, c1, c2, r_star=None,
                 name=None):
        if not math.isfinite(r) or r < 1:
            raise ValueError(f"r must be a finite number >= 1, got {r}")
        self._policy, r_star = resolve_r_star(r_star, r)
        # Anchor on whichever server the ray leaves effectively slower, so eq 21's base
        # term gamma/mu provisions the BINDING server and eq 22's zeta is taken against a
        # rate the station actually has. Without this, r_star < 1 silently starves server
        # 2: `_check_stable(S*mu)` guards one server, and it would be the wrong one.
        # `mu` may be None here -- pass it through so Station raises the canonical error.
        k = min(1.0, r_star)
        super().__init__(gamma, mu if mu is None else mu * k, weight, name=name)
        if not math.isfinite(c1) or c1 <= 0:
            raise ValueError(f"c1 must be a finite number > 0, got {c1}")
        if not math.isfinite(c2) or c2 <= 0:
            raise ValueError(f"c2 must be a finite number > 0, got {c2}")
        self.mu_base = mu
        self.r_base = r
        self.c1 = c1
        self.c2 = c2
        self._initial_r_star = r_star
        self._anchor(r_star)

    def _anchor(self, r_star):
        """Move onto the ray `r_star`, re-deriving the effective `mu` and `r` from it.

        `mu` is recomputed from `mu_base` by the SAME expression __init__ handed to
        Station, so re-anchoring to the constructed ray is bit-for-bit a no-op.
        """
        k = min(1.0, r_star)
        self.r_star = r_star
        self.mu = self.mu_base * k
        self.r = max(1.0, r_star) / k

    @property
    def policy(self):
        """Which r_star policy this station runs (a qopt.R_STAR_* constant).

        Read-only and fixed at construction: retuning moves `r_star` within the `tuned`
        policy, it does not consume or change the policy itself.
        """
        return self._policy

    def retune(self, S):
        """Under `tuned`, move to the locally optimal ray for this station's own spend.

        The spend is `S*alloc_cost`, which in effective rates is exactly
        `beta_1*m1 + beta_2*m2` with `beta_k = c_k/mu_k` -- so the local condition is
        applied at the spend eq 21 gave, priced as eq 21 priced it. That consistency is
        the whole difference from the inner-split embedding this replaces, which re-split
        at a frozen `c1 + c2`: there "S" stopped meaning "server 1's capacity", eq 22's
        zeta anchored to a rate the station did not have, and convergence degraded.

        One scalar minimization, no inner iteration -- at a fixed spend the optimum is
        determined. The fixed point is closed by the OUTER loop, because the spend comes
        from eq 21, whose prices depend on the r_star chosen here.

        Deliberately mutating, and the only method that rewrites a station's queueing
        coefficients (`bind_gamma` mutates too, but only to attach a derived gamma once):
        `r_star`, `mu` and `r` are what the allocator reads, so the retuned station has to
        *be* the retuned station. The chosen ray is readable off `r_star` after a run.

        Mutation makes a tuned station stateful, so `reset_policy` exists to undo it and
        the Optimizer calls that at the start of every run. Reusing the same objects is
        therefore safe and bit-for-bit reproducible; between runs, though, `r_star` reads
        as the last run's answer rather than as the constructed ray.
        """
        if self._policy != R_STAR_TUNED:
            return S
        spend = S * self.alloc_cost
        self._anchor(optimal_ray(self.gamma, self.mu_base, self.r_base,
                                 self.c1, self.c2, spend))
        return spend / self.alloc_cost

    def reset_policy(self):
        """Return to the constructed ray, undoing every `retune` a previous run applied.

        Inert on every policy but `tuned`, whose ray is the only one that moves: for the
        others this re-anchors to the value already held, which `_anchor` does bit-for-bit.

        For `tuned` it is load-bearing, not just hygiene. The station's floor over the
        family is minimized at exactly the constructed ray r_star = 1, and strictly so, so
        a run that ends anywhere else advertises a higher floor than the policy's own; and
        the Optimizer evaluates
        `min_feasible_budget` ONCE, before the first retune. Without this a station reused
        at a lower budget rejects it against the PREVIOUS run's floor -- so a descending
        budget sweep broke partway down while a freshly constructed equivalent converged.

        Restoring the ray is the whole fix, and repricing the floor instead would not be:
        eq 21 is allocated before the first retune too, so at a budget between the two
        floors a stale ray produces negative slack and an immediate InstabilityError.
        """
        self._anchor(self._initial_r_star)

    @property
    def alloc_cost(self):
        """c1 + c2*r_star/r -- the true cost of the two capacities the ray buys.

        Parenthesized so that the default r_star == r_base divides to exactly 1.0 (IEEE
        754 gives x/x == 1.0 for any finite x != 0) and this is bit-for-bit c1 + c2.
        """
        return self.c1 + self.c2 * (self.r_star / self.r_base)

    def server_capacities(self, S):
        """(S_1, S_2) -- the capacity each CONSTRUCTED server receives at variable S.

        Server 1 takes S by definition of the variable; server 2 takes S*r_star/r, which
        equals S only at the default r_star = r. Reporting that sums a fork-join's
        capacity per unit of hardware needs the two separately.
        """
        return S, S * (self.r_star / self.r_base)

    @property
    def default_zeta(self):
        return 1.5

    def sojourn_time(self, S):
        m1 = S * self.mu          # slower server (binds stability)
        m2 = S * self.r * self.mu  # faster server
        self._check_stable(m1)
        return t_ul(self.gamma, m1, m2)

    def sim_node(self, S, job_class):
        """The ray's two effective rates as branches joined on "all".

        `mu` and `r` are the EFFECTIVE anchor and ratio, so the branches come out ordered
        slower-first and this emits the ray the station actually runs on at any r_star.
        """
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
