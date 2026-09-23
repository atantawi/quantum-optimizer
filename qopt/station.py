"""Station hierarchy: each station owns its queueing math."""

import math
from abc import ABC, abstractmethod

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul
from qopt.forkjoin_policy import R_STAR_TUNED, optimal_ray, resolve_r_star
from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE, resolve_zeta_mode


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


_FD_STEP = 1e-7
"""Central-difference step for `Station.dT_dS`, as a FRACTION OF SPARE CAPACITY.

Scaled to spare capacity rather than to S, which is what keeps `S - h` strictly inside the
stability region for a station run arbitrarily close to its boundary. 1e-7 is near the
cube-root-of-epsilon optimum for a central difference and is measured to agree with the
closed forms to 6.7e-09 relative (docs/slope-calibrated-zeta/probe-output.txt section 1).
"""


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

    def __init__(self, gamma=None, mu=None, weight=1.0, *, name=None,
                 zeta_mode=ZETA_LEVEL):
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
        # Validated here so `zeta_from` can branch on ZETA_SLOPE alone and treat
        # anything else as level -- an unknown mode cannot reach the hot path.
        self._zeta_mode = resolve_zeta_mode(zeta_mode)

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

    @property
    def zeta_mode(self):
        """Which ζ calibration this station uses (a qopt.ZETA_* constant).

        Read-only and fixed at construction, for the reason `ForkJoinStation.policy` is:
        a run prices iterations against a calibration, so changing it mid-run would leave
        a converged ζ that no single rule produced.
        """
        return self._zeta_mode

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

    @property
    def min_spend(self):
        """This station's own term in `min_feasible_budget`: the spend that just stabilizes
        it, `alloc_cost * gamma / mu`.

        A hook rather than an expression in the allocator because a station with a free
        policy parameter can be sitting on a different value than the one a run will start
        from -- and it is the STARTING value that decides which budgets the Optimizer can
        serve. Overridden by `ForkJoinStation` for exactly that reason; for every other
        station this is the plain expression, unchanged.

        The parenthesization is load-bearing: eq 21 needs `base = gamma/mu` for the capacity
        formula and prices the floor as `alloc_cost * base`, so writing this `(a*g)/m`
        instead of `a*(g/m)` puts the two one ulp apart whenever `mu` is not a power of two
        -- and a budget in that gap passes the Optimizer's guard only to meet a non-positive
        slack in `allocate`. Pinned by
        test_the_reported_floor_is_bit_for_bit_the_one_allocate_prices.
        """
        return self.alloc_cost * (self.gamma / self.mu)

    def dT_dS(self, S):
        """dE[T]/dS at capacity S -- negative, since capacity cannot lengthen a queue.

        Only slope-calibrated ζ reads this, so a level-mode station never pays for it.

        Deliberately concrete rather than abstract: slope calibration then works for ANY
        station, including a user subclass qopt has never seen, and no existing subclass
        breaks. `GG1Station` and `ForkJoinStation` override it with closed forms, which
        this default is tested against.

        The step is a fraction of SPARE CAPACITY, `S - gamma/mu`, which buys two things a
        fixed step does not: `S - h` is inside the stability region by construction for
        any stable S, and at `S == gamma/mu` the step is 0 so `sojourn_time` raises
        InstabilityError rather than this dividing by zero. Below the boundary h is
        negative, and the `S + h` evaluation is the one that raises -- so do not reorder
        these two calls or take an absolute value.

        For a fork-join this differences along the station's FIXED CURRENT RAY, because
        `sojourn_time` scales both servers with S. That is the radial derivative slope
        calibration needs (see ForkJoinStation.dT_dS), so the default cannot accidentally
        reproduce the `forkjoin_policy._dt_dm1` defect.
        """
        h = _FD_STEP * (S - self.gamma / self.mu)
        return (self.sojourn_time(S + h) - self.sojourn_time(S - h)) / (2.0 * h)

    def phi(self, S):
        """Elasticity of E[T] in spare capacity: -d log T / d log x, with x = S*mu - gamma.

        The ratio of the true slope to the one eq 22's level calibration implies, so
        `phi == 1` means eq 22 is already slope-correct at S and the two calibrations
        agree. Identically 1 for M/M/1; exactly `1 - rho` for a cov = 0 station; up to
        1.64 for G/G/1 with cov = 5.

        Uses this station's ANALYTIC `sojourn_time`, always -- including on the simulated
        path, where `zeta_from` receives a MEASURED E[T] and this supplies only the shape.
        That hybrid is load-bearing in both directions (spec section 4.4): a fully
        analytic slope ζ cancels T and would cut the simulator out of the allocation
        entirely, while a secant slope from consecutive loop iterates degenerates to 0/0
        as they converge. It also costs no simulation calls.

        Overridable: a user who knows the true arrival variability but cannot express it
        as a constructor `cov_a` should override this rather than reach for a new API.
        """
        x = S * self.mu - self.gamma
        return -self.dT_dS(S) * x / (self.mu * self.sojourn_time(S))

    def zeta_from(self, T, S):
        """Invert the functional form for an externally supplied E[T].

        Pure station arithmetic, independent of where E[T] came from — the analytic
        sojourn time or a simulation run. The single point at which either calibration is
        applied, so the branch lives here and nowhere else.

        LEVEL (eq 22, the default) makes the surrogate zeta/x pass through (S, T). SLOPE
        matches its derivative instead, which is the only thing eq 21 reads -- see
        qopt/zeta.py. The level arm is the shipped expression, operation for operation, so
        the default path does not move by one ulp.

        Both arms are LINEAR IN T, which `Optimizer._noise_floor` depends on: it passes a
        CI half-width in the T position and needs the result to be the correspondingly
        scaled perturbation of zeta.

        No clamp on phi. It is legitimately tiny -- exactly `1 - rho` for a cov = 0
        station, measured down to 1e-6 -- and `allocate` requires only that zeta be
        finite and positive.
        """
        x = S * self.mu - self.gamma
        if self._zeta_mode == ZETA_SLOPE:
            phi = self.phi(S)
            if not (math.isfinite(phi) and phi > 0.0):
                raise ValueError(
                    f"station {self.name!r}: slope-calibrated zeta needs a finite, "
                    f"strictly positive phi, got {phi} at S={S}. E[T] must strictly "
                    f"decrease in capacity for there to be a slope to calibrate to."
                )
            return phi * T * x
        return T * x

    def zeta(self, S):
        """This station's own calibration -- level (eq 22) or slope, per its mode --
        evaluated at its own analytic sojourn time."""
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

        `retune` mutates, so the Optimizer calls this once per run -- after every preflight
        guard, so that a run REJECTED AT PREFLIGHT leaves a previous answer intact, and
        before eq 21's first allocation, which prices each station at the parameter's
        current value. Preflight is the limit of that protection: once the loop is entered
        the parameter moves, so a failure after that point -- an analyzer or transport
        error, a mid-loop instability, a strict quality failure -- leaves it wherever the
        loop got to, as any partially-completed run would. `min_spend` stays correct
        throughout regardless, since it reads the constructed value. That
        is what keeps a run a pure function of (stations-as-constructed, budget) even when
        the same objects are reused, and it is why `min_spend` is a separate hook: the
        feasibility check runs before this does, so it cannot read the current value.

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

    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, name=None,
                 zeta_mode=ZETA_LEVEL):
        super().__init__(gamma, mu, weight, name=name, zeta_mode=zeta_mode)
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

    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, cov_a, cov_s, name=None,
                 zeta_mode=ZETA_LEVEL):
        super().__init__(gamma, mu, weight, c=c, name=name, zeta_mode=zeta_mode)
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

    def dT_dS(self, S):
        """Closed form of the Allen-Cunneen derivative.

            E[T] = 1/m + k*gamma/(m*x),   m = S*mu,  x = m - gamma,  k = (cov_a^2+cov_s^2)/2

        differentiated in S:

            dT/dS = -mu * [ 1/m^2 + k*gamma*(2m - gamma)/(m*x)^2 ]

        Every term is negative, so no sign can cancel silently. At k = 1 this gives
        phi == 1 exactly, which is the M/M/1 invariant; at k = 0 it gives phi = 1 - rho.
        """
        m = S * self.mu
        self._check_stable(m)
        x = m - self.gamma
        k = (self.cov_a ** 2 + self.cov_s ** 2) / 2.0
        return -self.mu * (
            1.0 / m ** 2 + k * self.gamma * (2.0 * m - self.gamma) / (m * x) ** 2
        )

    @classmethod
    def mm1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None,
            zeta_mode=ZETA_LEVEL):
        """M/M/1 preset (cov_a = cov_s = 1); zeta is identically 1.

        phi is identically 1 too, so `zeta_mode` makes no difference on this station --
        the two calibrations agree exactly. It is accepted so a network can be switched
        wholesale without special-casing its M/M/1 members.
        """
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=1.0, name=name,
                   zeta_mode=zeta_mode)

    @classmethod
    def md1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None,
            zeta_mode=ZETA_LEVEL):
        """M/D/1 preset (cov_a = 1, cov_s = 0); zeta = 1 - rho/2."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=0.0, name=name,
                   zeta_mode=zeta_mode)

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
                 name=None, zeta_mode=ZETA_LEVEL):
        if not math.isfinite(r) or r < 1:
            raise ValueError(f"r must be a finite number >= 1, got {r}")
        self._policy, r_star = resolve_r_star(r_star, r)
        # Hand Station the BINDING server's rate, so its validation is applied to the rate
        # this station will actually run on: `mu * k` can underflow to zero for an extreme
        # ray where `mu` alone is fine, and that has to be rejected here.
        #
        # Validation is ALL this line decides. `_anchor` below recomputes `self.mu` from
        # `mu_base` by the identical expression, so the value Station stores is overwritten
        # either way. The anchoring itself -- what keeps eq 21's base term and eq 22's zeta
        # on the binding server, and what stops `r_star < 1` from silently starving server
        # 2, since `_check_stable(S*mu)` guards only one of them -- lives in `_anchor`.
        # `mu` may be None here -- pass it through so Station raises the canonical error.
        k = min(1.0, r_star)
        super().__init__(gamma, mu if mu is None else mu * k, weight, name=name,
                         zeta_mode=zeta_mode)
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
        a run that ends anywhere else leaves the station needing more budget than the
        policy does. Reusing it at a lower budget must be served against the policy's floor
        -- which takes both halves: `min_spend` reports that floor to the Optimizer's
        feasibility check, and this puts the station back on the ray eq 21 is priced at.
        Neither alone is enough. Without the reset, a budget the check has just cleared
        gives eq 21 negative slack on the stale ray, and the run dies inside the loop
        rather than starting. Without `min_spend`, the check rejects a budget the station
        can serve, and a descending budget sweep breaks partway down.
        """
        self._anchor(self._initial_r_star)

    def _spend_floor_on(self, r_star):
        """`alloc_cost * gamma / mu` for the ray `r_star`, without moving onto it.

        Repeats `alloc_cost`'s and `_anchor`'s expressions rather than calling them, which
        is deliberate: written this way it is bit-for-bit the base-class expression when
        `r_star` is the ray the station is already on, so `min_spend` below cannot perturb
        any budget derived from the floor. That includes the grouping -- see
        `Station.min_spend` for why `a*(g/m)`, not `(a*g)/m`, is the spelling eq 21 needs.
        """
        alloc = self.c1 + self.c2 * (r_star / self.r_base)
        return alloc * (self.gamma / (self.mu_base * min(1.0, r_star)))

    @property
    def min_spend(self):
        """The floor at the ray a RUN starts from, not at the ray this station is on.

        The two differ only under `tuned`, whose ray `retune` moves and `reset_policy`
        restores. Reading the current ray made the exported `min_feasible_budget` both
        history-dependent and wrong: after a generous run it reported that run's ray floor
        while `Optimizer.run()` still served a smaller budget, since the run restores the
        ray before it checks. Budgets derived from the helper were inflated by run history.

        Under `tuned` the starting ray is also the family's floor-minimizing one (see
        `_INITIAL_R_STAR`), so this is simultaneously the minimum over every reachable ray
        -- which is what makes it the honest answer to "what budget does a run need".

        The two floors coincide except for a tuned station still carrying a finished run's
        ray. `allocate`, which prices the current ray, refuses a budget below the floor it
        prices, so composing the two can never allocate unstably -- it fails loudly and
        names the number it needed. `Optimizer.run()` never meets that case at all, having
        restored the ray before it allocates.
        """
        return self._spend_floor_on(self._initial_r_star)

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

    def dT_dS(self, S):
        """Radial derivative of `t_ul` along this station's CURRENT ray.

        Both effective rates scale with S, so this differentiates
        t_ul(gamma, a*S, b*S) in S with a = mu and b = mu*r held fixed. That is the
        derivative slope calibration needs, and it equals the true marginal of the
        coupled problem only ON the optimal ray -- which the Optimizer guarantees by
        calling `retune` LAST in each iteration.

        `t_bot` needs no max() here. `_anchor` pairs `mu` with the slower server and
        keeps `r >= 1`, so m1 <= m2 always and `t_ul`'s max() resolves to 1/x1 at every
        point of the ray. The branch therefore never switches and this is smooth, which
        is also why differencing `sojourn_time` agrees with it.

        NOT `forkjoin_policy._dt_dm1`: that takes the non-bottleneck branch at m1 == m2
        and drops the alpha*t_bot term, which is fine for the measure-zero kink inside
        `_min_on_spend_line` but wrong for pricing dT/d(spend) -- a 17.3% error at
        spend/floor = 1.05, and r_star = 1 is where tight budgets sit.

        alpha is homogeneous of degree -1 in S, hence the -alpha/S term.
        """
        a, b = self.mu, self.mu * self.r      # effective rates: a binds, b >= a
        m1, m2 = a * S, b * S
        self._check_stable(m1)
        x1, x2 = m1 - self.gamma, m2 - self.gamma
        D = x1 + x2
        t_ub = 1.0 / x1 + 1.0 / x2 - 1.0 / D
        t_bot = 1.0 / x1
        alpha = (self.gamma / m1 + self.gamma / m2) / 8.0
        d_ub = -a / x1 ** 2 - b / x2 ** 2 + (a + b) / D ** 2
        d_bot = -a / x1 ** 2
        return (alpha / S) * (t_ub - t_bot) + (1.0 - alpha) * d_ub + alpha * d_bot

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
