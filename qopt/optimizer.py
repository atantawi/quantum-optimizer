"""Fixed-point optimization loop (paper Steps 0-5), analytic or simulation-backed."""

import math
import warnings
from dataclasses import dataclass, field

from qopt.allocator import allocate, min_feasible_budget, noise_floor
from qopt.analyzer import AnalyticAnalyzer
from qopt.exceptions import InfeasibleBudgetError, SimulationQualityError
from qopt.network import Network


@dataclass
class Result:
    """Outcome of an optimization run (lists aligned to the station order)."""

    capacities: list
    sojourn_times: list
    zeta: list
    objective: float
    iterations: int
    converged: bool
    residual: float  # final DAMPED step, theta*|S_target - S| -- what the iterate
                      # actually moved. The loop tests convergence on residual/damping,
                      # so `tol` and `noise_floor` are both TARGET-space quantities and
                      # `residual` is smaller than the value compared against them by a
                      # factor of theta. At theta = 1.0 the three coincide.

    # Simulation-path diagnostics. All defaulted, so analytic construction is unchanged.
    sojourn_ci: list | None = None      # per-station (lower, upper), or None for a
                                         # station whose CI was missing; None (the whole
                                         # field) when analytic. Sourced from the fresh-
                                         # seed FINAL evaluation at default settings, and
                                         # from the last loop iterate when
                                         # final_evaluation=False suppresses that run.
    noise_floor: float | None = None   # UNDAMPED target-space spread attributable to
                                        # noise (6.4), directly comparable to `tol` and to
                                        # residual/damping. Taken from the last LOOP
                                        # iteration's common-random-numbers intervals -
                                        # a different run than sojourn_ci above whenever a
                                        # final evaluation ran, the same one when it did
                                        # not.
    stop_reason: str = "tol"           # "tol" | "noise-floor" | "max_iter"
    warm_start_iterations: int = 0     # analytic iterations before the simulated phase
    degraded: list = field(default_factory=list)   # per-iteration quality audit (6.8, 7.2)
    system_response_time: object = None           # qsim diagnostic; not optimized
    sim_calls: int = 0                            # POSTs issued — the real cost meter


class Optimizer:
    """Drives the fixed-point iteration for the capacity allocation problem.

    Loop: allocate from an initial zeta guess, then repeatedly recompute zeta from the
    current capacities (eq 22) and re-allocate (eq 21) until the step falls below the
    stopping threshold or max_iter is reached.

    `Optimizer(stations, budget)` is bit-identical to the pre-simulation implementation:
    it defaults to AnalyticAnalyzer with damping 1.0 and max_iter 1000, and the
    CI-driven machinery stays inert because the analytic path reports no CI.

    `strict=True` here is deliberately LATE, not fast: degraded entries accumulate
    across every loop iteration and the final evaluation, and SimulationQualityError is
    raised only once the whole run has finished (see `run`, near the end). This is a
    different timing from `SimulationAnalyzer(strict=True)`, which fails at the first
    degraded `evaluate()` call. A fully-degraded run at max_iter=20 therefore still
    burns all 21 POSTs before this raises (finding 7) - use the analyzer's strict flag
    instead if failing fast matters more than seeing the whole audit trail.
    """

    def __init__(self, stations, budget, *, analyzer=None, tol=1e-9, max_iter=None,
                 initial_zeta=None, damping=None, noise_kappa=1.0,
                 final_evaluation=True, strict=False, warm_start=True):
        if isinstance(stations, Network):
            self.network = stations
            self.stations = list(stations.stations)
        else:
            self.network = None
            self.stations = list(stations)
        self.budget = budget
        self.analyzer = AnalyticAnalyzer() if analyzer is None else analyzer
        self.tol = tol
        self.initial_zeta = initial_zeta
        self.final_evaluation = final_evaluation
        self.strict = strict
        self.warm_start = warm_start

        # Each simulated iteration is a full simulation run, so the caps differ by kind.
        stochastic = self.analyzer.is_stochastic
        self.max_iter = (20 if stochastic else 1000) if max_iter is None else max_iter
        self.damping = (0.5 if stochastic else 1.0) if damping is None else damping
        self.noise_kappa = noise_kappa

        if not math.isfinite(self.damping) or not 0.0 < self.damping <= 1.0:
            raise ValueError(
                f"damping must be a finite number in (0, 1], got {self.damping}"
            )
        if not math.isfinite(self.noise_kappa) or self.noise_kappa < 0:
            raise ValueError(
                f"noise_kappa must be a finite number >= 0, got {self.noise_kappa}"
            )

        if getattr(self.analyzer, "seed_policy", None) == "fixed" and not final_evaluation:
            warnings.warn(
                "seed_policy='fixed' with final_evaluation=False reports metrics from "
                "the common random numbers sample path: the loop converges crisply but "
                "the reported numbers are biased toward that one sample path. Set "
                "final_evaluation=True for an independently seeded final run (spec 6.5).",
                RuntimeWarning,
                stacklevel=2,
            )

    def run(self):
        stations = self.stations

        # Guard: budget must exceed the minimum needed for stability (eq 21 slack > 0).
        # `isfinite` first because NaN slips through every ordering comparison below.
        if not math.isfinite(self.budget):
            raise ValueError(f"budget must be a finite number, got {self.budget}")
        min_budget = min_feasible_budget(stations)
        if self.budget <= min_budget:
            raise InfeasibleBudgetError(
                f"budget {self.budget} <= minimum feasible {min_budget}"
            )

        # Guard: finite, strictly-positive initial zeta.
        if self.initial_zeta is None:
            zeta = [st.default_zeta for st in stations]
        else:
            zeta = list(self.initial_zeta)
            if len(zeta) != len(stations):
                raise ValueError("initial_zeta length must match number of stations")
        if not all(math.isfinite(z) and z > 0 for z in zeta):
            raise ValueError(
                f"initial zeta values must be finite and strictly positive, got {zeta}"
            )

        stochastic = self.analyzer.is_stochastic
        warm_start_iterations = 0
        if stochastic and self.warm_start:
            # The analytic pre-solve is deterministic and costs zero simulation calls,
            # so it is free and starts the expensive phase near the answer (spec 6.3).
            pre = Optimizer(
                stations, self.budget, tol=self.tol, initial_zeta=self.initial_zeta
            ).run()
            S = list(pre.capacities)
            warm_start_iterations = pre.iterations
        else:
            S = allocate(stations, self.budget, zeta)  # S^(1)

        degraded = []
        sim_calls = 0
        iterations = 0
        residual = math.inf
        floor = None
        stop_reason = "max_iter"
        evaluation = None

        for _ in range(self.max_iter):
            iterations += 1
            evaluation = self.analyzer.evaluate(stations, S)
            if stochastic:
                sim_calls += 1
            degraded.extend(evaluation.degraded)

            zeta = [
                st.zeta_from(T, Si)
                for st, T, Si in zip(stations, evaluation.sojourn_times, S)
            ]                                                    # eq 22
            S_target = allocate(stations, self.budget, zeta)      # eq 21

            floor = self._noise_floor(stations, S, zeta, evaluation.ci)
            if self.damping == 1.0:
                S_new = S_target       # explicit, so the analytic path adds no arithmetic
            else:
                theta = self.damping
                S_new = [
                    (1.0 - theta) * s + theta * t for s, t in zip(S, S_target)
                ]
            residual = max(abs(a - b) for a, b in zip(S_new, S))
            S = S_new

            # Convergence is tested in TARGET space, not on the damped iterate.
            #
            # `residual` is the damped step, theta * |S_target - S|, while both stopping
            # terms are naturally target-space quantities: `tol` is a tolerance on how far
            # `allocate` still wants to move, and `floor` is the spread noise induces in
            # `allocate`'s output. Comparing either against the damped step would scale it
            # by 1/theta -- `tol=1e-9` would mean 2e-9 and `noise_kappa=1.0` would mean 2.0
            # at the stochastic default theta=0.5.
            #
            # Normalizing the step once, rather than scaling each term, keeps both knobs
            # meaning exactly what they say at every damping value, and keeps `tol`
            # comparable across the analytic and simulated paths -- which is the premise
            # the whole feature rests on (spec 1.1). Division by 1.0 is exact in IEEE 754,
            # so this is bit-for-bit inert at theta = 1.0.
            step = residual / self.damping
            threshold = self.tol
            if floor is not None:
                threshold = max(self.tol, self.noise_kappa * floor)
            if step < threshold:
                # Label by where the step actually landed, not by which term won the
                # max(): a run that met `tol` outright is a "tol" stop even when the
                # noise floor was the larger threshold.
                stop_reason = "noise-floor" if step >= self.tol else "tol"
                break

        converged = stop_reason != "max_iter"
        if not converged:
            warnings.warn(
                f"Optimizer did not converge in {iterations} iterations "
                f"(max_iter={self.max_iter}, tol={self.tol}, final residual={residual:g}); "
                f"returned capacities are the last iterate and may be sub-optimal.",
                RuntimeWarning,
                stacklevel=2,
            )

        if stochastic:
            if self.final_evaluation or evaluation is None:
                # One more run at the converged S* with a fresh seed: those numbers are
                # the reported metrics, independent of the CRN sample path (spec 6.5).
                evaluation = self.analyzer.evaluate(stations, S, fresh_seed=True)
                sim_calls += 1
                degraded.extend(evaluation.degraded)
            # Otherwise the last loop iterate's numbers are reported as-is, which is what
            # final_evaluation=False asks for. They were measured at the pre-damping S.
        else:
            evaluation = self.analyzer.evaluate(stations, S)

        sojourn_times = list(evaluation.sojourn_times)
        zeta = [
            st.zeta_from(T, Si) for st, T, Si in zip(stations, sojourn_times, S)
        ]
        objective = sum(st.weight * T for st, T in zip(stations, sojourn_times))

        if self.strict and degraded:
            raise SimulationQualityError("; ".join(degraded))

        return Result(
            capacities=S,
            sojourn_times=sojourn_times,
            zeta=zeta,
            objective=objective,
            iterations=iterations,
            converged=converged,
            residual=residual,
            sojourn_ci=evaluation.ci,
            noise_floor=floor,
            stop_reason=stop_reason,
            warm_start_iterations=warm_start_iterations,
            degraded=degraded,
            system_response_time=evaluation.extras.get("system_response_time"),
            sim_calls=sim_calls,
        )

    def _noise_floor(self, stations, S, zeta, ci):
        """Propagate CI half-widths into zeta and measure the spread in S (spec 6.4).

        A station's `ci` entry is None when its response-time measure had no confidence
        interval (qsim/measures.py already warned about it): treat its half-width as 0,
        so it contributes no noise instead of crashing on the missing bounds. If every
        entry is None, every dzeta is 0 and the floor comes out 0.0 - correctly meaning
        "no noise information", not "no noise".
        """
        if ci is None or self.noise_kappa <= 0.0:
            return None
        dzeta = [
            0.0 if entry is None else st.zeta_from(0.5 * (entry[1] - entry[0]), Si)
            for st, Si, entry in zip(stations, S, ci)
        ]
        return noise_floor(stations, self.budget, zeta, dzeta)
