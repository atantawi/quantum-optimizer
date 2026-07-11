"""Fixed-point optimization loop (paper Steps 0-5)."""

import math
import warnings
from dataclasses import dataclass

from qopt.allocator import allocate, min_feasible_budget
from qopt.exceptions import InfeasibleBudgetError


@dataclass
class Result:
    """Outcome of an optimization run (lists aligned to the station order)."""

    capacities: list
    sojourn_times: list
    zeta: list
    objective: float
    iterations: int
    converged: bool
    residual: float  # final ||S_new - S||_inf; how close the last iterate came to tol


class Optimizer:
    """Drives the fixed-point iteration for the capacity allocation problem.

    Loop: allocate from an initial zeta guess, then repeatedly recompute zeta from the
    current capacities (eq 22) and re-allocate (eq 21) until ||S_new - S||_inf < tol or
    max_iter is reached.
    """

    def __init__(self, stations, budget, *, tol=1e-9, max_iter=1000, initial_zeta=None):
        self.stations = list(stations)
        self.budget = budget
        self.tol = tol
        self.max_iter = max_iter
        self.initial_zeta = initial_zeta

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

        S = allocate(stations, self.budget, zeta)  # S^(1)
        converged = False
        iterations = 0
        residual = math.inf
        for _ in range(self.max_iter):
            iterations += 1
            zeta = [st.zeta(Si) for st, Si in zip(stations, S)]  # eq 22
            S_new = allocate(stations, self.budget, zeta)        # eq 21
            residual = max(abs(a - b) for a, b in zip(S_new, S))
            S = S_new
            if residual < self.tol:
                converged = True
                break

        if not converged:
            warnings.warn(
                f"Optimizer did not converge in {iterations} iterations "
                f"(max_iter={self.max_iter}, tol={self.tol}, final residual={residual:g}); "
                f"returned capacities are the last iterate and may be sub-optimal.",
                RuntimeWarning,
                stacklevel=2,
            )

        zeta = [st.zeta(Si) for st, Si in zip(stations, S)]
        sojourn_times = [st.sojourn_time(Si) for st, Si in zip(stations, S)]
        objective = sum(
            st.weight * t for st, t in zip(stations, sojourn_times)
        )
        return Result(
            capacities=S,
            sojourn_times=sojourn_times,
            zeta=zeta,
            objective=objective,
            iterations=iterations,
            converged=converged,
            residual=residual,
        )
