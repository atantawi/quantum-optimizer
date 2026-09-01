"""Closed-form capacity allocation (paper eq 21)."""

import math

from qopt.exceptions import InfeasibleBudgetError


def min_feasible_budget(stations):
    """Minimum budget to keep every station stable: sum_j alloc_cost_j * gamma_j / mu_j.

    A budget strictly greater than this makes eq 21's slack term positive, so every
    allocated capacity satisfies S_i * mu_i > gamma_i.

    Summed through `Station.min_spend` rather than inline, so that a station carrying a
    free policy parameter reports the floor at the ray a RUN starts from rather than at
    whatever ray it currently sits on. Without that this helper -- which is public, and
    which the README's normal usage scales budgets from -- disagreed with `Optimizer.run()`
    on a reused tuned station, and its answer depended on run history. The default
    `min_spend` is the plain expression, so nothing else moves.

    The two coincide for every station on its starting ray, which is every station
    `Optimizer.run()` ever allocates for, since it restores that ray first. They diverge
    only for a tuned fork-join still carrying a finished run's ray, and there `allocate`
    refuses rather than allocating unstably -- so `C > min_feasible_budget(stations)` can
    never silently produce an unstable capacity, whichever way the two are composed.
    """
    return sum(st.min_spend for st in stations)


def allocate(stations, C, zeta_vec):
    """Optimal capacities for fixed zeta (paper eq 21).

        S_i = gamma_i/mu_i
            + (C - sum_j c_j gamma_j/mu_j) * sqrt(w_i zeta_i/(c_i mu_i)) / sum_j sqrt(w_j zeta_j c_j/mu_j)

    where c_i = station.alloc_cost, w_i = station.weight. Every zeta_i must be > 0, which
    the Optimizer enforces. Returns a list aligned to `stations`.

    Raises InfeasibleBudgetError unless C exceeds the stations' floor AT THE RAYS THEY ARE
    CURRENTLY ON, which is what this function prices. Checked rather than assumed because
    eq 21 has no stability test of its own: a non-positive slack term SUBTRACTS from the
    base term gamma/mu, so every returned capacity silently lands below its stability
    boundary. That floor is normally `min_feasible_budget(stations)` exactly; it is higher
    only for a tuned fork-join left on a finished run's ray, and the message reports the
    number actually required so the two are never confused.
    """
    base = [st.gamma / st.mu for st in stations]
    floor = sum(st.alloc_cost * b for st, b in zip(stations, base))
    slack = C - floor
    if not slack > 0.0:      # `not >` rather than `<=`, so a NaN budget is rejected too
        raise InfeasibleBudgetError(
            f"budget {C} <= {floor}, the minimum these stations need at the rays they are "
            f"currently priced on"
        )
    denom = sum(
        math.sqrt(st.weight * z * st.alloc_cost / st.mu)
        for st, z in zip(stations, zeta_vec)
    )
    capacities = []
    for st, b, z in zip(stations, base, zeta_vec):
        num = math.sqrt(st.weight * z / (st.alloc_cost * st.mu))
        capacities.append(b + slack * num / denom)
    return capacities


ZETA_FLOOR = 1e-12
"""Smallest zeta handed to `allocate`, which takes its square root."""


def noise_floor(stations, C, zeta_vec, dzeta):
    """How much of a capacity change is attributable to evaluation noise (spec 6.4).

    `allocate` is closed-form and pure, so this costs zero simulation calls: propagate
    each reported CI half-width h_i into zeta as dzeta_i = h_i * (S_i*mu_i - gamma_i),
    then measure the spread in S that a perturbation of that size can produce.

    The perturbation is ANTI-CORRELATED, not uniform. Eq 21 is invariant under uniform
    positive scaling of zeta, so moving every station up together is nearly a no-op
    rather than a worst case. For each station i we evaluate `allocate` with component i
    up and all others down, plus the mirror:

        noise_floor = max_i |S_i(zeta+) - S_i(zeta-)| / 2

    That is 2n closed-form evaluations, negligible against one simulation run.
    """
    n = len(zeta_vec)
    if n == 0 or all(d == 0.0 for d in dzeta):
        return 0.0
    worst = 0.0
    for i in range(n):
        up = [
            max(zeta_vec[k] + dzeta[k] if k == i else zeta_vec[k] - dzeta[k], ZETA_FLOOR)
            for k in range(n)
        ]
        down = [
            max(zeta_vec[k] - dzeta[k] if k == i else zeta_vec[k] + dzeta[k], ZETA_FLOOR)
            for k in range(n)
        ]
        S_up = allocate(stations, C, up)
        S_down = allocate(stations, C, down)
        worst = max(worst, abs(S_up[i] - S_down[i]) / 2.0)
    return worst
