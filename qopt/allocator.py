"""Closed-form capacity allocation (paper eq 21)."""

import math


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
    """
    return sum(st.min_spend for st in stations)


def allocate(stations, C, zeta_vec):
    """Optimal capacities for fixed zeta (paper eq 21).

        S_i = gamma_i/mu_i
            + (C - sum_j c_j gamma_j/mu_j) * sqrt(w_i zeta_i/(c_i mu_i)) / sum_j sqrt(w_j zeta_j c_j/mu_j)

    where c_i = station.alloc_cost, w_i = station.weight. Assumes C is feasible and every
    zeta_i > 0 (enforced by the Optimizer). Returns a list aligned to `stations`.
    """
    base = [st.gamma / st.mu for st in stations]
    slack = C - sum(st.alloc_cost * b for st, b in zip(stations, base))
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
