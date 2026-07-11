"""Closed-form capacity allocation (paper eq 21)."""

import math


def min_feasible_budget(stations):
    """Minimum budget to keep every station stable: sum_j alloc_cost_j * gamma_j / mu_j.

    A budget strictly greater than this makes eq 21's slack term positive, so every
    allocated capacity satisfies S_i * mu_i > gamma_i.
    """
    return sum(st.alloc_cost * st.gamma / st.mu for st in stations)


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
