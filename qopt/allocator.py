"""Closed-form capacity allocation (paper eq 21)."""

import math

from qopt.exceptions import InfeasibleBudgetError


def min_feasible_budget(stations):
    """Minimum budget to keep every station stable: sum_j alloc_cost_j * gamma_j / mu_j.

    "Stable" is `S*mu > gamma`, except for a deterministic G/G/1, which is priceable AT
    `S*mu == gamma` and so is stable at exactly its own term in this sum -- see
    `Station.admits_full_utilization`. `allocate` still requires a budget strictly above this
    total, because the slack it distributes has to be positive for anything else to move; a
    network of nothing but deterministic stations is therefore the one case where this floor
    is feasible in the model but still refused by eq 21.

    A budget strictly greater than this makes eq 21's slack term positive. That is an
    AGGREGATE statement and does not promise each station a margin: eq 21 distributes the
    slack, so a station's share can round away entirely and leave `S_i * mu_i == gamma_i`
    exactly -- measured on plain M/M/1 stations, not only on the fork-join policies.

    What decides that is the ratio of a station's SHARE to its own base `gamma_i/mu_i`, not
    the budget's distance from this floor: the share is added to the base, so it vanishes
    once it falls below half an ulp of it. A lopsided weight vector is therefore enough on
    its own. An earlier version of this note put the effect "within a few ulps of the
    floor", which understates it: budget proximity and weight skew are two independent ways
    in. It then overcorrected, reporting a grid measured entirely on cov = 0 stations as eq
    21's rounding. Two mechanisms are in play and only the first is:

      * Eq 21's rounding, which is calibration-independent. Measured on two stations at
        gamma/mu = 0.5 with cov = 1, where the two calibrations are bit-identical because
        phi == 1: the light station rounds onto its boundary at a weight ratio of 1e30 and a
        budget 1.01x this floor, under BOTH modes, and survives from 2x upwards (x = 2.0e-15
        there). At a ratio of 1e20 it survives every multiple from 1.01x up.
      * The k = 0 zeta degeneracy, which is specific to deterministic stations, reaches much
        further, and is NOT a failure. For any k = (cov_a^2+cov_s^2)/2 > 0, zeta tends to
        `k*rho` as the spare capacity x tends to 0 -- 1.0e-02 at k = 0.01 -- so it stays
        bounded away from zero and the share cannot round away for budget reasons alone. At
        k = 0 the congestion term is absent and zeta tends to 0 instead: as x under level, and
        as x^2 under slope. Shares go as sqrt(zeta), so slope's share map is linear in x and
        contracts onto the station's own `gamma/mu`. A cov = 0 light station therefore lands
        there under slope at a ratio of 1e20 at 1.01x, 2x, 100x AND 10000x this floor, where
        level needs 1e30.

        That outcome is CORRECT rather than tolerated: `E[T] = 1/(S*mu)` is finite at rho == 1,
        so `gamma/mu` is a capacity such a station can be priced at, and it is frequently the
        optimal one -- on the reference network of
        test_a_deterministic_station_may_sit_exactly_on_its_boundary the objective's infimum
        lies exactly there and slope calibration attains it, while level stops 2.5e-09 short
        and pays up to 1.9%. `Station.admits_full_utilization` is what makes the point
        reachable, and it is the only station that gets it.

    The first is pinned by
    test_an_extreme_weight_ratio_raises_rather_than_returning_a_boundary_capacity, which keeps
    its cov = 1 arms separate from the k = 0 case for exactly this reason, and the second by
    test_a_deterministic_station_may_sit_exactly_on_its_boundary.

    What happens next depends on how `gamma_i/mu_i` itself rounds, and only one of the two
    outcomes is loud. Where `b * mu == gamma` exactly -- gamma=0.5, mu=1.0 -- the station is
    unstable and `check_stable`/`sojourn_time` refuse, so the caller gets an error naming the
    station. Where `b * mu` rounds a hair ABOVE gamma the station is technically stable:
    gamma=0.1, mu=0.39 leaves x = 1.4e-17, and E[T] comes back as a finite 9.0e+16 with
    phi = 0.8, with no error anywhere. That second case is a wrong number, not an error, and
    this note previously claimed otherwise. Both are pinned, the silent one deliberately, by
    test_an_extreme_weight_ratio_raises_rather_than_returning_a_boundary_capacity and
    test_a_collapsed_share_can_land_on_a_technically_stable_capacity.

    Eq 21 is nevertheless left to round as it rounds, and NOT nudged up to the next
    representable capacity, for three independent reasons: a nudge spends budget the caller
    did not allocate, it breaks the bit-exactness the level path is pinned to, and at 1.4e-17
    of spare capacity the sojourn time is meaningless whichever neighbouring float is used.
    Making the collapse itself raise is also ruled out, and by an existing invariant rather
    than by taste: test_the_reported_floor_is_bit_for_bit_the_one_allocate_prices requires
    the smallest representable budget ABOVE this floor to allocate, and that budget collapses
    every share by construction. The input is what is wrong. Scale a budget off this floor
    and keep weights within a few orders of magnitude of each other; do not sit on either
    edge.

    Summed through `Station.min_spend` rather than inline, so that a station carrying a
    free policy parameter reports the floor at the ray a RUN starts from rather than at
    whatever ray it currently sits on. Without that this helper -- which is public, and
    which the README's normal usage scales budgets from -- disagreed with `Optimizer.run()`
    on a reused tuned station, and its answer depended on run history. The default
    `min_spend` is the plain expression, so nothing else moves.

    The two coincide for every station on its starting ray, which is every station
    `Optimizer.run()` ever allocates for, since it restores that ray first. They diverge
    only for a tuned fork-join still carrying a finished run's ray, and there `allocate`
    refuses rather than allocating unstably -- so ray drift cannot turn a budget above this
    floor into a silently unstable capacity, whichever way the two are composed. The
    ulp-scale caveat above is separate, older, and applies to every station type.
    """
    return sum(st.min_spend for st in stations)


def allocate(stations, C, zeta_vec):
    """Optimal capacities for fixed zeta (paper eq 21).

        S_i = gamma_i/mu_i
            + (C - sum_j c_j gamma_j/mu_j) * sqrt(w_i zeta_i/(c_i mu_i)) / sum_j sqrt(w_j zeta_j c_j/mu_j)

    where c_i = station.alloc_cost, w_i = station.weight. Returns a list aligned to
    `stations`.

    Validates `zeta_vec` rather than assuming the Optimizer has: this is root-exported, and
    an unchecked zeta was the other half of the feasibility hole the ray guards closed. A
    SHORT vector was the worst of them, because `zip` truncates in silence -- the result
    came back with fewer capacities than there were stations, the budget renormalized
    across the survivors. A zero left its own station at exactly `S*mu == gamma` with the
    budget far above the floor, and a NaN propagated into every capacity.

    Raises InfeasibleBudgetError unless C exceeds the stations' floor AT THE RAYS THEY ARE
    CURRENTLY ON, which is what this function prices. Checked rather than assumed because
    eq 21 has no stability test of its own: a non-positive slack term SUBTRACTS from the
    base term gamma/mu, so every returned capacity silently lands below its stability
    boundary. That floor is normally `min_feasible_budget(stations)` exactly; it is higher
    only for a tuned fork-join left on a finished run's ray, and the message reports the
    number actually required so the two are never confused.
    """
    zeta_vec = list(zeta_vec)
    if len(zeta_vec) != len(stations):
        raise ValueError(
            f"zeta_vec length {len(zeta_vec)} must match the {len(stations)} stations"
        )
    for st, z in zip(stations, zeta_vec):
        # Zero is admissible from exactly one kind of station: one whose E[T] is finite at
        # rho == 1. Its share is then zero, leaving it at `gamma/mu`, which for that station
        # is a priceable capacity and generally the optimal one -- see
        # Station.admits_full_utilization. For every other station a zero zeta means a zero
        # share on a DIVERGENT E[T], which is the silent-instability hole this check closed,
        # so the strict test still applies to them.
        usable = math.isfinite(z) and (
            z >= 0.0 if st.admits_full_utilization else z > 0.0
        )
        if not usable:
            raise ValueError(
                f"zeta values must be finite and strictly positive, got {zeta_vec} -- "
                f"zero is accepted only from a station whose E[T] is finite at rho == 1"
            )
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
    # Every other factor in that sum is validated positive -- weight and mu by the Station
    # constructor, alloc_cost by each subclass -- so in exact arithmetic the sum is zero
    # exactly when EVERY zeta is zero, which the per-station loop above now lets through for a
    # network of nothing but full-utilization stations. Eq 21 has no answer there and neither
    # does its limit: the formula is invariant under uniform positive scaling of zeta, so
    # zeta = (eps, eps) tends to shares proportional to sqrt(w/(c*mu)), while (eps, eps^2)
    # tends to giving the first station everything. The limit is path-dependent, so there is
    # no value to return and this raises instead of picking one.
    #
    # In floating point one more input reaches the same place: every zeta positive but every
    # `w*zeta*c/mu` UNDERFLOWING to zero, which a denormal zeta does as soon as `w*c/mu < 1`
    # -- zeta = 5e-324 with w = 1e-300, c = 1e-10 and mu = 1.0 measured. That is not a
    # degenerate limit, it is a lost product, so the message names the denominator and lists
    # both causes rather than asserting the zetas are zero. Both arms are pinned by
    # test_allocate_rejects_an_all_zero_zeta_vector.
    #
    # Not reachable from `Optimizer.run()` on anything we can construct, though the argument
    # is a floating-point one and not the exact-arithmetic `sum_i c_i*x_i == slack > 0`: the
    # shares are what round away, so what matters is whether EVERY station can lose its own.
    # Collapsing station i needs its share below `ulp(base_i)/2 <= base_i * 2^-53`, so all of
    # them collapsing needs `slack <= floor * 2^-53`, while the smallest representable budget
    # above the floor gives `slack >= ulp(floor) > floor * 2^-53`. The two cannot hold at
    # once, and the margin is a factor of about 2 rather than orders of magnitude: measured
    # at n = 2, 4, 8 and 16 identical deterministic stations with `base = 1e16` and the
    # minimum representable slack, every station keeps `x = 2.0` against a collapse threshold
    # of 1.0. The summation itself rounds, so this is an argument plus a measurement rather
    # than a proof. Checked anyway -- this function is root-exported and a caller can hand it
    # any vector at all.
    if not denom > 0.0:
        raise ValueError(
            f"eq 21's share denominator is {denom}, got zeta {zeta_vec} -- either every zeta "
            f"is zero, where eq 21 has no allocation because its limit depends on the path "
            f"the zetas take to zero, or every w*zeta*c/mu underflowed. At least one station "
            f"must carry a zeta large enough to survive that product."
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
    if len(dzeta) != n:
        raise ValueError(
            f"dzeta length {len(dzeta)} must match the {n} zeta values"
        )
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
