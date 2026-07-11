"""UL approximation for the heterogeneous 2-queue fork-join mean response time.

Lifted from the fork-join repo (forkjoin/analytical.py::mean_response_time):
a convex blend of the independent upper bound and the bottleneck lower bound,
  T_UL = (1 - alpha) * T_UB + alpha * T_bot,  alpha = (rho1 + rho2) / 8.
Exact for the homogeneous case (mu1 == mu2). Copied here to avoid a runtime
dependency on that repo.
"""

from qopt.exceptions import InstabilityError


def t_ul(lam, mu1, mu2):
    """Mean response time of a 2-queue fork-join system (UL interpolation).

    Args:
        lam: Poisson arrival rate to the fork-join station.
        mu1, mu2: effective service rates of the two servers.

    Requires stability: lam < min(mu1, mu2).
    """
    if lam >= mu1 or lam >= mu2:
        raise InstabilityError(
            f"fork-join unstable: need lam < min(mu1, mu2), "
            f"got lam={lam}, mu1={mu1}, mu2={mu2}"
        )
    rho1 = lam / mu1
    rho2 = lam / mu2
    alpha = (rho1 + rho2) / 8.0
    t_ub = 1.0 / (mu1 - lam) + 1.0 / (mu2 - lam) - 1.0 / (mu1 + mu2 - 2.0 * lam)
    t_bot = max(1.0 / (mu1 - lam), 1.0 / (mu2 - lam))
    return (1.0 - alpha) * t_ub + alpha * t_bot
