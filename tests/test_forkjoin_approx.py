import math

import pytest

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul


def test_homogeneous_matches_nelson_tantawi():
    # For mu1 == mu2, T_UL reduces to (12 - rho) / (8 (mu - lam)).
    lam, mu = 0.5, 1.0
    rho = lam / mu
    expected = (12 - rho) / (8 * (mu - lam))  # = 2.875
    assert t_ul(lam, mu, mu) == pytest.approx(expected, rel=1e-12)


def test_heterogeneous_known_value():
    # Cross-checked against the fork-join repo's mean_response_time (doc table row
    # mu=1.0, mu=2.0, lam=0.6 -> ~2.641).
    assert t_ul(0.6, 2.0, 1.0) == pytest.approx(2.640873, rel=1e-5)


def test_symmetric_in_rates():
    assert t_ul(0.6, 2.0, 1.0) == pytest.approx(t_ul(0.6, 1.0, 2.0), rel=1e-12)


def test_unstable_raises():
    with pytest.raises(InstabilityError):
        t_ul(1.0, 1.0, 2.0)  # lam >= min(mu1, mu2)
