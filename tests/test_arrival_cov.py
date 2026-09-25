"""Measured arrival cov_a: station arithmetic and optimizer plumbing.

Spec: docs/superpowers/specs/2026-09-25-measured-arrival-cov-design.md
"""

import math

import pytest

from qopt.station import ForkJoinStation, GG1Station
from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE


def _gg1(cov_a, mode=ZETA_SLOPE, cov_s=0.5):
    return GG1Station(0.6, 1.5, c=2.0, cov_a=cov_a, cov_s=cov_s, name="g", zeta_mode=mode)


S_GRID = [0.5, 0.8, 1.2, 3.0]          # gamma/mu = 0.4, so every S here is stable


# --- station arithmetic -------------------------------------------------------

@pytest.mark.parametrize("S", S_GRID)
@pytest.mark.parametrize("c", [0.0, 0.3, 1.0, 2.5])
def test_a_measured_cov_a_prices_exactly_like_a_station_built_with_it(S, c):
    # Exact, not approx: the override must run the SAME expression on the measured value,
    # so a station built with cov_a=c is its bit-for-bit reference.
    built, measured = _gg1(c), _gg1(1.7)
    assert measured.sojourn_time(S, cov_a=c) == built.sojourn_time(S)
    assert measured.dT_dS(S, cov_a=c) == built.dT_dS(S)
    assert measured.phi(S, cov_a=c) == built.phi(S)
    assert measured.zeta_from(0.9, S, cov_a=c) == built.zeta_from(0.9, S)


def test_the_override_moves_slope_zeta():
    # Anti-vacuity for the test above: if the override were ignored, 1.7 and 0.2 would
    # still agree there only by coincidence -- they must not.
    st = _gg1(1.7)
    for S in S_GRID:
        assert st.zeta_from(0.9, S, cov_a=0.2) != st.zeta_from(0.9, S)


def test_the_override_does_not_persist_on_the_station():
    st = _gg1(1.7)
    before = [st.phi(S) for S in S_GRID]
    for S in S_GRID:
        st.phi(S, cov_a=0.2)
        st.zeta_from(0.9, S, cov_a=0.2)
        st.sojourn_time(S, cov_a=0.2)
    assert st.cov_a == 1.7
    assert [st.phi(S) for S in S_GRID] == before


def test_the_level_arm_ignores_the_override():
    st = _gg1(1.7, mode=ZETA_LEVEL)
    for S in S_GRID:
        x = S * st.mu - st.gamma
        assert st.zeta_from(0.9, S, cov_a=0.2) == 0.9 * x


def test_a_measured_zero_cov_takes_the_deterministic_branch():
    # cov_s = 0 and a measured cov_a = 0 give k == 0 from the OVERRIDE, although the
    # constructor k is 0.5. phi must then be exactly the k == 0 value, 1 - rho.
    st = _gg1(1.0, cov_s=0.0)
    for S in S_GRID:
        rho = st.gamma / (S * st.mu)
        assert st.phi(S, cov_a=0.0) == pytest.approx(1.0 - rho, rel=1e-12)
        assert st.sojourn_time(S, cov_a=0.0) == 1.0 / (S * st.mu)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, -math.inf])
def test_an_invalid_override_is_rejected_like_the_constructor_argument(bad):
    st = _gg1(1.7)
    with pytest.raises(ValueError, match="cov_a must be a finite number >= 0"):
        st.phi(1.0, cov_a=bad)
    with pytest.raises(ValueError, match="cov_a must be a finite number >= 0"):
        st.sojourn_time(1.0, cov_a=bad)


def test_a_fork_join_station_ignores_the_override():
    fj = ForkJoinStation(0.5, 1.0, r=2.0, c1=1.0, c2=1.0, name="fj", zeta_mode=ZETA_SLOPE)
    for S in [0.8, 1.5, 3.0]:
        assert fj.phi(S, cov_a=0.3) == fj.phi(S)
        assert fj.zeta_from(0.9, S, cov_a=0.3) == fj.zeta_from(0.9, S)


def test_only_a_slope_station_that_reads_cov_a_uses_a_measurement():
    assert GG1Station.reads_arrival_cov is True
    assert ForkJoinStation.reads_arrival_cov is False
    assert _gg1(1.0).uses_measured_cov_a is True
    assert _gg1(1.0, mode=ZETA_LEVEL).uses_measured_cov_a is False
    fj = ForkJoinStation(0.5, 1.0, r=2.0, c1=1.0, c2=1.0, zeta_mode=ZETA_SLOPE)
    assert fj.uses_measured_cov_a is False


def test_zeta_from_does_not_pass_the_keyword_when_there_is_no_measurement():
    # A user subclass written before the keyword existed must keep working on every
    # path that never measures.
    class Legacy(GG1Station):
        def phi(self, S):
            return super().phi(S)

    st = Legacy(0.6, 1.5, c=2.0, cov_a=1.7, cov_s=0.5, zeta_mode=ZETA_SLOPE)
    assert st.zeta_from(0.9, 1.0) == _gg1(1.7).zeta_from(0.9, 1.0)
