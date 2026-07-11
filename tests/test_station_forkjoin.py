import pytest

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul
from qopt.station import ForkJoinStation, Station


def test_alloc_cost_is_sum_of_both_servers():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=3.0)
    assert st.alloc_cost == pytest.approx(4.0)
    assert st.default_zeta == pytest.approx(1.5)
    assert isinstance(st, Station)


def test_sojourn_uses_t_ul_with_both_effective_rates():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    # m1 = S*mu = 1.0 (slower), m2 = S*r*mu = 2.0 (faster)
    assert st.sojourn_time(1.0) == pytest.approx(t_ul(0.6, 1.0, 2.0), rel=1e-12)


def test_homogeneous_forkjoin_matches_nelson_tantawi():
    st = ForkJoinStation(gamma=0.5, mu=1.0, r=1.0, c1=1.0, c2=1.0)
    rho = 0.5
    expected = (12 - rho) / (8 * (1.0 - 0.5))  # 2.875
    assert st.sojourn_time(1.0) == pytest.approx(expected, rel=1e-12)


def test_zeta_uses_slower_server_rate():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    expected = st.sojourn_time(1.0) * (1.0 * 1.0 - 0.6)
    assert st.zeta(1.0) == pytest.approx(expected, rel=1e-12)


def test_unstable_raises_on_slower_server():
    st = ForkJoinStation(gamma=1.0, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    with pytest.raises(InstabilityError):
        st.sojourn_time(1.0)  # S*mu = 1.0 == gamma (slower server binds)


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0.6, mu=1.0, r=0.9, c1=1.0, c2=1.0),   # r < 1
    dict(gamma=0.6, mu=1.0, r=2.0, c1=0.0, c2=1.0),   # c1 <= 0
    dict(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=-1.0),  # c2 <= 0
])
def test_forkjoin_validation(kwargs):
    with pytest.raises(ValueError):
        ForkJoinStation(**kwargs)
