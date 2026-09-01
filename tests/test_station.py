import pytest

from qopt.exceptions import InstabilityError
from qopt.station import GG1Station, SingleServerStation, Station


def test_gg1_cannot_be_singleserver_instantiated():
    # SingleServerStation is abstract (no sojourn_time).
    with pytest.raises(TypeError):
        SingleServerStation(gamma=0.5, mu=1.0, c=1.0)  # type: ignore[abstract]


def test_mm1_sojourn_and_zeta():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    # mu_eff = S*mu = 1.0; E[T] = 1/(1 - 0.6) = 2.5; zeta = 2.5 * 0.4 = 1.0
    assert st.sojourn_time(1.0) == pytest.approx(2.5, rel=1e-12)
    assert st.zeta(1.0) == pytest.approx(1.0, rel=1e-12)
    assert st.alloc_cost == pytest.approx(2.0)
    assert st.default_zeta == pytest.approx(1.0)
    assert isinstance(st, Station)


def test_mm1_zeta_is_one_across_capacities():
    st = GG1Station.mm1(gamma=0.4, mu=1.0, c=1.0)
    for S in (0.6, 1.0, 2.5, 10.0):
        assert st.zeta(S) == pytest.approx(1.0, rel=1e-12)


def test_md1_sojourn_and_zeta():
    st = GG1Station.md1(gamma=0.6, mu=1.0, c=1.0)
    # rho = 0.6; E[T] = 1 * (1 + 0.5 * 0.6/0.4) = 1.75; zeta = 1.75 * 0.4 = 0.7 = 1 - rho/2
    assert st.sojourn_time(1.0) == pytest.approx(1.75, rel=1e-12)
    assert st.zeta(1.0) == pytest.approx(0.7, rel=1e-12)


def test_md1_zeta_is_load_dependent():
    st = GG1Station.md1(gamma=0.6, mu=1.0, c=1.0)
    for S in (1.0, 2.0, 4.0):
        rho = 0.6 / (S * 1.0)
        assert st.zeta(S) == pytest.approx(1 - rho / 2, rel=1e-12)


def test_gg1_general_cov_sojourn_and_zeta():
    # General G/G/1 with cov_a != 1 and cov_s not in {0, 1}, so the (cov_a^2 + cov_s^2)/2
    # Kingman coefficient is exercised at a value the mm1/md1 presets never reach.
    st = GG1Station(gamma=0.5, mu=1.0, c=1.0, cov_a=1.5, cov_s=0.5)
    # mu_eff = 1.0; rho = 0.5; k = (1.5^2 + 0.5^2)/2 = 1.25
    # E[T] = 1 * (1 + 1.25 * 0.5/0.5) = 2.25; zeta = 2.25 * (1.0 - 0.5) = 1.125
    assert st.sojourn_time(1.0) == pytest.approx(2.25, rel=1e-12)
    assert st.zeta(1.0) == pytest.approx(1.125, rel=1e-12)


def test_sojourn_time_unstable_raises():
    st = GG1Station.mm1(gamma=1.0, mu=1.0, c=1.0)
    with pytest.raises(InstabilityError):
        st.sojourn_time(1.0)  # S*mu = 1.0 == gamma


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0.0, mu=1.0, c=1.0),
    dict(gamma=0.5, mu=0.0, c=1.0),
    dict(gamma=0.5, mu=1.0, c=0.0),
    dict(gamma=0.5, mu=1.0, weight=0.0, c=1.0),
    dict(gamma=float("nan"), mu=1.0, c=1.0),   # NaN slips past `<= 0`; isfinite must catch it
    dict(gamma=0.5, mu=float("inf"), c=1.0),
])
def test_construction_validation(kwargs):
    with pytest.raises(ValueError):
        GG1Station.mm1(**kwargs)


@pytest.mark.parametrize("cov_a,cov_s", [(-1.0, 1.0), (1.0, -1.0), (float("nan"), 1.0)])
def test_invalid_cov_rejected(cov_a, cov_s):
    with pytest.raises(ValueError):
        GG1Station(gamma=0.5, mu=1.0, c=1.0, cov_a=cov_a, cov_s=cov_s)


def test_retune_is_a_no_op_for_a_station_with_no_free_policy_parameter():
    """The hook exists on the base class so the Optimizer can call it unconditionally.
    A single-server station has nothing to reprice, so it must return S untouched and
    leave its own coefficients alone."""
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="q")
    before = (st.mu, st.alloc_cost)
    for S in (1.0, 2.5, 1e6):
        assert st.retune(S) is S
    assert (st.mu, st.alloc_cost) == before
