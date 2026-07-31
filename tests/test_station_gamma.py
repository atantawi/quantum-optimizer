import pytest

from qopt.exceptions import QOptError, SimulationError, TopologyError
from qopt.station import ForkJoinStation, GG1Station


def test_new_exceptions_are_qopt_errors():
    from qopt.exceptions import (
        MeasureMissingError,
        SimulationEngineError,
        SimulationQualityError,
        SimulationRequestError,
        SimulationTransportError,
    )

    assert issubclass(TopologyError, QOptError)
    assert issubclass(SimulationError, QOptError)
    for cls in (
        SimulationTransportError,
        SimulationRequestError,
        SimulationEngineError,
        SimulationQualityError,
        MeasureMissingError,
    ):
        assert issubclass(cls, SimulationError)


def test_explicit_gamma_still_works():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    assert st.gamma == 0.6


def test_mu_is_required():
    with pytest.raises(ValueError, match="mu is required"):
        GG1Station.mm1(gamma=0.6, c=2.0)


def test_gamma_omitted_raises_on_first_use():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="unbound")
    with pytest.raises(ValueError, match="no gamma"):
        st.gamma
    with pytest.raises(ValueError, match="no gamma"):
        st.sojourn_time(2.0)


def test_bind_gamma_fills_it():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    st.bind_gamma(0.6)
    assert st.gamma == 0.6
    assert st.sojourn_time(2.0) == pytest.approx(1.0 / (2.0 - 0.6), rel=1e-12)


def test_bind_gamma_is_idempotent_for_the_same_value():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    st.bind_gamma(0.6)
    st.bind_gamma(0.6)
    assert st.gamma == 0.6


def test_bind_gamma_rejects_a_conflicting_value():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    st.bind_gamma(0.6)
    with pytest.raises(ValueError, match="cannot rebind"):
        st.bind_gamma(0.7)


def test_bind_gamma_rejects_an_explicitly_constructed_gamma():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="a")
    with pytest.raises(ValueError, match="explicit gamma"):
        st.bind_gamma(0.6)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bind_gamma_validates_the_value(bad):
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    with pytest.raises(ValueError):
        st.bind_gamma(bad)


def test_check_stable_is_public_and_uses_the_same_guard():
    from qopt.exceptions import InstabilityError

    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="a")
    st.check_stable(1.0)                      # S*mu = 1.0 > 0.6, no raise
    with pytest.raises(InstabilityError, match="unstable"):
        st.check_stable(0.6)                  # S*mu = 0.6 == gamma


def test_check_stable_requires_a_bound_gamma():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="unbound")
    with pytest.raises(ValueError, match="no gamma"):
        st.check_stable(2.0)


def test_zeta_from_accepts_an_external_sojourn_time():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    # zeta = T * (S*mu - gamma) = 2.5 * (1.0 - 0.6)
    assert st.zeta_from(2.5, 1.0) == pytest.approx(1.0, rel=1e-12)


def test_zeta_delegates_to_zeta_from_bitwise():
    for st in (
        GG1Station.md1(gamma=0.6, mu=1.0, c=1.0),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),
    ):
        for S in (1.5, 2.0, 4.0):
            assert st.zeta(S) == st.zeta_from(st.sojourn_time(S), S)


def test_forkjoin_and_md1_accept_omitted_gamma():
    fj = ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj")
    md = GG1Station.md1(mu=1.0, c=1.0, name="md")
    fj.bind_gamma(0.5)
    md.bind_gamma(0.4)
    assert fj.gamma == 0.5 and md.gamma == 0.4
