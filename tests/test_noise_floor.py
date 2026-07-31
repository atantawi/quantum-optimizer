import pytest

from qopt.allocator import allocate, noise_floor
from qopt.station import ForkJoinStation, GG1Station


def _stations():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


C = 15.6
ZETA = [1.0, 0.9451279819531168, 1.1377787740190126]


def test_allocate_is_invariant_under_uniform_zeta_scaling():
    # The property section 6.4 depends on: uniform perturbation is NOT the worst case.
    base = allocate(_stations(), C, ZETA)
    for k in (0.5, 2.0, 7.3, 100.0):
        scaled = allocate(_stations(), C, [k * z for z in ZETA])
        assert scaled == pytest.approx(base, rel=1e-12)


def test_zero_perturbation_gives_a_zero_floor():
    assert noise_floor(_stations(), C, ZETA, [0.0, 0.0, 0.0]) == 0.0


def test_floor_grows_with_the_perturbation():
    stations = _stations()
    small = noise_floor(stations, C, ZETA, [0.01 * z for z in ZETA])
    large = noise_floor(stations, C, ZETA, [0.10 * z for z in ZETA])
    assert 0.0 < small < large


def test_floor_is_positive_for_a_realistic_ci_width():
    # 1% CI half-width on zeta at the mixed network's converged point.
    floor = noise_floor(_stations(), C, ZETA, [0.01 * z for z in ZETA])
    assert floor == pytest.approx(0.024349965940745344, rel=1e-9)


def test_anti_correlated_beats_uniform_perturbation():
    stations = _stations()
    dzeta = [0.10 * z for z in ZETA]
    anti = noise_floor(stations, C, ZETA, dzeta)
    up = allocate(stations, C, [z + d for z, d in zip(ZETA, dzeta)])
    down = allocate(stations, C, [z - d for z, d in zip(ZETA, dzeta)])
    uniform = max(abs(a - b) / 2.0 for a, b in zip(up, down))
    assert anti > 10 * uniform


def test_huge_perturbation_is_clamped_not_crashed():
    # zeta - dzeta goes negative; allocate would take sqrt of it unclamped.
    floor = noise_floor(_stations(), C, ZETA, [10.0 * z for z in ZETA])
    assert floor > 0.0


def test_empty_station_list_is_zero():
    assert noise_floor([], C, [], []) == 0.0
