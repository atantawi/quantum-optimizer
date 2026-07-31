import pytest

from qopt.exceptions import MeasureMissingError
from qopt.qsim.measures import SYSTEM_STATION, extract
from qopt.station import ForkJoinStation, GG1Station


def _stations():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


def test_system_station_key_is_the_empty_string():
    # Inferred from referenceNode="" (spec 5.3 gotcha 2), not pinned by a fixture.
    assert SYSTEM_STATION == ""


def test_extract_returns_sojourn_times_in_station_order(sim_response):
    stations = _stations()
    response = sim_response(
        sojourn={"fj": 0.29, "mm1": 0.42},        # deliberately out of station order
        throughput={"mm1": 0.6, "fj": 0.5},
        system=1.15,
    )
    T, ci, degraded, extras = extract(response, stations, "jobs")
    assert T == [0.42, 0.29]
    # Compared pairwise via pytest.approx (rather than one list == [...] literal):
    # 0.29 - 0.01 is 0.27999999999999997 in float64, one ULP off the 0.28 literal, and
    # pytest.approx in this pytest version does not support nested tuples-in-a-list.
    assert len(ci) == 2
    assert ci[0] == pytest.approx((0.41, 0.43))
    assert ci[1] == pytest.approx((0.28, 0.30))
    assert degraded == []
    assert extras["system_response_time"] == (1.15, (1.14, 1.16))
    assert extras["throughput"] == {"mm1": (0.6, (0.59, 0.61)),
                                    "fj": (0.5, (0.49, 0.51))}


def test_missing_station_response_time_is_a_hard_error(sim_response):
    stations = _stations()
    response = sim_response(sojourn={"mm1": 0.42}, throughput={"mm1": 0.6}, system=1.15)
    with pytest.raises(MeasureMissingError, match="'fj'"):
        extract(response, stations, "jobs")


def test_null_mean_counts_as_missing(sim_response):
    stations = _stations()
    response = sim_response(sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6})
    for m in response["measures"]:
        if m["station"] == "fj" and m["type"] == "response-time":
            m["mean"] = None
    with pytest.raises(MeasureMissingError):
        extract(response, stations, "jobs")


def test_missing_system_response_time_warns_and_records(sim_response):
    stations = _stations()
    response = sim_response(sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6})
    with pytest.warns(RuntimeWarning, match="system-response-time"):
        T, ci, degraded, extras = extract(response, stations, "jobs")
    assert extras["system_response_time"] is None
    assert any("system-response-time" in d for d in degraded)
    assert T == [0.42, 0.29]        # the run is still usable


def test_missing_throughput_for_a_checked_station_warns(sim_response):
    stations = _stations()
    response = sim_response(sojourn={"mm1": 0.42, "fj": 0.29}, system=1.15)
    with pytest.warns(RuntimeWarning, match="no 'throughput' for station 'mm1'"):
        _, _, degraded, extras = extract(response, stations, "jobs")
    assert "mm1" not in extras["throughput"]
    assert any("cannot run" in d for d in degraded)


def test_missing_throughput_for_an_exempt_station_is_silent(sim_response, recwarn):
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6}, system=1.15
    )
    _, _, degraded, extras = extract(response, stations, "jobs")
    assert "fj" not in extras["throughput"]
    assert degraded == []
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_completed_false_warns_and_records(sim_response):
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6}, system=1.15,
        completed=False,
    )
    with pytest.warns(RuntimeWarning, match="completed=false"):
        _, _, degraded, _ = extract(response, stations, "jobs")
    assert any("completed=false" in d for d in degraded)


def test_per_measure_success_false_warns_and_records(sim_response):
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6}, system=1.15,
        success=False,
    )
    with pytest.warns(RuntimeWarning, match="success=false"):
        T, _, degraded, _ = extract(response, stations, "jobs")
    assert T == [0.42, 0.29]                      # the mean is used anyway
    assert any("success=false" in d for d in degraded)


def test_wrong_job_class_is_treated_as_missing(sim_response):
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6}, job_class="web"
    )
    with pytest.raises(MeasureMissingError):
        extract(response, stations, "jobs")
