import math
import warnings

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
    # Verified against a live qsim-service (spec 5.3 gotcha 2; module docstring above).
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


def test_missing_response_time_bounds_are_treated_as_a_miss(sim_response):
    # Mirrors test_qsim_analyzer.py::test_missing_throughput_bounds_are_treated_as_a_miss,
    # but for the measure eq 22 actually needs: a mean without a CI must not raise, only
    # degrade (finding 1) — that CI feeds the noise floor (optimizer.py's _noise_floor),
    # not eq 22 itself.
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6, "fj": 0.5}, system=1.15
    )
    for m in response["measures"]:
        if m["station"] == "mm1" and m["type"] == "response-time":
            m["lower"] = None
            m["upper"] = None
    with pytest.warns(RuntimeWarning, match="mm1"):
        T, ci, degraded, extras = extract(response, stations, "jobs")
    assert T == [0.42, 0.29]                      # the mean is still usable for eq 22
    assert ci[0] is None
    assert ci[1] == pytest.approx((0.28, 0.30))
    assert any("mm1" in d for d in degraded)


def test_missing_system_response_time_warns_and_records(sim_response):
    stations = _stations()
    response = sim_response(sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6})
    with pytest.warns(RuntimeWarning, match="system-response-time"):
        T, ci, degraded, extras = extract(response, stations, "jobs")
    assert extras["system_response_time"] is None
    assert any("system-response-time" in d for d in degraded)
    assert T == [0.42, 0.29]        # the run is still usable


def test_null_system_response_time_mean_counts_as_missing(sim_response):
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6}, system=1.15
    )
    for m in response["measures"]:
        if m["station"] == "" and m["type"] == "system-response-time":
            m["mean"] = None
    with pytest.warns(RuntimeWarning, match="system-response-time"):
        T, ci, degraded, extras = extract(response, stations, "jobs")
    assert extras["system_response_time"] is None
    assert any("system-response-time" in d for d in degraded)
    assert T == [0.42, 0.29]        # the run is still usable


def test_missing_system_response_time_bounds_are_treated_as_a_miss(sim_response):
    # The third site of the finding-1 pattern: a mean present, its CI absent. Guarded for
    # a station response-time above and for throughput in qsim/analyzer.py, so the
    # system-level diagnostic must warn and record too rather than pass Nones through
    # silently — a bare TypeError in a caller that formats the bounds is not the error
    # contract spec 7.1 defines for a missing diagnostic.
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6, "fj": 0.5}, system=1.15
    )
    for m in response["measures"]:
        if m["station"] == "" and m["type"] == "system-response-time":
            m["lower"] = None
            m["upper"] = None
    with pytest.warns(RuntimeWarning, match="system-response-time"):
        T, ci, degraded, extras = extract(response, stations, "jobs")
    mean, bounds = extras["system_response_time"]
    assert mean == 1.15                           # the mean is still reportable
    assert bounds == (None, None)
    assert any("system-response-time" in d for d in degraded)
    assert T == [0.42, 0.29]                      # the run is still usable
    assert ci[0] == pytest.approx((0.41, 0.43))   # station CIs untouched


def test_missing_throughput_for_a_checked_station_warns(sim_response):
    stations = _stations()
    response = sim_response(sojourn={"mm1": 0.42, "fj": 0.29}, system=1.15)
    with pytest.warns(RuntimeWarning, match="no 'throughput' for station 'mm1'"):
        _, _, degraded, extras = extract(response, stations, "jobs")
    assert "mm1" not in extras["throughput"]
    assert any("cannot run" in d for d in degraded)


def test_null_throughput_mean_counts_as_missing(sim_response):
    stations = _stations()
    response = sim_response(
        sojourn={"mm1": 0.42, "fj": 0.29}, throughput={"mm1": 0.6}, system=1.15
    )
    for m in response["measures"]:
        if m["station"] == "mm1" and m["type"] == "throughput":
            m["mean"] = None
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


# --- measured arrival cov_a (spec 2026-09-25 §3.2) ------------------------------

from qopt.analyzer import Evaluation
from qopt.qsim.measures import extract_arrival_cov
from qopt.qsim.spec import COV_A_MEASURE
from qopt.station import ForkJoinStation, GG1Station
from qopt.zeta import ZETA_SLOPE


def _ia(station, mean, variance, *, success=True, job_class="jobs"):
    """An interarrival-time entry in the shape qsim-service's SolutionsParser emits at
    8e7358b: lower/upper null (JMT's CI is on the rate), variance and stdDev populated."""
    std = variance ** 0.5 if isinstance(variance, float) and variance >= 0 else None
    return {
        "station": station, "class": job_class, "type": COV_A_MEASURE,
        "mean": mean, "lower": None, "upper": None, "alpha": 0.05, "precision": 0.02,
        "success": success, "samplesAnalyzed": 40000, "samplesDiscarded": 1000,
        "variance": variance, "stdDev": std,
    }


def _cov_stations():
    return [
        GG1Station.md1(0.4, 1.0, c=1.0, name="md1", zeta_mode=ZETA_SLOPE),
        GG1Station.mm1(0.6, 1.0, c=2.0, name="mm1"),                  # level
        ForkJoinStation(0.5, 1.0, r=2.0, c1=1.0, c2=1.0, name="fj", zeta_mode=ZETA_SLOPE),
    ]


def _cov_response(*entries):
    return {"completed": True, "measures": list(entries)}


def test_evaluation_arrival_cov_defaults_to_none():
    assert Evaluation(sojourn_times=[1.0]).arrival_cov is None


def test_cov_a_is_the_square_root_of_variance_over_mean_squared():
    response = _cov_response(_ia("md1", 2.0, 1.0), _ia("mm1", 1.0, 1.0), _ia("fj", 1.0, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # the missing CI must NOT warn
        cov, degraded = extract_arrival_cov(response, _cov_stations(), "jobs")
    assert cov == [0.5, None, None]             # level mm1 and fork-join: silently None
    assert degraded == []


def test_a_zero_scv_is_deterministic_arrivals_not_a_failure():
    cov, degraded = extract_arrival_cov(
        _cov_response(_ia("md1", 2.0, 0.0)), _cov_stations(), "jobs")
    assert cov[0] == 0.0
    assert degraded == []


def test_a_station_that_uses_no_measurement_is_not_flagged_when_it_has_none():
    cov, degraded = extract_arrival_cov(
        _cov_response(_ia("md1", 2.0, 1.0)), _cov_stations(), "jobs")
    assert cov == [0.5, None, None]
    assert degraded == []


@pytest.mark.parametrize("entry", [
    None,                                     # measure absent
    _ia("md1", 2.0, 1.0, job_class="other"),  # present, but for another class
    _ia("md1", None, 1.0),
    _ia("md1", 0.0, 1.0),
    _ia("md1", -1.0, 1.0),
    _ia("md1", math.nan, 1.0),
    _ia("md1", math.inf, 1.0),                # variance/inf/inf would be a bogus 0.0
    _ia("md1", 2.0, None),
    _ia("md1", 2.0, -0.1),
    _ia("md1", 2.0, math.nan),
    _ia("md1", 2.0, math.inf),
    _ia("md1", 1e-200, 1.0),                  # mean**2 underflows; must not divide by 0
], ids=["absent", "other-class", "mean-none", "mean-zero", "mean-neg", "mean-nan",
        "mean-inf", "var-none", "var-neg", "var-nan", "var-inf", "mean-tiny"])
def test_an_unusable_measurement_falls_back_and_is_recorded(entry):
    response = _cov_response(*([] if entry is None else [entry]))
    with pytest.warns(RuntimeWarning, match="md1"):
        cov, degraded = extract_arrival_cov(response, _cov_stations(), "jobs")
    assert cov[0] is None
    assert len(degraded) == 1 and "md1" in degraded[0]
    assert "constructor cov_a" in degraded[0]


def test_a_tiny_mean_with_zero_variance_is_still_zero():
    cov, degraded = extract_arrival_cov(
        _cov_response(_ia("md1", 1e-200, 0.0)), _cov_stations(), "jobs")
    assert cov[0] == 0.0 and degraded == []


def test_a_weak_measurement_is_used_and_flagged():
    with pytest.warns(RuntimeWarning, match="success=false"):
        cov, degraded = extract_arrival_cov(
            _cov_response(_ia("md1", 2.0, 1.0, success=False)), _cov_stations(), "jobs")
    assert cov[0] == 0.5
    assert len(degraded) == 1 and "success=false" in degraded[0]
