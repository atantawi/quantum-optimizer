"""Response to per-station E[T], CIs, throughput, and quality flags (spec 5.3, 7)."""

import math
import warnings

from qopt.exceptions import MeasureMissingError
from qopt.qsim.spec import COV_A_MEASURE

SYSTEM_STATION = ""
"""Station key that system-level measures come back under.

CONFIRMED against a live qsim-service at 51a99c7 (tests/test_integration_qsim.py::
test_system_measure_key_inference_holds): a single-station M/M/1 network returned
{"station": "", "class": "jobs", "type": "system-response-time", "mean": 0.994325},
with nothing keyed on "system". This matches the inference from MeasureMapper
emitting referenceNode="" for system measures and SolutionsParser.domainStation
passing an empty name through (spec 5.3 gotcha 2) — the doubt existed only because no
qsim-service fixture pinned it, and that repo's own spec example says "system". If a
future service version changes this, the symptom is system_response_time is None plus
a RuntimeWarning, and the fix is this one line.
"""


def extract(response, stations, job_class):
    """Return (sojourn_times, ci, degraded, extras) for `stations`, in their order.

    Raises MeasureMissingError only for a station response-time: eq 22 then has no
    input at all, so warn-and-proceed does not apply. The other two requested measures
    are diagnostics, and their absence must not abort a run that has everything the
    mathematics requires (spec 7.1).

    A station's `ci` entry is None, rather than a (lower, upper) tuple, when the mean is
    present but the confidence interval is not: the mean is still usable for eq 22, but
    the noise floor cannot be estimated for that station (mirrors the throughput handling
    in qsim/analyzer.py's gamma-conservation check).

    The two diagnostics in `extras` keep their (mean, (lower, upper)) shape in that case
    and carry (None, None) bounds instead, so a caller that formats those bounds must
    handle the Nones. Either way the miss is warned and recorded in `degraded`: a mean
    without a CI is never passed through silently.
    """
    degraded = []
    if not response.get("completed", True):
        message = (
            f"qsim run {response.get('modelName')!r} reported completed=false: a cap "
            f"fired before all confidence intervals converged"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        degraded.append(message)

    index = {
        (m.get("station"), m.get("class"), m.get("type")): m
        for m in response.get("measures", [])
    }

    sojourn_times = []
    ci = []
    for st in stations:
        measure = index.get((st.name, job_class, st.SIM_MEASURE_TYPE))
        if measure is None or measure.get("mean") is None:
            raise MeasureMissingError(
                f"response has no {st.SIM_MEASURE_TYPE!r} for station {st.name!r} "
                f"class {job_class!r}; eq 22 has no input"
            )
        degraded.extend(_flag_weak(measure))
        sojourn_times.append(measure["mean"])
        lower, upper = measure.get("lower"), measure.get("upper")
        if lower is None or upper is None:
            message = (
                f"{st.name}: simulated response-time {measure['mean']:.6f} has no "
                f"confidence interval, so the noise floor cannot be estimated for it"
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            degraded.append(message)
            ci.append(None)
        else:
            ci.append((lower, upper))

    extras = {}
    system = index.get((SYSTEM_STATION, job_class, "system-response-time"))
    if system is None or system.get("mean") is None:
        message = (
            f"response has no 'system-response-time' keyed on station "
            f"{SYSTEM_STATION!r}; reporting it as None (spec 5.3 gotcha 2)"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        degraded.append(message)
        extras["system_response_time"] = None
    else:
        degraded.extend(_flag_weak(system))
        lower, upper = system.get("lower"), system.get("upper")
        if lower is None or upper is None:
            message = (
                f"'system-response-time' {system['mean']:.6f} has no confidence "
                f"interval; reporting the mean without one"
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            degraded.append(message)
        extras["system_response_time"] = (system["mean"], (lower, upper))

    throughput = {}
    for st in stations:
        measure = index.get((st.name, job_class, "throughput"))
        if measure is None or measure.get("mean") is None:
            if st.sim_conservation_checked:
                message = (
                    f"response has no 'throughput' for station {st.name!r}; the "
                    f"gamma-conservation check cannot run for it"
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                degraded.append(message)
            continue
        degraded.extend(_flag_weak(measure))
        throughput[st.name] = (
            measure["mean"], (measure.get("lower"), measure.get("upper"))
        )
    extras["throughput"] = throughput

    return sojourn_times, ci, degraded, extras


def extract_arrival_cov(response, stations, job_class):
    """Return (arrival_cov, degraded): each station's measured `cov_a`, or None.

    `cov_a = sqrt(variance / mean**2)` from the station's interarrival-time measure, read
    only for stations with `uses_measured_cov_a`. Every other entry is None silently, even
    though the response carries the measure for them too: reading it would record a number
    that enters nothing, and flagging a missing one would degrade a run for nothing.

    None, warned and recorded, for a using station whose measure is missing or whose moments
    cannot give a cov. The caller then prices that station at its constructor `cov_a` for
    this evaluation only. A missing CI is the measure's documented contract, so it is not
    flagged. An SCV of exactly 0 is deterministic arrivals and comes back as 0.0.
    `success == false` is used and flagged, like every other weak measure (7.2).
    """
    index = {
        (m.get("station"), m.get("class"), m.get("type")): m
        for m in response.get("measures", [])
    }
    arrival_cov = []
    degraded = []
    for st in stations:
        if not st.uses_measured_cov_a:
            arrival_cov.append(None)
            continue
        measure = index.get((st.name, job_class, COV_A_MEASURE))
        if measure is None:
            cov, reason = None, f"response has no {COV_A_MEASURE!r} for class {job_class!r}"
        else:
            cov, reason = _cov_from_moments(measure)
        if reason is not None:
            message = (
                f"{st.name}: no usable measured arrival cov ({reason}); pricing it at its "
                f"constructor cov_a for this evaluation"
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            degraded.append(message)
            arrival_cov.append(None)
            continue
        degraded.extend(_flag_weak(measure))
        arrival_cov.append(cov)
    return arrival_cov, degraded


def _cov_from_moments(measure):
    """(cov, None) from a present interarrival measure, or (None, reason) when it gives none.

    Divides by `mean` twice rather than by `mean**2`: a tiny mean then overflows the SCV
    to a rejected inf instead of underflowing `mean**2` to 0.0 and raising
    ZeroDivisionError in the middle of a run.
    """
    mean, variance = measure.get("mean"), measure.get("variance")
    if not _is_finite_number(mean) or mean <= 0:
        return None, f"interarrival mean is {mean!r}"
    if not _is_finite_number(variance) or variance < 0:
        return None, f"interarrival variance is {variance!r}"
    scv = variance / mean / mean
    if not math.isfinite(scv):
        return None, f"variance/mean**2 is not finite (mean={mean!r}, variance={variance!r})"
    return math.sqrt(scv), None


def _is_finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _flag_weak(measure):
    """success=false means that measure missed its CI target; use its mean anyway (7.2)."""
    if measure.get("success", True):
        return []
    message = (
        f"measure {measure.get('type')!r} at station {measure.get('station')!r} "
        f"reported success=false (precision {measure.get('precision')}); "
        f"using its mean anyway"
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    return [message]
