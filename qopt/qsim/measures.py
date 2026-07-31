"""Response to per-station E[T], CIs, throughput, and quality flags (spec 5.3, 7)."""

import warnings

from qopt.exceptions import MeasureMissingError

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
        ci.append((measure.get("lower"), measure.get("upper")))

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
        extras["system_response_time"] = (
            system["mean"], (system.get("lower"), system.get("upper"))
        )

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
