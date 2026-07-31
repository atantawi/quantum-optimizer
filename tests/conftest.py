"""Shared test doubles for the simulation path."""

import json

import pytest


class FakeTransport:
    """Records every call and replays a scripted (status, body) sequence.

    A single (status, payload) pair is replayed for every call; a list is consumed
    one entry per call, so a test can script a failure on iteration 3.
    """

    def __init__(self, script, health=(200, {"status": "ok"})):
        self.script = script if isinstance(script, list) else [script]
        self.repeat_last = not isinstance(script, list)
        self.health = health
        self.calls = []          # [(url, request_dict_or_None, timeout)]

    def __call__(self, url, body, timeout):
        request = None if body is None else json.loads(body)
        self.calls.append((url, request, timeout))
        if body is None:
            status, payload = self.health
        elif self.repeat_last:
            status, payload = self.script[-1]
        else:
            status, payload = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return status, raw

    @property
    def requests(self):
        return [r for _, r, _ in self.calls if r is not None]


def _measure(station, job_class, type_, mean, half_width=0.01, success=True):
    return {
        "station": station, "class": job_class, "type": type_,
        "mean": mean, "lower": mean - half_width, "upper": mean + half_width,
        "alpha": 0.05, "precision": 0.02, "success": success,
        "samplesAnalyzed": 40000, "samplesDiscarded": 1000,
        "variance": 0.01, "stdDev": 0.1,
    }


@pytest.fixture
def measure():
    """Factory for one qsim MeasureResult entry."""
    return _measure


@pytest.fixture
def sim_response():
    """Factory for a full /simulate response body.

    sim_response(sojourn={"mm1": 0.4}, throughput={"mm1": 0.6}, system=1.2)
    """
    def build(*, sojourn, throughput=None, system=None, job_class="jobs",
              completed=True, seed=20260729, model_name="qopt-mixed-network",
              half_width=0.01, success=True):
        measures = [
            _measure(name, job_class, "response-time", mean, half_width, success)
            for name, mean in sojourn.items()
        ]
        for name, mean in (throughput or {}).items():
            measures.append(
                _measure(name, job_class, "throughput", mean, half_width, success)
            )
        if system is not None:
            measures.append(
                _measure("", job_class, "system-response-time", system, half_width, success)
            )
        return {
            "modelName": model_name,
            "solutionMethod": "simulation",
            "seed": seed,
            "wallClockSeconds": 8.3,
            "completed": completed,
            "measures": measures,
        }

    return build
