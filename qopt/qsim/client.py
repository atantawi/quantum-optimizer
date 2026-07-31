"""Transport, POST /simulate, and HTTP-status-to-exception mapping (spec 7.1, 7.3, 7.4)."""

import json
import urllib.error
import urllib.request

from qopt.exceptions import (
    SimulationEngineError,
    SimulationRequestError,
    SimulationTransportError,
)

DEFAULT_STOPPING = {
    "alpha": 0.05,
    "precision": 0.05,
    "minSamples": 20000,
    "maxSamples": 1000000,
    "maxWallClockSeconds": 120,
}

TIMEOUT_MARGIN_SECONDS = 10.0
"""How far the client's read timeout must clear the server's own watchdog."""

_REQUEST_STATUSES = (400, 405, 413, 422)


def urllib_transport(url, body, timeout):
    """Default transport: POST when `body` is bytes, GET when it is None.

    Returns (status, body_bytes). 4xx/5xx are returned rather than raised, because
    qsim-service puts a structured {"error", "details"} body on every failure.
    """
    request = urllib.request.Request(
        url,
        data=body,
        method="GET" if body is None else "POST",
        headers={} if body is None else {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SimulationTransportError(f"{url}: {exc}") from exc


class QsimClient:
    """Speaks POST /simulate and GET /health to a qsim-service instance."""

    def __init__(self, base_url, *, timeout=None, stopping=None, transport=None,
                 preflight=False):
        self.base_url = base_url.rstrip("/")
        self.stopping = dict(DEFAULT_STOPPING if stopping is None else stopping)
        wall_clock = self.stopping.get("maxWallClockSeconds")
        if wall_clock is None:
            raise ValueError(
                "stopping must set maxWallClockSeconds so the client timeout can be "
                "checked against it (spec 7.3)"
            )
        self.timeout = (
            float(wall_clock) + 2 * TIMEOUT_MARGIN_SECONDS if timeout is None
            else float(timeout)
        )
        if self.timeout <= wall_clock + TIMEOUT_MARGIN_SECONDS:
            raise ValueError(
                f"timeout {self.timeout} must exceed maxWallClockSeconds {wall_clock} "
                f"plus a {TIMEOUT_MARGIN_SECONDS}s margin, or the client kills runs the "
                f"server would have completed"
            )
        self.transport = urllib_transport if transport is None else transport
        if preflight:
            self.health()

    def health(self):
        """One GET, so a misconfigured URL fails here instead of on iteration 1."""
        status, raw = self.transport(f"{self.base_url}/health", None, self.timeout)
        if status != 200:
            raise SimulationTransportError(
                f"{self.base_url}/health returned HTTP {status}: {raw[:200]!r}"
            )
        return self._decode(raw)

    def post_simulate(self, request):
        """Run one simulation. Returns the parsed response body."""
        body = json.dumps(request).encode("utf-8")
        status, raw = self.transport(f"{self.base_url}/simulate", body, self.timeout)
        if status == 200:
            return self._decode(raw)
        detail = self._error_detail(raw)
        if status in _REQUEST_STATUSES:
            # Our JSON was wrong: a spec.py bug, or a network qsim will not accept.
            raise SimulationRequestError(f"HTTP {status} from /simulate: {detail}")
        if 500 <= status < 600:
            raise SimulationEngineError(f"HTTP {status} from /simulate: {detail}")
        raise SimulationTransportError(
            f"unexpected HTTP {status} from /simulate: {detail}"
        )

    @staticmethod
    def _decode(raw):
        try:
            return json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise SimulationEngineError(
                f"unreadable response body: {raw[:200]!r}"
            ) from exc

    @staticmethod
    def _error_detail(raw):
        """qsim errors are {"error": str, "details": [str]}; fall back to raw bytes."""
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return repr(raw[:200])
        if isinstance(payload, dict) and "error" in payload:
            details = "; ".join(payload.get("details") or [])
            return payload["error"] + (f" ({details})" if details else "")
        return repr(payload)
