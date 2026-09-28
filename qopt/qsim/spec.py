"""Request envelope: a Network plus S becomes the POST /simulate body (spec 5.4)."""

MEASURES = (
    "response-time",         # E[T] for eq 22, every station type (spec 5.2)
    "system-response-time",  # Result diagnostic; the fork-join identity oracle (spec 8)
    "throughput",            # gamma-conservation witness (spec 6.8)
)
"""The closed, always-explicit measure list.

Never omit it and never send it empty: MeasureMapper then substitutes
DEFAULTS = [response-time, utilization, throughput, queue-length], and `utilization` and
`queue-length` are join-station numbers at a fork-join node that come back with
success: true and no warning. The list is closed because nothing outside these three
enters eq 21, eq 22, the objective, or the fixed point.
"""

COV_A_MEASURE = "interarrival-time"
"""Per-station interarrival moments, from which a measured `cov_a` is computed.

Requested only when some station has `uses_measured_cov_a` (SimulationAnalyzer). Facts
from qsim-service's README at 8e7358b: it reports `mean`, `variance` and `stdDev` of the
interarrival time, so SCV = variance/mean**2; it reports NO confidence interval, because
JMT's is on the rate; it switches on JMT's per-sample logging for itself alone, roughly
25-30% wall clock plus temporary disk -- which is why `secondMoments` is never sent, since
that would put every other measure on the same cost; and on a fork-join node it is taken
at the fork. A service older than 8e7358b rejects it as an unsupported measure type.
qsim's measure list applies network-wide, so this is taken at every station and its cost
is per station: it grows with network size (see SimulationAnalyzer for qopt's figures).
"""


def build_request(network, S, *, seed, stopping, measures=MEASURES):
    """Wrap network.to_model_dict(S) in qsim's seed / stopping / measures envelope.

    `seed` of None omits the field, letting qsim choose (seed_policy=None).
    """
    measures = tuple(measures)
    if not measures:
        raise ValueError(
            "measures must be non-empty; qsim substitutes its own DEFAULTS otherwise, "
            "two of which are join-station numbers at a fork-join node (spec 5.4)"
        )
    request = {
        "model": network.to_model_dict(S),
        "stopping": dict(stopping),
        "measures": list(measures),
    }
    if seed is not None:
        request["seed"] = seed
    return request
