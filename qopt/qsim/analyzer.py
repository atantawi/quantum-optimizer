"""SimulationAnalyzer: one POST per evaluate(), plus the gamma-conservation check."""

import warnings

from qopt.analyzer import Analyzer, Evaluation
from qopt.exceptions import SimulationQualityError, SimulationRequestError
from qopt.qsim.measures import extract, extract_arrival_cov
from qopt.qsim.spec import COV_A_MEASURE, MEASURES, build_request

FRESH_SEED_OFFSET = 1_000_000
"""Offset for the final independently-seeded evaluation (spec 6.5)."""

_SEED_POLICIES = ("fixed", "vary", None)

_OLD_SERVICE_REJECTION = f"unsupported measure type: '{COV_A_MEASURE}'"
"""What a qsim-service older than 8e7358b says to a request carrying the measure."""


class SimulationAnalyzer(Analyzer):
    """Obtains E[T] for the whole network from one qsim-service run per evaluate().

    `strict=True` here is deliberately EARLY, not late: it fails at the first degraded
    `evaluate()` call (this instance's own `degraded` list is checked at the end of that
    one call). This is a different timing from `Optimizer(strict=True)`, which only
    raises after the whole loop plus the final evaluation have run, so it can report the
    whole run's accumulated audit trail at once. Both are deliberate (finding 7); this
    class's strict does not propagate to or from the optimizer's.

    `measure_cov_a=True` asks qsim for each slope G/G/1 station's interarrival moments
    whenever the network has one (`Station.uses_measured_cov_a`), so `phi` prices the
    arrival variability the station actually sees rather than its constructor `cov_a`. It
    needs qsim-service 8e7358b+. The measure list is network-wide, so the measure is taken
    at every station, not only at the ones that use it, and its cost grows with network
    size: qsim-service quotes 25-30% wall clock per measure, and qopt's own runs measured
    +35% on a 3-station network and about +50% on a 14-station one. False sends exactly
    the pre-measurement request and prices every station at its constructor `cov_a`.
    """

    is_stochastic = True
    requires_strict_stability = True
    """`rho == 1` is outside what this analyzer can simulate; see `evaluate`'s preflight."""

    def __init__(self, network, client, *, seed=20260729, seed_policy="fixed",
                 strict=False, measure_cov_a=True):
        if seed_policy not in _SEED_POLICIES:
            raise ValueError(
                f"seed_policy must be 'fixed', 'vary', or None, got {seed_policy!r}"
            )
        self.network = network
        self.client = client
        self.seed = seed
        self.seed_policy = seed_policy
        self.strict = strict
        self.measure_cov_a = measure_cov_a
        self.iteration = 0

    def _measures_cov_a(self, stations):
        """Ask for interarrival moments only when some station will use them.

        For every other network the request -- and its cost -- is exactly what it was
        before the measure existed.
        """
        return self.measure_cov_a and any(st.uses_measured_cov_a for st in stations)

    def _seed_for(self, fresh_seed):
        if self.seed_policy is None:
            return None
        if fresh_seed:
            return self.seed + FRESH_SEED_OFFSET
        if self.seed_policy == "vary":
            return self.seed + self.iteration
        return self.seed                      # common random numbers

    def evaluate(self, stations, S, *, fresh_seed=False):
        stations = list(stations)
        if len(stations) != len(self.network.stations) or any(
            a is not b for a, b in zip(stations, self.network.stations)
        ):
            raise ValueError(
                "stations must be this analyzer's network stations, in order"
            )
        for st, Si in zip(stations, S):
            # Fail before spending minutes of simulation on a saturated network (7.3).
            # Same guard and message sojourn_time uses, but STRICT: `S*mu == gamma` is
            # refused here even for a station that admits full utilization analytically.
            # `Station.admits_full_utilization` is a statement about `E[T] = 1/(S*mu)` at
            # `k = (cov_a^2 + cov_s^2)/2 == 0`, and `cov_a` never reaches the simulator --
            # `Network.to_model_dict` takes the arrival distribution from
            # `Network.arrival_scv` and the routing, and only `cov_s` goes into the service
            # node. So a `cov_a == 0` station in a network with the default
            # `arrival_scv == 1.0` emits exponential arrivals against deterministic service:
            # at the boundary that is a saturated M/D/1, exactly what this guard is for.
            # Relaxing it would need the EMITTED arrival process shown deterministic at that
            # station, which qopt cannot currently conclude from the model dict alone.
            #
            # This still has to be checked here, even though `Optimizer.run` now keeps its
            # candidates inside this domain: `evaluate` is public and is called directly,
            # and the optimizer's guard reads `requires_strict_stability` rather than
            # duplicating the test.
            st.check_stable(Si, strict=self.requires_strict_stability)

        measure_cov_a = self._measures_cov_a(stations)
        request = build_request(
            self.network, S,
            seed=self._seed_for(fresh_seed),
            stopping=self.client.stopping,
            measures=MEASURES + (COV_A_MEASURE,) if measure_cov_a else MEASURES,
        )
        try:
            response = self.client.post_simulate(request)
        except SimulationRequestError as exc:
            # No retry without the measure: that would quietly price every station at its
            # constructor cov_a, the assumption this measurement exists to replace.
            if measure_cov_a and _OLD_SERVICE_REJECTION in str(exc):
                raise SimulationRequestError(
                    f"{exc} -- measuring arrival cov_a needs qsim-service 8e7358b or "
                    f"later; pass SimulationAnalyzer(measure_cov_a=False) to run against "
                    f"this service on the stations' constructor cov_a instead"
                ) from exc
            raise
        if not fresh_seed:
            self.iteration += 1               # the final evaluation is not an iteration

        sojourn_times, ci, degraded, extras = extract(
            response, stations, self.network.job_class
        )
        degraded.extend(_conservation_misses(stations, extras["throughput"]))
        extras["seed"] = response.get("seed")
        extras["wallClockSeconds"] = response.get("wallClockSeconds")

        arrival_cov = None
        if measure_cov_a:
            arrival_cov, cov_degraded = extract_arrival_cov(
                response, stations, self.network.job_class
            )
            degraded.extend(cov_degraded)

        if self.strict and degraded:
            raise SimulationQualityError("; ".join(degraded))
        return Evaluation(
            sojourn_times=sojourn_times, ci=ci, degraded=degraded, extras=extras,
            arrival_cov=arrival_cov,
        )


def _conservation_misses(stations, throughput):
    """Simulated throughput must bracket the derived gamma at every station (6.8).

    An independent witness that solve_traffic and to_model_dict describe the same
    network. Warn and record rather than fail: a watchdog-truncated run can widen or
    bias throughput enough to miss legitimately.
    """
    misses = []
    for st in stations:
        if not st.sim_conservation_checked:   # fork-join: qsim-service#8
            continue
        entry = throughput.get(st.name)
        if entry is None:
            continue                          # already flagged by measures.extract
        mean, (lower, upper) = entry
        if lower is None or upper is None:
            message = (
                f"{st.name}: simulated throughput {mean:.6f} has no confidence "
                f"interval, so the gamma-conservation check cannot run"
            )
        elif lower <= st.gamma <= upper:
            continue
        else:
            message = (
                f"{st.name}: simulated throughput {mean:.6f} CI "
                f"({lower:.6f}, {upper:.6f}) excludes derived gamma={st.gamma:.6f}"
            )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        misses.append(message)
    return misses
