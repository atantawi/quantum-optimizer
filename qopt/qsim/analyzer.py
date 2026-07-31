"""SimulationAnalyzer: one POST per evaluate(), plus the gamma-conservation check."""

import warnings

from qopt.analyzer import Analyzer, Evaluation
from qopt.exceptions import SimulationQualityError
from qopt.qsim.measures import extract
from qopt.qsim.spec import build_request

FRESH_SEED_OFFSET = 1_000_000
"""Offset for the final independently-seeded evaluation (spec 6.5)."""

_SEED_POLICIES = ("fixed", "vary", None)


class SimulationAnalyzer(Analyzer):
    """Obtains E[T] for the whole network from one qsim-service run per evaluate()."""

    is_stochastic = True

    def __init__(self, network, client, *, seed=20260729, seed_policy="fixed",
                 strict=False):
        if seed_policy not in _SEED_POLICIES:
            raise ValueError(
                f"seed_policy must be 'fixed', 'vary', or None, got {seed_policy!r}"
            )
        self.network = network
        self.client = client
        self.seed = seed
        self.seed_policy = seed_policy
        self.strict = strict
        self.iteration = 0

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
            # Same guard and message sojourn_time uses.
            st.check_stable(Si)

        request = build_request(
            self.network, S,
            seed=self._seed_for(fresh_seed),
            stopping=self.client.stopping,
        )
        response = self.client.post_simulate(request)
        if not fresh_seed:
            self.iteration += 1               # the final evaluation is not an iteration

        sojourn_times, ci, degraded, extras = extract(
            response, stations, self.network.job_class
        )
        degraded.extend(_conservation_misses(stations, extras["throughput"]))
        extras["seed"] = response.get("seed")
        extras["wallClockSeconds"] = response.get("wallClockSeconds")
        if self.strict and degraded:
            raise SimulationQualityError("; ".join(degraded))
        return Evaluation(
            sojourn_times=sojourn_times, ci=ci, degraded=degraded, extras=extras
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
