# Simulation Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `qopt`'s fixed-point optimizer obtain `E[T]` from a discrete-event simulation of the whole network via the `qsim-service` HTTP/JSON API, instead of from independent per-station closed-form approximations.

**Architecture:** A new `Network` owns topology (stations + probabilistic routes + exogenous λ), solves the traffic equations to *derive* each station's `γ`, and serializes itself into `qsim-service`'s `model` block. A new network-level `Analyzer` seam replaces the optimizer's per-station `st.zeta(Si)` pull with a vector-in / vector-out `evaluate(stations, S) -> Evaluation`; `AnalyticAnalyzer` preserves today's behavior bit-for-bit and `SimulationAnalyzer` issues exactly one POST per iteration. The mathematics (eq 21 `allocate`, eq 22 `ζ = E[T]·(Sµ − γ)`, the objective, the loop shape) is untouched — only the *source* of `E[T]` differs.

**Tech Stack:** Python 3.12 (3.10+ compatible), stdlib only (`math`, `abc`, `dataclasses`, `json`, `urllib.request`, `warnings`); `pytest` for tests. No third-party runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-07-29-simulation-support-design.md`. Section references (`§4.2`, `§6.8`, …) below point into it.

## Global Constraints

- Python **>= 3.10**. Target/dev interpreter is 3.12. Run everything through `.venv/bin/python`.
- `qopt` declares **zero runtime dependencies** — `pyproject.toml`'s `dependencies = []` must not change. The default HTTP transport is stdlib `urllib.request`. No `numpy`, no `httpx`, no NetworkX.
- The HTTP boundary to `qsim-service` is a **licensing firewall**: `qsim-service` is GPL v2, `qopt` is Apache-2.0. `qopt` speaks HTTP/JSON and never imports or links JMT code.
- Tests use **pytest**. Numeric comparisons use `pytest.approx(..., rel=1e-9)` or tighter unless a looser tolerance is explicitly stated. Where this plan says **bit-for-bit** / **bitwise**, use `==` on floats, not `approx`.
- **The existing test suite must pass unmodified** (spec §11 criterion 1). Baseline before starting: `48 passed`. No test file that exists today may be edited by tasks 1–9.
- The exogenous job class is `"jobs"`. Emitted source/sink node names are `"src"` and `"snk"`.
- Station names must be non-empty, unique, contain no `__`, and not be `"src"` or `"snk"` (§4.2). They are JSON node names, routing keys, and DOT identifiers.
- Structural `Network` failures raise `TopologyError`; scalar parameter guards raise `ValueError`, matching the current convention.
- The requested measure list is **closed and always explicit**: `("response-time", "system-response-time", "throughput")` (§5.4). Never omit it, never send it empty.
- `qsim-service` contract facts this plan encodes were verified against that repo at commit `51a99c7`. Re-verify if `MeasureMapper`, `JsimgWriter.expandedMeasureNode`, or `SolutionsParser.REVERSE` change.

---

## File Structure

```
quantum-optimizer/                       (repo root == cwd)
  qopt/
    exceptions.py                  MODIFY  + TopologyError, SimulationError tree
    station.py                     MODIFY  gamma optional + bind_gamma, zeta_from,
                                           sim_node, SIM_MEASURE_TYPE,
                                           sim_conservation_checked, DOT_SHAPE,
                                           distribution_dict()
    traffic.py                     CREATE  solve_traffic() — 20 lines, no deps
    network.py                     CREATE  Route, Network (validation, γ derivation,
                                           to_model_dict, to_dot)
    analyzer.py                    CREATE  Analyzer ABC, Evaluation, AnalyticAnalyzer
    allocator.py                   MODIFY  + noise_floor()
    optimizer.py                   MODIFY  + analyzer seam, warm start, damping,
                                           CI-aware stop, extended Result
    __init__.py                    MODIFY  + new public exports
    qsim/
      __init__.py                  CREATE  package marker
      client.py                    CREATE  QsimClient, transport, HTTP → exception
      spec.py                      CREATE  MEASURES, build_request()
      measures.py                  CREATE  extract() — response → E[T]/CI/throughput
      analyzer.py                  CREATE  SimulationAnalyzer + γ-conservation check
  examples/
    mixed_network.py               MODIFY  §4.1.1 topology, γ derived
    simulated_tandem.py            CREATE  M/D/1 → M/M/1, analytic vs simulated
    simulated_mixed_network.py     CREATE  §4.1.1 analytic vs simulated (incl. fork-join)
  tests/
    conftest.py                    CREATE  FakeTransport + response-builder fixtures
    fixtures/
      qopt_mixed_network_request.json  CREATE  golden model block at S=(3,4,5)
    test_traffic.py                CREATE
    test_network.py                CREATE
    test_network_model_dict.py     CREATE
    test_analyzer.py               CREATE
    test_qsim_client.py            CREATE
    test_qsim_spec.py              CREATE
    test_qsim_measures.py          CREATE
    test_qsim_analyzer.py          CREATE
    test_noise_floor.py            CREATE
    test_optimizer_loop.py         CREATE
    test_integration_qsim.py       CREATE  gated on QOPT_QSIM_URL, skipped by default
```

Why this split: `traffic.py` is pure arithmetic with no knowledge of stations, so it is testable against closed-form λ values alone. `network.py` owns the *model* vocabulary; `qsim/spec.py` owns the *request envelope*; neither knows the other's job. `qsim/` is a subpackage because its four files change together and only together — swapping the simulation backend touches nothing outside it.

Interface summary — locked here, tasks must match exactly:

```python
# qopt/exceptions.py
TopologyError(QOptError)
SimulationError(QOptError)
  SimulationTransportError(SimulationError)
  SimulationRequestError(SimulationError)
  SimulationEngineError(SimulationError)
  SimulationQualityError(SimulationError)
  MeasureMissingError(SimulationError)

# qopt/station.py
distribution_dict(rate: float, scv: float) -> dict
Station.__init__(self, gamma=None, mu=None, weight=1.0, *, name=None)   # mu required via guard
Station.gamma                        -> property; ValueError if unset
Station.bind_gamma(value)            -> None; idempotent; ValueError if gamma was explicit
Station.zeta_from(T, S)              -> T * (S * self.mu - self.gamma)
Station.zeta(S)                      -> self.zeta_from(self.sojourn_time(S), S)
Station.check_stable(S)              -> None; raises InstabilityError if S*mu <= gamma
Station.SIM_MEASURE_TYPE             = "response-time"    # base constant, NOT abstract
Station.sim_conservation_checked     = True               # class attr, overridable
Station.DOT_SHAPE                    = "box"
Station.sim_node(S, job_class)        -> abstract; a qsim node dict
GG1Station.sim_node(S, job_class)     -> queue node, service from cov_s
ForkJoinStation.sim_node(S, job_class)-> fork-join node, branches S*mu / S*r*mu, join "all"
ForkJoinStation.sim_conservation_checked = False          # qsim-service#8
ForkJoinStation.DOT_SHAPE                = "box3d"

# qopt/traffic.py
solve_traffic(nodes, edges, arrival_rate, source, sink, *, tol=1e-12, max_iter=10_000)
    -> (dict[str, float], int)   # (lambdas, iterations); TopologyError if the cap is hit

# qopt/network.py
Route(src, dst, probability=1.0)                      # frozen dataclass
Network(stations, routes, arrival_rate, *, name="qopt-network",
        arrival_scv=1.0, job_class="jobs")
Network.SOURCE = "src";  Network.SINK = "snk"
Network.stations, .routes, .arrival_rate, .arrival_scv, .job_class, .name
Network.gammas         -> dict[str, float]
Network.traffic_iterations -> int
Network.__len__(), Network.__iter__()                  # len/iter over stations
Network.to_model_dict(S) -> dict                       # exactly qsim's `model` block
Network.to_dot() -> str
# NO from_model_dict — S is not recoverable from the emitted S*mu product

# qopt/analyzer.py
Evaluation(sojourn_times, ci=None, degraded=<list>, extras=<dict>)   # dataclass
Analyzer.is_stochastic : bool
Analyzer.evaluate(stations, S, *, fresh_seed=False) -> Evaluation    # abstract
AnalyticAnalyzer()                                    # is_stochastic = False; ci = None

# qopt/allocator.py
noise_floor(stations, C, zeta_vec, dzeta) -> float

# qopt/qsim/client.py
DEFAULT_STOPPING : dict
TIMEOUT_MARGIN_SECONDS = 10.0
urllib_transport(url, body, timeout) -> (int, bytes)   # body None => GET
QsimClient(base_url, *, timeout=None, stopping=None, transport=None, preflight=False)
QsimClient.stopping, .timeout, .base_url, .transport
QsimClient.health() -> dict
QsimClient.post_simulate(request: dict) -> dict

# qopt/qsim/spec.py
MEASURES = ("response-time", "system-response-time", "throughput")
build_request(network, S, *, seed, stopping, measures=MEASURES) -> dict

# qopt/qsim/measures.py
SYSTEM_STATION = ""
extract(response, stations, job_class) -> (list, list, list, dict)
    # (sojourn_times, ci, degraded, extras)
    # extras["throughput"]: name -> (mean, (lo, hi));  extras["system_response_time"]

# qopt/qsim/analyzer.py
SimulationAnalyzer(network, client, *, seed=20260729, seed_policy="fixed", strict=False)
SimulationAnalyzer.is_stochastic = True
SimulationAnalyzer.iteration : int

# qopt/optimizer.py
Optimizer(stations, budget, *, analyzer=None, tol=1e-9, max_iter=None, initial_zeta=None,
          damping=None, noise_kappa=1.0, final_evaluation=True, strict=False, warm_start=True)
Result(capacities, sojourn_times, zeta, objective, iterations, converged, residual,
       sojourn_ci=None, noise_floor=None, stop_reason="tol", warm_start_iterations=0,
       degraded=<list>, system_response_time=None, sim_calls=0)
```

---

## Task 1: Exceptions, optional γ, and `zeta_from`

Widens `Station` so `γ` can be derived later without changing any existing call site, and splits eq 22 so an externally supplied `E[T]` can drive it.

**Files:**
- Modify: `qopt/exceptions.py` (append after line 13)
- Modify: `qopt/station.py:20-55` (`Station.__init__`, `gamma`, `zeta`), and the three subclass signatures at `:61`, `:84`, `:100-108`, `:119`
- Test: `tests/test_station_gamma.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `TopologyError`, `SimulationError`, `SimulationTransportError`, `SimulationRequestError`, `SimulationEngineError`, `SimulationQualityError`, `MeasureMissingError` (all under `QOptError`); `Station.__init__(gamma=None, mu=None, weight=1.0, *, name=None)`; `Station.gamma` property; `Station.bind_gamma(value)`; `Station.zeta_from(T, S)`; `Station.check_stable(S)`.

**Why the signature change is safe** (verified, do not re-derive): every `Station` subclass construction in `tests/` and `examples/` is keyword-based, so defaulting `gamma` and `mu` to `None` with an explicit "mu is required" guard is backward compatible. There are no ad-hoc `Station` subclasses in `tests/`, so adding an abstract method in Task 4 is also safe.

- [ ] **Step 1: Write the failing test** — `tests/test_station_gamma.py`

```python
import pytest

from qopt.exceptions import QOptError, SimulationError, TopologyError
from qopt.station import ForkJoinStation, GG1Station


def test_new_exceptions_are_qopt_errors():
    from qopt.exceptions import (
        MeasureMissingError,
        SimulationEngineError,
        SimulationQualityError,
        SimulationRequestError,
        SimulationTransportError,
    )

    assert issubclass(TopologyError, QOptError)
    assert issubclass(SimulationError, QOptError)
    for cls in (
        SimulationTransportError,
        SimulationRequestError,
        SimulationEngineError,
        SimulationQualityError,
        MeasureMissingError,
    ):
        assert issubclass(cls, SimulationError)


def test_explicit_gamma_still_works():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    assert st.gamma == 0.6


def test_mu_is_required():
    with pytest.raises(ValueError, match="mu is required"):
        GG1Station.mm1(gamma=0.6, c=2.0)


def test_gamma_omitted_raises_on_first_use():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="unbound")
    with pytest.raises(ValueError, match="no gamma"):
        st.gamma
    with pytest.raises(ValueError, match="no gamma"):
        st.sojourn_time(2.0)


def test_bind_gamma_fills_it():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    st.bind_gamma(0.6)
    assert st.gamma == 0.6
    assert st.sojourn_time(2.0) == pytest.approx(1.0 / (2.0 - 0.6), rel=1e-12)


def test_bind_gamma_is_idempotent_for_the_same_value():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    st.bind_gamma(0.6)
    st.bind_gamma(0.6)
    assert st.gamma == 0.6


def test_bind_gamma_rejects_a_conflicting_value():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    st.bind_gamma(0.6)
    with pytest.raises(ValueError, match="cannot rebind"):
        st.bind_gamma(0.7)


def test_bind_gamma_rejects_an_explicitly_constructed_gamma():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="a")
    with pytest.raises(ValueError, match="explicit gamma"):
        st.bind_gamma(0.6)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bind_gamma_validates_the_value(bad):
    st = GG1Station.mm1(mu=1.0, c=2.0, name="a")
    with pytest.raises(ValueError):
        st.bind_gamma(bad)


def test_check_stable_is_public_and_uses_the_same_guard():
    from qopt.exceptions import InstabilityError

    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="a")
    st.check_stable(1.0)                      # S*mu = 1.0 > 0.6, no raise
    with pytest.raises(InstabilityError, match="unstable"):
        st.check_stable(0.6)                  # S*mu = 0.6 == gamma


def test_check_stable_requires_a_bound_gamma():
    st = GG1Station.mm1(mu=1.0, c=2.0, name="unbound")
    with pytest.raises(ValueError, match="no gamma"):
        st.check_stable(2.0)


def test_zeta_from_accepts_an_external_sojourn_time():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    # zeta = T * (S*mu - gamma) = 2.5 * (1.0 - 0.6)
    assert st.zeta_from(2.5, 1.0) == pytest.approx(1.0, rel=1e-12)


def test_zeta_delegates_to_zeta_from_bitwise():
    for st in (
        GG1Station.md1(gamma=0.6, mu=1.0, c=1.0),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),
    ):
        for S in (1.5, 2.0, 4.0):
            assert st.zeta(S) == st.zeta_from(st.sojourn_time(S), S)


def test_forkjoin_and_md1_accept_omitted_gamma():
    fj = ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj")
    md = GG1Station.md1(mu=1.0, c=1.0, name="md")
    fj.bind_gamma(0.5)
    md.bind_gamma(0.4)
    assert fj.gamma == 0.5 and md.gamma == 0.4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_station_gamma.py -v`
Expected: FAIL — `ImportError: cannot import name 'TopologyError'`.

- [ ] **Step 3: Append the new exceptions to `qopt/exceptions.py`**

```python
class TopologyError(QOptError):
    """Raised when a Network's structure is not a well-formed open chain (spec 4.2)."""


class SimulationError(QOptError):
    """Base class for failures of the simulation-backed evaluation path."""


class SimulationTransportError(SimulationError):
    """The simulation service was unreachable: refused, timed out, or DNS failure."""


class SimulationRequestError(SimulationError):
    """The service rejected our request (HTTP 400/422): a spec.py bug or an invalid network."""


class SimulationEngineError(SimulationError):
    """The service failed while simulating (HTTP 500), or returned an unreadable body."""


class SimulationQualityError(SimulationError):
    """A simulation result was degraded and strict mode was requested (spec 7.2)."""


class MeasureMissingError(SimulationError):
    """The response lacked a station response-time that eq 22 requires (spec 7.1)."""
```

- [ ] **Step 4: Make `gamma` optional and add `bind_gamma` / `zeta_from` in `qopt/station.py`**

Replace `Station.__init__` (`:20-31`) with:

```python
    def __init__(self, gamma=None, mu=None, weight=1.0, *, name=None):
        # `isfinite` first: NaN passes every ordering comparison, so `nan <= 0` is False.
        if gamma is not None and (not math.isfinite(gamma) or gamma <= 0):
            raise ValueError(f"gamma must be a finite number > 0, got {gamma}")
        if mu is None:
            raise ValueError("mu is required")
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError(f"mu must be a finite number > 0, got {mu}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight must be a finite number > 0, got {weight}")
        self._gamma = gamma
        self._gamma_explicit = gamma is not None
        self.mu = mu
        self.weight = weight
        self.name = name

    @property
    def gamma(self):
        """Arrival rate. Either passed explicitly or derived by a Network (spec 4.1)."""
        if self._gamma is None:
            raise ValueError(
                f"station {self.name!r} has no gamma: pass gamma=... explicitly, or add "
                f"the station to a Network, which derives it from the traffic equations"
            )
        return self._gamma

    def bind_gamma(self, value):
        """Attach a Network-derived gamma. Idempotent for an identical value.

        gamma is derived-only for stations in a Network: there is no silent override of an
        explicitly constructed value, and no rebinding to a second network (spec 4.1).
        """
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"derived gamma must be a finite number > 0, got {value}")
        if self._gamma_explicit:
            raise ValueError(
                f"station {self.name!r} was constructed with an explicit gamma="
                f"{self._gamma}; gamma is derived-only for stations in a Network"
            )
        if self._gamma is not None and self._gamma != value:
            raise ValueError(
                f"station {self.name!r} is already bound to gamma={self._gamma}, "
                f"cannot rebind to {value}"
            )
        self._gamma = value
```

Also update the class docstring's `gamma:` line to read:

```
        gamma: arrival rate. Optional at construction: a Network derives it from the
            traffic equations and binds it via bind_gamma().
```

Replace `Station.zeta` (`:47-49`) with:

```python
    def zeta_from(self, T, S):
        """Invert the functional form (eq 22) for an externally supplied E[T].

        Pure station arithmetic, independent of where E[T] came from — the analytic
        sojourn time or a simulation run.
        """
        return T * (S * self.mu - self.gamma)

    def zeta(self, S):
        """Eq 22 evaluated at this station's own analytic sojourn time."""
        return self.zeta_from(self.sojourn_time(S), S)
```

Add `check_stable` next to the existing `_check_stable`, so callers outside `station.py`
have a public way to run the stability guard without duplicating its message:

```python
    def check_stable(self, S):
        """Raise InstabilityError if capacity S leaves this station unstable.

        Public counterpart to `_check_stable`, which takes the already-computed
        effective rate. Lets a caller fail fast before spending an expensive
        evaluation (spec 7.3) without reimplementing the check or its message.
        """
        self._check_stable(S * self.mu)
```

Then default `gamma` and `mu` to `None` in the three subclass signatures, changing nothing else about them:

```python
class SingleServerStation(Station):
    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, name=None):
        # body unchanged: super().__init__(gamma, mu, weight, name=name) + the c guard

class GG1Station(SingleServerStation):
    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, cov_a, cov_s, name=None):
        # body unchanged: super().__init__(...) + the cov_a / cov_s guards

    @classmethod
    def mm1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None):
        # body unchanged: return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=1.0, name=name)

    @classmethod
    def md1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None):
        # body unchanged: return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=0.0, name=name)

class ForkJoinStation(Station):
    def __init__(self, gamma=None, mu=None, weight=1.0, *, r, c1, c2, name=None):
        # body unchanged: super().__init__(...) + the r / c1 / c2 guards
```

Only the two defaults change in each. Do not touch the guard bodies — `gamma=0.0` must
still raise `ValueError`, which `tests/test_station.py::test_construction_validation`
asserts.

- [ ] **Step 5: Run the new test and the full suite**

Run: `.venv/bin/python -m pytest tests/test_station_gamma.py -v && .venv/bin/python -m pytest -q`
Expected: new file all PASS; full suite green with no failures. **Every one of the 48 pre-existing tests must still pass with no edits to their files** — check with `git diff --stat HEAD -- tests/test_station.py tests/test_allocator.py tests/test_optimizer.py tests/test_forkjoin_approx.py tests/test_smoke.py tests/test_example.py`, which must be empty.

- [ ] **Step 6: Commit**

```bash
git add qopt/exceptions.py qopt/station.py tests/test_station_gamma.py
git commit -m "feat: optional derived gamma, zeta_from, and the simulation exception tree"
```

---

## Task 2: `solve_traffic`

The traffic equations `λ = λ_ext + Pᵀλ`, solved by fixed-point iteration from `λ = 0`. Pure arithmetic over names and probabilities — it knows nothing about stations, so it is testable against closed-form λ values alone.

**Files:**
- Create: `qopt/traffic.py`
- Test: `tests/test_traffic.py`

**Interfaces:**
- Consumes: `qopt.exceptions.TopologyError` (Task 1).
- Produces: `solve_traffic(nodes, edges, arrival_rate, source, sink, *, tol=1e-12, max_iter=10_000) -> (dict[str, float], int)`. `nodes` is a sequence of station names (the unknowns); `edges` is a sequence of `(src, dst, probability)` triples that may reference `source` / `sink`; the return is `(name -> λ, iterations)`. Raises `TopologyError` when the cap is hit.

- [ ] **Step 1: Write the failing test** — `tests/test_traffic.py`

```python
import pytest

from qopt.exceptions import TopologyError
from qopt.traffic import solve_traffic

SRC = "src"
SNK = "snk"


def test_tandem_lambda_is_equal_throughout():
    lam, iterations = solve_traffic(
        ["a", "b"],
        [(SRC, "a", 1.0), ("a", "b", 1.0), ("b", SNK, 1.0)],
        2.0, SRC, SNK,
    )
    assert lam == {"a": 2.0, "b": 2.0}
    assert iterations >= 1


def test_branch_splits_lambda_by_probability():
    lam, _ = solve_traffic(
        ["a", "b"],
        [(SRC, "a", 0.3), (SRC, "b", 0.7), ("a", SNK, 1.0), ("b", SNK, 1.0)],
        10.0, SRC, SNK,
    )
    assert lam["a"] == pytest.approx(3.0, rel=1e-12)
    assert lam["b"] == pytest.approx(7.0, rel=1e-12)


def test_feedback_loop_amplifies_lambda():
    # lambda_a = lambda_0 + p * lambda_a  =>  lambda_a = lambda_0 / (1 - p)
    lam, _ = solve_traffic(
        ["a"],
        [(SRC, "a", 1.0), ("a", "a", 0.25), ("a", SNK, 0.75)],
        1.0, SRC, SNK,
    )
    assert lam["a"] == pytest.approx(1.0 / (1.0 - 0.25), rel=1e-9)


def test_mixed_network_topology_derives_the_documented_gammas():
    # Spec 4.1.1: the topology behind examples/mixed_network.py's hand-supplied gammas.
    lam, iterations = solve_traffic(
        ["mm1", "md1", "fj"],
        [
            (SRC, "mm1", 0.6), (SRC, "md1", 0.4),
            ("mm1", "fj", 0.5), ("mm1", SNK, 0.5),
            ("md1", "fj", 0.5), ("md1", SNK, 0.5),
            ("fj", SNK, 1.0),
        ],
        1.0, SRC, SNK,
    )
    # Bitwise, not approx: Task 3's regression test depends on these being exact.
    assert lam == {"mm1": 0.6, "md1": 0.4, "fj": 0.5}
    assert iterations == 3


def test_closed_subnetwork_hits_the_cap():
    # a -> b -> a with p = 1 each way and external inflow: lambda diverges.
    with pytest.raises(TopologyError, match="closed subnetwork"):
        solve_traffic(
            ["a", "b"],
            [(SRC, "a", 1.0), ("a", "b", 1.0), ("b", "a", 1.0)],
            1.0, SRC, SNK, max_iter=50,
        )


def test_no_stations_is_trivially_solved():
    lam, iterations = solve_traffic([], [], 1.0, SRC, SNK)
    assert lam == {}
    assert iterations == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_traffic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.traffic'`.

- [ ] **Step 3: Create `qopt/traffic.py`**

```python
"""Traffic equations: derive per-station arrival rates from a topology (spec 4)."""

from qopt.exceptions import TopologyError


def solve_traffic(nodes, edges, arrival_rate, source, sink, *, tol=1e-12, max_iter=10_000):
    """Solve lambda = lambda_ext + P^T lambda by fixed-point iteration from lambda = 0.

    Converges geometrically for any open chain, including branching and feedback cycles.

    Args:
        nodes: station names — the unknowns. `source` and `sink` are not among them.
        edges: (src, dst, probability) triples; endpoints may be `source` or `sink`.
        arrival_rate: exogenous lambda_0 entering at `source`.
        source, sink: endpoint sentinel names.
        tol: stop once max|delta lambda| < tol.
        max_iter: iteration cap. Hitting it means flow is trapped in a closed
            subnetwork, which is a structural error rather than slow convergence.

    Returns:
        (lambdas, iterations) — lambdas maps station name to arrival rate.

    Raises:
        TopologyError: the cap was reached without converging.
    """
    inflow = {n: [] for n in nodes}          # dst -> [(src, probability)]
    external = {n: 0.0 for n in nodes}       # dst -> lambda_0 * p(source -> dst)
    for src, dst, probability in edges:
        if dst == sink:
            continue                          # the sink is not an unknown
        if src == source:
            external[dst] += arrival_rate * probability
        else:
            inflow[dst].append((src, probability))

    lam = {n: 0.0 for n in nodes}
    delta = 0.0
    for iteration in range(1, max_iter + 1):
        nxt = {
            n: external[n] + sum(lam[s] * p for s, p in inflow[n])
            for n in nodes
        }
        delta = max((abs(nxt[n] - lam[n]) for n in nodes), default=0.0)
        lam = nxt
        if delta < tol:
            return lam, iteration

    raise TopologyError(
        f"traffic equations did not converge in {max_iter} iterations "
        f"(max|delta lambda| = {delta:g}); flow is trapped in a closed subnetwork"
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_traffic.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add qopt/traffic.py tests/test_traffic.py
git commit -m "feat: solve_traffic — traffic equations by fixed-point iteration"
```

---

## Task 3: `Route`, `Network`, structural validation, and the §4.1.1 regression

`Network.__init__` is the single write point for `γ`: it validates the structure, solves the traffic equations, and binds the result onto the stations. Every existing consumer (`allocate`, `min_feasible_budget`, `Station.zeta`) then works unchanged, so the emitted JSON and eq 21 cannot disagree about arrival rates.

`examples/mixed_network.py` is converted here, which turns the current README table into a **regression test** rather than a new baseline.

**Files:**
- Create: `qopt/network.py`
- Modify: `examples/mixed_network.py` (whole file)
- Modify: `qopt/__init__.py` (add `Network`, `Route`, `TopologyError` exports)
- Test: `tests/test_network.py`

**Interfaces:**
- Consumes: `solve_traffic` (Task 2); `Station.bind_gamma`, `Station.gamma` (Task 1); `TopologyError`.
- Produces: `Route(src, dst, probability=1.0)`; `Network(stations, routes, arrival_rate, *, name="qopt-network", arrival_scv=1.0, job_class="jobs")` with attributes `stations`, `routes`, `arrival_rate`, `arrival_scv`, `job_class`, `name`, `gammas`, `traffic_iterations`, class constants `SOURCE = "src"` / `SINK = "snk"`, and `__len__` / `__iter__` over stations. `examples.mixed_network.build_network()` now returns a `Network`.
- `to_model_dict` and `to_dot` land in Task 4, not here.

**Note on `__len__` / `__iter__`:** they exist so `tests/test_example.py` — which does `len(build_network())` and `zip(build_network(), ...)` — keeps passing **unmodified** after `build_network()` starts returning a `Network`. That is a hard constraint (spec §11 criterion 1), not a convenience.

**Note on station names:** the example's stations are renamed from `"ingest (M/M/1)"` / `"transform (M/D/1)"` / `"fork-join"` to `mm1` / `md1` / `fj`, because §4.2 forbids the spaces and parentheses that would otherwise become JSON node names and DOT identifiers. So the example's printed *labels* change while every printed *number* stays bit-identical. The spec §9 wording "output must remain byte-identical" is satisfied in the sense that matters and cannot be satisfied literally — the naming rules the same spec mandates forbid the old labels. The regression test therefore compares numbers.

- [ ] **Step 1: Write the failing test** — `tests/test_network.py`

```python
import math

import pytest

from qopt.exceptions import TopologyError
from qopt.network import Network, Route
from qopt.optimizer import Optimizer
from qopt.allocator import min_feasible_budget
from qopt.station import ForkJoinStation, GG1Station

SRC = Network.SOURCE
SNK = Network.SINK


def _stations():
    return [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


def _routes():
    return [
        Route(SRC, "mm1", 0.6), Route(SRC, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", SNK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", SNK, 0.5),
        Route("fj", SNK, 1.0),
    ]


def _network():
    return Network(_stations(), _routes(), arrival_rate=1.0, name="qopt-mixed-network")


# --- Route -------------------------------------------------------------------

def test_route_defaults_to_probability_one():
    assert Route("a", "b").probability == 1.0


def test_route_is_frozen():
    import dataclasses

    r = Route("a", "b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.probability = 0.5


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_route_validates_probability(bad):
    with pytest.raises(ValueError):
        Route("a", "b", bad)


# --- gamma derivation --------------------------------------------------------

def test_network_derives_gammas_and_binds_them():
    net = _network()
    assert net.gammas == {"mm1": 0.6, "md1": 0.4, "fj": 0.5}
    assert [st.gamma for st in net.stations] == [0.6, 0.4, 0.5]
    assert net.traffic_iterations == 3


def test_network_is_iterable_and_sized():
    net = _network()
    assert len(net) == 3
    assert [st.name for st in net] == ["mm1", "md1", "fj"]


def test_explicit_gamma_in_a_network_is_rejected():
    stations = _stations()
    stations[0] = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1")
    with pytest.raises(ValueError, match="explicit gamma"):
        Network(stations, _routes(), arrival_rate=1.0)


# --- 4.2 structural validation, one test per row -----------------------------

def test_empty_station_name_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="")]
    with pytest.raises(TopologyError, match="non-empty"):
        Network(stations, [Route(SRC, ""), Route("", SNK)], arrival_rate=1.0)


def test_duplicate_station_names_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    with pytest.raises(TopologyError, match="unique"):
        Network(stations, [Route(SRC, "a"), Route("a", SNK)], arrival_rate=1.0)


def test_double_underscore_in_name_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a__join")]
    with pytest.raises(TopologyError, match="__"):
        Network(stations, [Route(SRC, "a__join"), Route("a__join", SNK)], arrival_rate=1.0)


@pytest.mark.parametrize("reserved", [Network.SOURCE, Network.SINK])
def test_reserved_endpoint_names_rejected(reserved):
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name=reserved)]
    with pytest.raises(TopologyError, match="reserved"):
        Network(stations, [Route(SRC, reserved), Route(reserved, SNK)], arrival_rate=1.0)


def test_dangling_route_endpoint_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    with pytest.raises(TopologyError, match="not a station name"):
        Network(stations, [Route(SRC, "a"), Route("a", "typo")], arrival_rate=1.0)


def test_source_with_an_in_edge_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(SRC, "a"), Route("a", SRC, 0.5), Route("a", SNK, 0.5)]
    with pytest.raises(TopologyError, match="no in-edges"):
        Network(stations, routes, arrival_rate=1.0)


def test_sink_with_an_out_edge_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(SRC, "a"), Route("a", SNK), Route(SNK, "a")]
    with pytest.raises(TopologyError, match="no out-edges"):
        Network(stations, routes, arrival_rate=1.0)


def test_out_edge_probabilities_must_sum_to_one():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="b")]
    routes = [Route(SRC, "a", 0.5), Route(SRC, "b", 0.4),
              Route("a", SNK), Route("b", SNK)]
    with pytest.raises(TopologyError, match="sum to"):
        Network(stations, routes, arrival_rate=1.0)


def test_station_with_no_out_edge_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    with pytest.raises(TopologyError, match="no out-edge"):
        Network(stations, [Route(SRC, "a")], arrival_rate=1.0)


def test_unreachable_station_rejected():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="orphan")]
    routes = [Route(SRC, "a"), Route("a", SNK), Route("orphan", SNK)]
    with pytest.raises(TopologyError, match="unreachable from"):
        Network(stations, routes, arrival_rate=1.0)


def test_flow_black_hole_rejected():
    # 'hole' can be reached but can never reach the sink.
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a"),
                GG1Station.mm1(mu=1.0, c=1.0, name="hole")]
    routes = [Route(SRC, "a"), Route("a", "hole", 0.5), Route("a", SNK, 0.5),
              Route("hole", "hole")]
    with pytest.raises(TopologyError, match="unreachable from stations"):
        Network(stations, routes, arrival_rate=1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_arrival_rate_validated(bad):
    with pytest.raises(ValueError, match="arrival_rate"):
        Network(_stations(), _routes(), arrival_rate=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_arrival_scv_validated(bad):
    with pytest.raises(ValueError, match="arrival_scv"):
        Network(_stations(), _routes(), arrival_rate=1.0, arrival_scv=bad)


# --- the regression that matters --------------------------------------------

# Captured from the pre-change analytic run of examples/mixed_network.py at budget
# 6 * min_feasible_budget = 15.600000000000001. Compared bitwise on purpose: deriving
# gamma from the topology must not perturb a single float.
LEGACY_BUDGET = 15.600000000000001
LEGACY_S = [2.9601176145885644, 3.644844988735743, 3.017459891043565]
LEGACY_T = [0.4237076973701281, 0.2912706108409073, 0.45195507506074634]
LEGACY_ZETA = [1.0, 0.9451279819531168, 1.1377787740190126]
LEGACY_OBJECTIVE = 1.1669333832717816


def test_derived_gamma_reproduces_the_legacy_result_bitwise():
    net = _network()
    budget = 6 * min_feasible_budget(net.stations)
    assert budget == LEGACY_BUDGET
    assert min_feasible_budget(net.stations) == 2.6

    result = Optimizer(net.stations, budget=budget).run()
    assert result.converged
    assert result.iterations == 6
    assert result.capacities == LEGACY_S
    assert result.sojourn_times == LEGACY_T
    assert result.zeta == LEGACY_ZETA
    assert result.objective == LEGACY_OBJECTIVE


def test_example_build_network_returns_a_network_with_the_same_numbers():
    from examples.mixed_network import build_network, main

    net = build_network()
    assert isinstance(net, Network)
    assert [st.gamma for st in net] == [0.6, 0.4, 0.5]
    result = main()
    assert result.capacities == LEGACY_S
    assert result.objective == LEGACY_OBJECTIVE
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_network.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.network'`.

- [ ] **Step 3: Create `qopt/network.py`**

```python
"""Network: stations, probabilistic routing, and exogenous arrivals (spec 3, 4)."""

import math
from dataclasses import dataclass

from qopt.exceptions import TopologyError
from qopt.traffic import solve_traffic


@dataclass(frozen=True)
class Route:
    """One routing edge.

    `src` / `dst` rather than `from` / `to` because `from` is a Python keyword; the
    qsim `model` block spells them `from` / `to` (see Network.to_model_dict).
    """

    src: str
    dst: str
    probability: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.probability) or not 0.0 < self.probability <= 1.0:
            raise ValueError(
                f"probability must be a finite number in (0, 1], got {self.probability}"
            )


def _reachable(adjacency, start):
    """Names reachable from `start` (inclusive) through `adjacency`."""
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in adjacency.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


class Network:
    """A single open chain: stations plus routing plus one exogenous arrival stream.

    Construction validates the structure (spec 4.2), solves the traffic equations, and
    binds each station's derived gamma. That single write point is why the emitted JSON
    and eq 21 cannot disagree about arrival rates.
    """

    SOURCE = "src"
    SINK = "snk"

    def __init__(self, stations, routes, arrival_rate, *, name="qopt-network",
                 arrival_scv=1.0, job_class="jobs"):
        if not math.isfinite(arrival_rate) or arrival_rate <= 0:
            raise ValueError(
                f"arrival_rate must be a finite number > 0, got {arrival_rate}"
            )
        if not math.isfinite(arrival_scv) or arrival_scv < 0:
            raise ValueError(
                f"arrival_scv must be a finite number >= 0, got {arrival_scv}"
            )
        self.stations = list(stations)
        self.routes = list(routes)
        self.arrival_rate = arrival_rate
        self.arrival_scv = arrival_scv
        self.job_class = job_class
        self.name = name

        self._validate()
        self.gammas, self.traffic_iterations = solve_traffic(
            [st.name for st in self.stations],
            [(r.src, r.dst, r.probability) for r in self.routes],
            arrival_rate, self.SOURCE, self.SINK,
        )
        for st in self.stations:
            st.bind_gamma(self.gammas[st.name])

    def __len__(self):
        return len(self.stations)

    def __iter__(self):
        return iter(self.stations)

    def _validate(self):
        """Every row of spec 4.2. Structural failures are TopologyError."""
        names = [st.name for st in self.stations]
        for n in names:
            if not isinstance(n, str) or not n:
                raise TopologyError(
                    f"station names must be non-empty strings, got {n!r}"
                )
            if "__" in n:
                raise TopologyError(
                    f"station name {n!r} contains '__', which could collide with qsim's "
                    f"internal fork-join names (<node>__b0 / <node>__join)"
                )
            if n in (self.SOURCE, self.SINK):
                raise TopologyError(
                    f"station name {n!r} is reserved for the emitted source/sink node"
                )
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise TopologyError(
                f"station names must be unique, duplicated: {duplicated}"
            )

        known = set(names) | {self.SOURCE, self.SINK}
        for r in self.routes:
            for endpoint in (r.src, r.dst):
                if endpoint not in known:
                    raise TopologyError(
                        f"route endpoint {endpoint!r} is not a station name, "
                        f"{self.SOURCE!r}, or {self.SINK!r}"
                    )
            if r.dst == self.SOURCE:
                raise TopologyError(
                    f"{self.SOURCE!r} must have no in-edges, got {r}"
                )
            if r.src == self.SINK:
                raise TopologyError(
                    f"{self.SINK!r} must have no out-edges, got {r}"
                )

        out_edges = {}
        for r in self.routes:
            out_edges.setdefault(r.src, []).append(r)
        for src, routes in out_edges.items():
            total = math.fsum(r.probability for r in routes)
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise TopologyError(
                    f"out-edge probabilities from {src!r} sum to {total!r}, not 1.0"
                )
        if self.SOURCE not in out_edges:
            raise TopologyError(
                f"{self.SOURCE!r} has no out-edge; nothing enters the network"
            )
        for n in names:
            if n not in out_edges:
                raise TopologyError(
                    f"station {n!r} has no out-edge, so {self.SINK!r} is unreachable "
                    f"from it and flow is not conserved"
                )

        forward = {src: [r.dst for r in rs] for src, rs in out_edges.items()}
        reached = _reachable(forward, self.SOURCE)
        unreachable = [n for n in names if n not in reached]
        if unreachable:
            # An unreachable station gets lambda = 0 and is trivially "optimal".
            raise TopologyError(
                f"stations unreachable from {self.SOURCE!r}: {unreachable}"
            )

        backward = {}
        for r in self.routes:
            backward.setdefault(r.dst, []).append(r.src)
        reaches_sink = _reachable(backward, self.SINK)
        black_holes = [n for n in names if n not in reaches_sink]
        if black_holes:
            raise TopologyError(
                f"{self.SINK!r} is unreachable from stations: {black_holes}"
            )
```

- [ ] **Step 4: Convert `examples/mixed_network.py`** (whole file)

```python
"""Sample mixed network (spec 4.1.1): two single-server queues feeding a fork-join.

    source (lambda_0 = 1.0)
       +- 0.6 -> mm1 --+- 0.5 -> fj - 1.0 -> sink
       +- 0.4 -> md1 --+
                       +- 0.5 -> sink

The gammas (0.6, 0.4, 0.5) are the traffic-equation solution of this topology, so they
are derived here rather than hand-supplied. Every printed number is unchanged from the
version that supplied them by hand; only the station labels differ, because 4.2 requires
routing-safe names.
"""

from qopt import ForkJoinStation, GG1Station, Network, Optimizer, Route, min_feasible_budget


def build_network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6),
        Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5),
        Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5),
        Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def main():
    network = build_network()
    budget = 6 * min_feasible_budget(network.stations)
    result = Optimizer(network, budget=budget).run()

    print(f"budget = {budget:.4f}   converged = {result.converged} "
          f"in {result.iterations} iterations")
    print(f"{'station':22s} {'gamma':>8s} {'S*':>10s} {'E[T]':>10s} {'zeta':>10s}")
    for st, S, t, z in zip(
        network.stations, result.capacities, result.sojourn_times, result.zeta
    ):
        print(f"{st.name:22s} {st.gamma:8.4f} {S:10.4f} {t:10.4f} {z:10.4f}")
    print(f"objective (sum w*E[T]) = {result.objective:.6f}")
    return result


if __name__ == "__main__":
    main()
```

`Optimizer(network, budget=budget)` works only after Task 9. Until then, pass `network.stations`. **In this task, write `Optimizer(network.stations, budget=budget)`** and change it to `Optimizer(network, budget=budget)` in Task 9 Step 6.

- [ ] **Step 5: Add exports to `qopt/__init__.py`**

```python
from qopt.exceptions import (
    InfeasibleBudgetError,
    InstabilityError,
    QOptError,
    TopologyError,
)
from qopt.network import Network, Route
```

and add `"TopologyError"`, `"Network"`, `"Route"` to `__all__`.

- [ ] **Step 6: Run the new test, the example, and the full suite**

Run:
```bash
.venv/bin/python -m pytest tests/test_network.py -v
.venv/bin/python examples/mixed_network.py
.venv/bin/python -m pytest -q
```
Expected: `tests/test_network.py` all PASS. The example prints `budget = 15.6000`, `objective (sum w*E[T]) = 1.166933`, and the `S*` / `E[T]` / `zeta` columns `2.9601 3.6448 3.0175` / `0.4237 0.2913 0.4520` / `1.0000 0.9451 1.1378`. Full suite green, with `tests/test_example.py` passing **unmodified** (its `len()` and `zip()` over the return value are why `Network` has `__len__` / `__iter__`).

- [ ] **Step 7: Commit**

```bash
git add qopt/network.py qopt/__init__.py examples/mixed_network.py tests/test_network.py
git commit -m "feat: Network with derived gamma; convert the mixed-network example"
```

---

## Task 4: `sim_node`, `to_model_dict`, `to_dot`, and the golden fixture

Stations own their qsim node fragment, so adding a station type stays a one-file change instead of growing an `isinstance` ladder in `spec.py`. The accepted cost is that `station.py` knows the qsim schema shape.

**Files:**
- Modify: `qopt/station.py` (module-level `distribution_dict`; class constants and `sim_node` on `Station`, `GG1Station`, `ForkJoinStation`)
- Modify: `qopt/network.py` (add `to_model_dict`, `to_dot`)
- Create: `tests/fixtures/qopt_mixed_network_request.json`
- Test: `tests/test_network_model_dict.py`

**Interfaces:**
- Consumes: `Network` (Task 3).
- Produces: `distribution_dict(rate, scv) -> dict`; `Station.SIM_MEASURE_TYPE = "response-time"`; `Station.sim_conservation_checked = True`; `Station.DOT_SHAPE = "box"`; abstract `Station.sim_node(S, job_class) -> dict`; `GG1Station.sim_node`; `ForkJoinStation.sim_node`, `ForkJoinStation.sim_conservation_checked = False`, `ForkJoinStation.DOT_SHAPE = "box3d"`; `Network.to_model_dict(S) -> dict`; `Network.to_dot() -> str`.

**Two contract facts to respect** (verified against `qsim-service` at `51a99c7`):
1. `SIM_MEASURE_TYPE` is a **base-class constant, not an abstract property**. Post qsim-service#7, `response-time` on a `fork-join` node *is* the fork-to-join sojourn, so the measure type does not vary by station type. Do not reintroduce it as a property, and never request the literal `fork-join-response-time` — that is a 400.
2. `distribution_dict` takes a **rate**, not a mean. Passing a rate keeps `{"type": "exponential", "rate": S*mu}` bit-exact; going through `1/mean` would risk a round-trip perturbation in the byte-compared golden fixture.

- [ ] **Step 1: Write the failing test** — `tests/test_network_model_dict.py`

```python
import json
import pathlib

import pytest

from qopt.network import Network, Route
from qopt.station import ForkJoinStation, GG1Station, Station, distribution_dict

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "qopt_mixed_network_request.json"


def _mixed_network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


# --- distribution emission (spec 5.2) ---------------------------------------

def test_distribution_exponential_when_scv_is_one():
    assert distribution_dict(3.0, 1.0) == {"type": "exponential", "rate": 3.0}


def test_distribution_deterministic_when_scv_is_zero():
    assert distribution_dict(4.0, 0.0) == {"type": "deterministic", "value": 0.25}


def test_distribution_moment_form_otherwise():
    assert distribution_dict(2.0, 1.5) == {"mean": 0.5, "scv": 1.5}


# --- station node fragments -------------------------------------------------

def test_measure_type_is_one_constant_for_every_station_type():
    assert Station.SIM_MEASURE_TYPE == "response-time"
    assert GG1Station.mm1(mu=1.0, c=1.0, name="a").SIM_MEASURE_TYPE == "response-time"
    fj = ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj")
    assert fj.SIM_MEASURE_TYPE == "response-time"


def test_forkjoin_is_exempt_from_the_conservation_check():
    # qsim-service#8: a fork-join node's throughput is the internal join station's number.
    assert GG1Station.mm1(mu=1.0, c=1.0, name="a").sim_conservation_checked is True
    assert ForkJoinStation(
        mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"
    ).sim_conservation_checked is False


def test_gg1_sim_node_emits_a_queue():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1")
    assert st.sim_node(3.0, "jobs") == {
        "name": "mm1", "type": "queue", "servers": 1, "scheduling": "fcfs",
        "capacity": None,
        "service": {"jobs": {"distribution": {"type": "exponential", "rate": 3.0}}},
    }


def test_md1_sim_node_emits_a_deterministic_service():
    st = GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="md1")
    node = st.sim_node(4.0, "jobs")
    assert node["service"]["jobs"]["distribution"] == {
        "type": "deterministic", "value": 0.25
    }


def test_forkjoin_sim_node_emits_both_branches_and_join_all():
    st = ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj")
    assert st.sim_node(5.0, "jobs") == {
        "name": "fj", "type": "fork-join",
        "branches": [
            {"service": {"jobs": {"distribution": {"type": "exponential", "rate": 5.0}}}},
            {"service": {"jobs": {"distribution": {"type": "exponential", "rate": 10.0}}}},
        ],
        "join": "all",
    }


def test_station_sim_node_is_abstract():
    from qopt.station import SingleServerStation

    assert getattr(Station.sim_node, "__isabstractmethod__", False)
    with pytest.raises(TypeError):
        SingleServerStation(gamma=0.5, mu=1.0, c=1.0)  # type: ignore[abstract]


# --- to_model_dict ----------------------------------------------------------

def test_to_model_dict_matches_the_golden_fixture_byte_for_byte():
    model = _mixed_network().to_model_dict([3.0, 4.0, 5.0])
    assert json.dumps(model, indent=2) + "\n" == FIXTURE.read_text()


def test_to_model_dict_node_order_is_source_stations_sink():
    model = _mixed_network().to_model_dict([3.0, 4.0, 5.0])
    assert [n["name"] for n in model["nodes"]] == ["src", "mm1", "md1", "fj", "snk"]


def test_to_model_dict_rejects_a_mismatched_capacity_vector():
    with pytest.raises(ValueError, match="length"):
        _mixed_network().to_model_dict([3.0, 4.0])


def test_to_model_dict_arrival_distribution_comes_from_the_network():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(Network.SOURCE, "a"), Route("a", Network.SINK)]
    net = Network(stations, routes, arrival_rate=2.0, arrival_scv=0.0)
    source = net.to_model_dict([3.0])["nodes"][0]
    # arrival_scv = 0 => deterministic inter-arrival times of 1/2.0, never a station's cov_a.
    assert source["arrivals"]["jobs"]["distribution"] == {
        "type": "deterministic", "value": 0.5
    }


def test_to_model_dict_uses_the_configured_job_class():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(Network.SOURCE, "a"), Route("a", Network.SINK)]
    net = Network(stations, routes, arrival_rate=1.0, job_class="web")
    model = net.to_model_dict([3.0])
    assert model["classes"] == [{"name": "web", "type": "open"}]
    assert "web" in model["routing"]
    assert "web" in model["nodes"][1]["service"]


def test_network_has_no_from_model_dict():
    # S is not recoverable from the emitted S*mu product, so a round trip is undefined.
    assert not hasattr(Network, "from_model_dict")


# --- to_dot -----------------------------------------------------------------

def test_to_dot_emits_every_node_and_edge():
    dot = _mixed_network().to_dot()
    assert dot.startswith('digraph "qopt-mixed-network" {')
    assert dot.rstrip().endswith("}")
    for name in ("src", "snk", "mm1", "md1", "fj"):
        assert f'"{name}"' in dot
    assert '"src" -> "mm1"' in dot
    assert '"fj" -> "snk"' in dot
    assert dot.count("->") == 7
    assert "box3d" in dot          # the fork-join station's shape
```

- [ ] **Step 2: Create the golden fixture** — `tests/fixtures/qopt_mixed_network_request.json`

The §4.1.1 topology at `S = (3.0, 4.0, 5.0)`: `mm1` → `exponential rate 3.0`, `md1` → `deterministic 0.25`, `fj` → branches `exponential 5.0` and `10.0`. Written exactly as `json.dumps(model, indent=2) + "\n"` produces it.

```json
{
  "name": "qopt-mixed-network",
  "classes": [
    {
      "name": "jobs",
      "type": "open"
    }
  ],
  "nodes": [
    {
      "name": "src",
      "type": "source",
      "arrivals": {
        "jobs": {
          "distribution": {
            "type": "exponential",
            "rate": 1.0
          }
        }
      }
    },
    {
      "name": "mm1",
      "type": "queue",
      "servers": 1,
      "scheduling": "fcfs",
      "capacity": null,
      "service": {
        "jobs": {
          "distribution": {
            "type": "exponential",
            "rate": 3.0
          }
        }
      }
    },
    {
      "name": "md1",
      "type": "queue",
      "servers": 1,
      "scheduling": "fcfs",
      "capacity": null,
      "service": {
        "jobs": {
          "distribution": {
            "type": "deterministic",
            "value": 0.25
          }
        }
      }
    },
    {
      "name": "fj",
      "type": "fork-join",
      "branches": [
        {
          "service": {
            "jobs": {
              "distribution": {
                "type": "exponential",
                "rate": 5.0
              }
            }
          }
        },
        {
          "service": {
            "jobs": {
              "distribution": {
                "type": "exponential",
                "rate": 10.0
              }
            }
          }
        }
      ],
      "join": "all"
    },
    {
      "name": "snk",
      "type": "sink"
    }
  ],
  "routing": {
    "jobs": [
      {
        "from": "src",
        "to": "mm1",
        "probability": 0.6
      },
      {
        "from": "src",
        "to": "md1",
        "probability": 0.4
      },
      {
        "from": "mm1",
        "to": "fj",
        "probability": 0.5
      },
      {
        "from": "mm1",
        "to": "snk",
        "probability": 0.5
      },
      {
        "from": "md1",
        "to": "fj",
        "probability": 0.5
      },
      {
        "from": "md1",
        "to": "snk",
        "probability": 0.5
      },
      {
        "from": "fj",
        "to": "snk",
        "probability": 1.0
      }
    ]
  }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_network_model_dict.py -v`
Expected: FAIL — `ImportError: cannot import name 'distribution_dict' from 'qopt.station'`.

- [ ] **Step 4: Add the station-side emission to `qopt/station.py`**

Module-level, after the imports:

```python
def distribution_dict(rate, scv):
    """qsim distribution fragment for a given rate (1/mean) and squared CV (spec 5.2).

    The same three-branch rule serves both service and inter-arrival distributions.
    Takes a rate rather than a mean so the exponential form stays bit-exact: the
    emitted `rate` is the caller's S*mu, not a value round-tripped through 1/mean.
    """
    if scv == 1.0:
        return {"type": "exponential", "rate": rate}
    if scv == 0.0:
        return {"type": "deterministic", "value": 1.0 / rate}
    return {"mean": 1.0 / rate, "scv": scv}
```

On `Station`, add the class constants right under the docstring and the abstract method next to `sojourn_time`:

```python
    # --- qsim facts a station carries at class level (spec 5.2) ---
    SIM_MEASURE_TYPE = "response-time"
    """Which qsim measure supplies E[T] for eq 22.

    Deliberately a constant rather than an abstract property: post qsim-service#7 a
    fork-join node's `response-time` *is* the fork-to-join sojourn, so no station type
    varies it. A hook every subclass implements identically is dead abstraction.
    """

    sim_conservation_checked = True
    """Is simulated throughput a valid independent witness on this station's gamma?"""

    DOT_SHAPE = "box"
    """Graphviz node shape used by Network.to_dot()."""

    @abstractmethod
    def sim_node(self, S, job_class):
        """This station's qsim node dict under capacity S (spec 5.2)."""
```

On `GG1Station`:

```python
    def sim_node(self, S, job_class):
        return {
            "name": self.name,
            "type": "queue",
            "servers": 1,
            "scheduling": "fcfs",
            "capacity": None,
            "service": {
                job_class: {"distribution": distribution_dict(S * self.mu, self.cov_s ** 2)}
            },
        }
```

On `ForkJoinStation`:

```python
    sim_conservation_checked = False   # qsim-service#8; delete this line when it lands
    DOT_SHAPE = "box3d"

    def sim_node(self, S, job_class):
        """Two branches at S*mu and S*r*mu joined on "all" — the shared-capacity semantics."""
        return {
            "name": self.name,
            "type": "fork-join",
            "branches": [
                {"service": {job_class: {
                    "distribution": distribution_dict(S * self.mu, 1.0)}}},
                {"service": {job_class: {
                    "distribution": distribution_dict(S * self.r * self.mu, 1.0)}}},
            ],
            "join": "all",
        }
```

Note the branches are exponential unconditionally: `t_ul` is built from `1/(mu - lam)` terms and therefore assumes exponential servers, so `ForkJoinStation` has no per-branch `cov_s` to honor (spec §8.1, §10).

- [ ] **Step 5: Add `to_model_dict` and `to_dot` to `qopt/network.py`**

```python
    def to_model_dict(self, S):
        """Exactly qsim's `model` block: topology plus per-station service at capacity S.

        The request envelope (seed / stopping / measures) belongs to qopt.qsim.spec, so
        this method owns the *model* vocabulary and knows nothing about the request.

        There is intentionally no inverse: the emitted service rate is the product S*mu,
        and S is not recoverable from it, so a round trip is not well-defined.
        """
        S = list(S)
        if len(S) != len(self.stations):
            raise ValueError(
                f"S has length {len(S)}, expected {len(self.stations)}"
            )
        nodes = [{
            "name": self.SOURCE,
            "type": "source",
            "arrivals": {self.job_class: {
                "distribution": distribution_dict(self.arrival_rate, self.arrival_scv)
            }},
        }]
        nodes.extend(
            st.sim_node(Si, self.job_class) for st, Si in zip(self.stations, S)
        )
        nodes.append({"name": self.SINK, "type": "sink"})
        return {
            "name": self.name,
            "classes": [{"name": self.job_class, "type": "open"}],
            "nodes": nodes,
            "routing": {self.job_class: [
                {"from": r.src, "to": r.dst, "probability": r.probability}
                for r in self.routes
            ]},
        }

    def to_dot(self):
        """Graphviz DOT for diagrams — a plain string emitter, no dependency."""
        lines = [
            f'digraph "{self.name}" {{',
            "  rankdir=LR;",
            f'  "{self.SOURCE}" [shape=circle, label="{self.SOURCE}'
            f'\\nlambda={self.arrival_rate:g}"];',
            f'  "{self.SINK}" [shape=doublecircle];',
        ]
        for st in self.stations:
            lines.append(
                f'  "{st.name}" [shape={st.DOT_SHAPE}, '
                f'label="{st.name}\\ngamma={st.gamma:g}"];'
            )
        for r in self.routes:
            label = "" if r.probability == 1.0 else f' [label="{r.probability:g}"]'
            lines.append(f'  "{r.src}" -> "{r.dst}"{label};')
        lines.append("}")
        return "\n".join(lines) + "\n"
```

Add `distribution_dict` to the imports at the top of `network.py`:

```python
from qopt.station import distribution_dict
```

- [ ] **Step 6: Run the test to verify it passes, then the full suite**

Run:
```bash
.venv/bin/python -m pytest tests/test_network_model_dict.py -v
.venv/bin/python -m pytest -q
```
Expected: `tests/test_network_model_dict.py` all PASS (the fixture comparison included). Full suite green.

- [ ] **Step 7: Commit**

```bash
git add qopt/station.py qopt/network.py tests/fixtures/qopt_mixed_network_request.json tests/test_network_model_dict.py
git commit -m "feat: station qsim node fragments, to_model_dict, to_dot, golden fixture"
```

---
## Task 5: The `Analyzer` seam

The one stated seam of this feature. `Optimizer.run()` currently pulls per station: `[st.zeta(Si) for st, Si in zip(...)]`. A simulation answers for every station in one run, so evaluation becomes vector-in / vector-out. `AnalyticAnalyzer` exists so both paths share the loop verbatim and comparisons are apples-to-apples.

**Files:**
- Create: `qopt/analyzer.py`
- Modify: `qopt/__init__.py` (export `AnalyticAnalyzer`, `Analyzer`, `Evaluation`)
- Test: `tests/test_analyzer.py`

**Interfaces:**
- Consumes: `Station.sojourn_time` (existing).
- Produces: `Evaluation(sojourn_times, ci=None, degraded=<list>, extras=<dict>)` dataclass; `Analyzer` ABC with class attribute `is_stochastic: bool` and abstract `evaluate(self, stations, S, *, fresh_seed=False) -> Evaluation`; `AnalyticAnalyzer()` with `is_stochastic = False`.

`fresh_seed` is on the base signature even though the analytic implementation ignores it, so the optimizer's final-evaluation step (§6.5) has one call shape for both analyzer kinds.

- [ ] **Step 1: Write the failing test** — `tests/test_analyzer.py`

```python
import pytest

from qopt.analyzer import AnalyticAnalyzer, Analyzer, Evaluation
from qopt.station import ForkJoinStation, GG1Station


def _stations():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


def test_evaluation_defaults_are_independent_per_instance():
    a = Evaluation(sojourn_times=[1.0])
    b = Evaluation(sojourn_times=[2.0])
    a.degraded.append("x")
    a.extras["k"] = 1
    assert b.degraded == []
    assert b.extras == {}
    assert a.ci is None


def test_analyzer_is_abstract():
    assert getattr(Analyzer.evaluate, "__isabstractmethod__", False)
    with pytest.raises(TypeError):
        Analyzer()  # type: ignore[abstract]


def test_analytic_analyzer_is_not_stochastic():
    assert AnalyticAnalyzer.is_stochastic is False
    assert AnalyticAnalyzer().is_stochastic is False
    assert isinstance(AnalyticAnalyzer(), Analyzer)


def test_analytic_analyzer_mirrors_sojourn_time_bitwise():
    stations = _stations()
    S = [2.5, 3.5, 3.0]
    ev = AnalyticAnalyzer().evaluate(stations, S)
    assert ev.sojourn_times == [st.sojourn_time(Si) for st, Si in zip(stations, S)]
    assert ev.ci is None
    assert ev.degraded == []
    assert ev.extras == {}


def test_analytic_analyzer_ignores_fresh_seed():
    stations = _stations()
    S = [2.5, 3.5, 3.0]
    assert (
        AnalyticAnalyzer().evaluate(stations, S, fresh_seed=True).sojourn_times
        == AnalyticAnalyzer().evaluate(stations, S).sojourn_times
    )


def test_analytic_analyzer_propagates_instability():
    from qopt.exceptions import InstabilityError

    stations = [GG1Station.mm1(gamma=1.0, mu=1.0, c=1.0, name="a")]
    with pytest.raises(InstabilityError):
        AnalyticAnalyzer().evaluate(stations, [1.0])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.analyzer'`.

- [ ] **Step 3: Create `qopt/analyzer.py`**

```python
"""The evaluation seam: network-level E[T], analytic or simulated (spec 2.1, 6.1)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Evaluation:
    """One network-level evaluation of E[T] at a given capacity vector.

    Fields:
        sojourn_times: E[T] per station, aligned to the station order.
        ci: (lower, upper) per station, or None on a deterministic path.
        degraded: audit strings — weak measures, gamma-conservation misses (spec 6.8).
        extras: diagnostics — system_response_time, throughput, seed, wallClockSeconds.
    """

    sojourn_times: list
    ci: list = None
    degraded: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


class Analyzer(ABC):
    """Supplies E[T] for every station from one capacity vector."""

    is_stochastic = False
    """Drives the optimizer's warm-start and damping defaults (spec 6.2)."""

    @abstractmethod
    def evaluate(self, stations, S, *, fresh_seed=False):
        """Return an Evaluation for `stations` at capacities `S`.

        `fresh_seed` asks a stochastic analyzer for an independently seeded run; a
        deterministic one ignores it. One call shape serves both (spec 6.5).
        """


class AnalyticAnalyzer(Analyzer):
    """Delegates to each station's own closed-form sojourn_time. No confidence intervals."""

    is_stochastic = False

    def evaluate(self, stations, S, *, fresh_seed=False):
        return Evaluation(
            sojourn_times=[st.sojourn_time(Si) for st, Si in zip(stations, S)],
            ci=None,
        )
```

- [ ] **Step 4: Export from `qopt/__init__.py`**

```python
from qopt.analyzer import AnalyticAnalyzer, Analyzer, Evaluation
```

and add `"Analyzer"`, `"AnalyticAnalyzer"`, `"Evaluation"` to `__all__`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_analyzer.py -v && .venv/bin/python -m pytest -q`
Expected: all PASS in the new file; full suite green.

- [ ] **Step 6: Commit**

```bash
git add qopt/analyzer.py qopt/__init__.py tests/test_analyzer.py
git commit -m "feat: Analyzer seam with Evaluation record and AnalyticAnalyzer"
```

---

## Task 6: `QsimClient`

The HTTP boundary. Transport is injectable so tests use a fake and users can supply `httpx`, retries, or auth; the default is stdlib `urllib.request`, which is what keeps `dependencies = []`.

**Files:**
- Create: `qopt/qsim/__init__.py`
- Create: `qopt/qsim/client.py`
- Create: `tests/conftest.py`
- Test: `tests/test_qsim_client.py`

**Interfaces:**
- Consumes: `SimulationTransportError`, `SimulationRequestError`, `SimulationEngineError` (Task 1).
- Produces: `DEFAULT_STOPPING`, `TIMEOUT_MARGIN_SECONDS = 10.0`, `urllib_transport(url, body, timeout) -> (int, bytes)`, `QsimClient(base_url, *, timeout=None, stopping=None, transport=None, preflight=False)` with attributes `base_url`, `stopping`, `timeout`, `transport` and methods `health() -> dict`, `post_simulate(request: dict) -> dict`.
- Also produces the shared test doubles in `tests/conftest.py`: `FakeTransport` and the `measure` / `sim_response` factory fixtures, used by tasks 7, 8, 9, and 11.

**Transport contract:** `transport(url, body, timeout) -> (status, body_bytes)`. `body is None` means GET, otherwise POST with `Content-Type: application/json`. One seam covers `/simulate` and `/health`.

**Error mapping** (qsim-service returns `{"error": str, "details": [str]}` on every failure): `400/405/413/422 → SimulationRequestError`; `5xx → SimulationEngineError`; any other status → `SimulationTransportError`; connection refused / timeout / DNS → `SimulationTransportError`; a 200 whose body will not parse → `SimulationEngineError`.

**Timeout coherence** (§7.3): the client timeout must exceed `stopping["maxWallClockSeconds"]` plus `TIMEOUT_MARGIN_SECONDS`, or the client kills runs the server would have completed. Validated at construction, not on first use. The default timeout is `maxWallClockSeconds + 2 * TIMEOUT_MARGIN_SECONDS` so it clears its own guard.

- [ ] **Step 1: Create `tests/conftest.py`** (shared doubles for tasks 6–11)

```python
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
```

- [ ] **Step 2: Write the failing test** — `tests/test_qsim_client.py`

```python
import json

import pytest

from conftest import FakeTransport
from qopt.exceptions import (
    SimulationEngineError,
    SimulationRequestError,
    SimulationTransportError,
)
from qopt.qsim.client import DEFAULT_STOPPING, TIMEOUT_MARGIN_SECONDS, QsimClient

OK_BODY = {"modelName": "m", "completed": True, "measures": []}


def _client(transport, **kwargs):
    return QsimClient("http://qsim.test/", transport=transport, **kwargs)


def test_default_stopping_and_timeout():
    client = _client(FakeTransport((200, OK_BODY)))
    assert client.stopping == DEFAULT_STOPPING
    assert client.stopping is not DEFAULT_STOPPING          # defensively copied
    wall = DEFAULT_STOPPING["maxWallClockSeconds"]
    assert client.timeout == wall + 2 * TIMEOUT_MARGIN_SECONDS


def test_base_url_trailing_slash_stripped():
    client = _client(FakeTransport((200, OK_BODY)))
    client.post_simulate({"model": {}})
    assert client.transport.calls[0][0] == "http://qsim.test/simulate"


def test_timeout_must_clear_the_wall_clock_plus_margin():
    with pytest.raises(ValueError, match="must exceed maxWallClockSeconds"):
        _client(FakeTransport((200, OK_BODY)),
                stopping={"maxWallClockSeconds": 120}, timeout=120.0)


def test_timeout_just_above_the_margin_is_accepted():
    client = _client(FakeTransport((200, OK_BODY)),
                     stopping={"maxWallClockSeconds": 120}, timeout=131.0)
    assert client.timeout == 131.0


def test_stopping_without_a_wall_clock_is_rejected():
    with pytest.raises(ValueError, match="maxWallClockSeconds"):
        _client(FakeTransport((200, OK_BODY)), stopping={"alpha": 0.05})


def test_post_simulate_returns_the_parsed_body_and_sends_json():
    transport = FakeTransport((200, OK_BODY))
    client = _client(transport)
    assert client.post_simulate({"model": {"name": "m"}}) == OK_BODY
    url, request, timeout = transport.calls[0]
    assert url == "http://qsim.test/simulate"
    assert request == {"model": {"name": "m"}}
    assert timeout == client.timeout


@pytest.mark.parametrize("status", [400, 405, 413, 422])
def test_client_errors_map_to_simulation_request_error(status):
    body = {"error": "unprocessable model", "details": ["probabilities do not sum to 1"]}
    client = _client(FakeTransport((status, body)))
    with pytest.raises(SimulationRequestError, match="probabilities do not sum to 1"):
        client.post_simulate({"model": {}})


def test_unsupported_measure_type_maps_to_request_error():
    # Requesting the literal 'fork-join-response-time' is a 400 (spec 5.3); qopt must
    # never emit it, and if it ever does the failure must be this exception.
    body = {"error": "invalid request",
            "details": ["unsupported measure type: fork-join-response-time"]}
    client = _client(FakeTransport((400, body)))
    with pytest.raises(SimulationRequestError, match="fork-join-response-time"):
        client.post_simulate({"measures": ["fork-join-response-time"]})


def test_server_error_maps_to_engine_error():
    body = {"error": "simulation engine error", "details": ["correlationId=abc"]}
    client = _client(FakeTransport((500, body)))
    with pytest.raises(SimulationEngineError, match="correlationId=abc"):
        client.post_simulate({"model": {}})


def test_unexpected_status_maps_to_transport_error():
    client = _client(FakeTransport((302, b"")))
    with pytest.raises(SimulationTransportError, match="unexpected HTTP 302"):
        client.post_simulate({"model": {}})


def test_unreadable_success_body_maps_to_engine_error():
    client = _client(FakeTransport((200, b"<html>not json</html>")))
    with pytest.raises(SimulationEngineError, match="unreadable response body"):
        client.post_simulate({"model": {}})


def test_non_json_error_body_still_produces_a_message():
    client = _client(FakeTransport((422, b"plain text failure")))
    with pytest.raises(SimulationRequestError, match="plain text failure"):
        client.post_simulate({"model": {}})


def test_health_is_a_get():
    transport = FakeTransport((200, OK_BODY))
    client = _client(transport)
    assert client.health() == {"status": "ok"}
    url, request, _ = transport.calls[0]
    assert url == "http://qsim.test/health"
    assert request is None


def test_health_failure_is_a_transport_error():
    transport = FakeTransport((200, OK_BODY), health=(503, {"error": "down"}))
    client = _client(transport)
    with pytest.raises(SimulationTransportError, match="503"):
        client.health()


def test_preflight_calls_health_at_construction():
    transport = FakeTransport((200, OK_BODY))
    _client(transport, preflight=True)
    assert transport.calls[0][0] == "http://qsim.test/health"


def test_preflight_failure_surfaces_immediately():
    transport = FakeTransport((200, OK_BODY), health=(503, {"error": "down"}))
    with pytest.raises(SimulationTransportError):
        _client(transport, preflight=True)


def test_transport_exceptions_are_not_swallowed():
    def broken(url, body, timeout):
        raise SimulationTransportError("connection refused")

    client = _client(broken)
    with pytest.raises(SimulationTransportError, match="connection refused"):
        client.post_simulate({"model": {}})
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qsim_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.qsim'`.

- [ ] **Step 4: Create `qopt/qsim/__init__.py`**

```python
"""qsim-service client: the HTTP/JSON boundary to the simulation engine.

qsim-service is GPL v2 because it links JMT in-process. qopt is Apache-2.0 and speaks
only HTTP/JSON to it, never importing or linking JMT code. That boundary is the
licensing firewall, so nothing in this subpackage may grow a runtime dependency.
"""
```

- [ ] **Step 5: Create `qopt/qsim/client.py`**

```python
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
```

- [ ] **Step 6: Run the test to verify it passes, and confirm zero dependencies**

Run:
```bash
.venv/bin/python -m pytest tests/test_qsim_client.py -v
grep -n 'dependencies' pyproject.toml
.venv/bin/python -m pytest -q
```
Expected: all PASS; `dependencies = []` unchanged; full suite green.

- [ ] **Step 7: Commit**

```bash
git add qopt/qsim/__init__.py qopt/qsim/client.py tests/conftest.py tests/test_qsim_client.py
git commit -m "feat: QsimClient with injectable transport and HTTP error mapping"
```

---

## Task 7: Request envelope and measure extraction

`spec.py` wraps the model with the seed / stopping / **closed three-measure list**; `measures.py` turns the response into `E[T]`, CIs, throughput, and quality flags.

**Files:**
- Create: `qopt/qsim/spec.py`
- Create: `qopt/qsim/measures.py`
- Test: `tests/test_qsim_spec.py`
- Test: `tests/test_qsim_measures.py`

**Interfaces:**
- Consumes: `Network.to_model_dict` (Task 4); `Station.SIM_MEASURE_TYPE`, `Station.sim_conservation_checked` (Task 4); `MeasureMissingError` (Task 1).
- Produces: `MEASURES = ("response-time", "system-response-time", "throughput")`; `build_request(network, S, *, seed, stopping, measures=MEASURES) -> dict`; `SYSTEM_STATION = ""`; `extract(response, stations, job_class) -> (sojourn_times, ci, degraded, extras)` where `extras["throughput"]` maps station name → `(mean, (lo, hi))` and `extras["system_response_time"]` is `(mean, (lo, hi))` or `None`.

**Why the measure list is mandatory and closed (§5.4):** `MeasureMapper` falls back to `DEFAULTS = [response-time, utilization, throughput, queue-length]` when `measures` is null or empty, and two of those four — `utilization` and `queue-length` — are *join-station* numbers on a fork-join node that come back with `success: true` and no warning. Omitting the list is a silent-wrong-answer path, not a harmless default. The list is closed because nothing outside these three enters eq 21, eq 22, the objective, or the fixed point.

**Missing-measure policy (§7.1), exactly:**

| Missing | Consequence |
|---|---|
| `response-time` for any station | `MeasureMissingError` — eq 22 has no input |
| `system-response-time` | `extras["system_response_time"] = None` + `RuntimeWarning` + a `degraded` entry. Also the signal that §5.3's `station: ""` inference was wrong |
| `throughput` for a conservation-checked station | `RuntimeWarning` + a `degraded` entry saying the check *could not run* — distinct from the check running and failing (Task 8) |

`SYSTEM_STATION = ""` is an **inference, not a verified fact**: `MeasureMapper` emits `referenceNode=""` and `SolutionsParser.domainStation` passes an empty name through, but no `qsim-service` fixture pins it and that repo's own spec example says `"system"`. Task 10's first live run settles it. Keep the constant named and commented so a one-line change fixes it.

- [ ] **Step 1: Write the failing tests** — `tests/test_qsim_spec.py`

```python
import pytest

from qopt.network import Network, Route
from qopt.qsim.client import DEFAULT_STOPPING
from qopt.qsim.spec import MEASURES, build_request
from qopt.station import GG1Station


def _network():
    stations = [GG1Station.mm1(mu=1.0, c=1.0, name="a")]
    routes = [Route(Network.SOURCE, "a"), Route("a", Network.SINK)]
    return Network(stations, routes, arrival_rate=1.0, name="one")


def test_measure_list_is_the_closed_three():
    # Pins spec 5.4: qsim substitutes DEFAULTS (two of which are join-station numbers
    # at a fork-join node) whenever `measures` is null or empty.
    assert MEASURES == ("response-time", "system-response-time", "throughput")


def test_build_request_always_sends_the_exact_measure_list():
    request = build_request(_network(), [3.0], seed=7, stopping=DEFAULT_STOPPING)
    assert request["measures"] == [
        "response-time", "system-response-time", "throughput"
    ]
    assert request["measures"]  # never empty


def test_build_request_wraps_the_model_block():
    network = _network()
    request = build_request(network, [3.0], seed=7, stopping=DEFAULT_STOPPING)
    assert request["model"] == network.to_model_dict([3.0])
    assert request["seed"] == 7
    assert request["stopping"] == DEFAULT_STOPPING
    assert request["stopping"] is not DEFAULT_STOPPING     # copied, not aliased
    assert set(request) == {"model", "stopping", "measures", "seed"}


def test_build_request_omits_seed_when_none():
    request = build_request(_network(), [3.0], seed=None, stopping=DEFAULT_STOPPING)
    assert "seed" not in request


def test_build_request_rejects_an_empty_measure_list():
    with pytest.raises(ValueError, match="non-empty"):
        build_request(_network(), [3.0], seed=1, stopping=DEFAULT_STOPPING, measures=())
```

and `tests/test_qsim_measures.py`

```python
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
    assert ci == [(0.41, 0.43), (0.28, 0.30)]
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_qsim_spec.py tests/test_qsim_measures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.qsim.spec'`.

- [ ] **Step 3: Create `qopt/qsim/spec.py`**

```python
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
```

- [ ] **Step 4: Create `qopt/qsim/measures.py`**

```python
"""Response to per-station E[T], CIs, throughput, and quality flags (spec 5.3, 7)."""

import warnings

from qopt.exceptions import MeasureMissingError

SYSTEM_STATION = ""
"""Station key that system-level measures come back under.

INFERRED, not verified (spec 5.3 gotcha 2): MeasureMapper emits referenceNode="" for
system measures and SolutionsParser.domainStation passes an empty name through, so ""
is what the response should carry — but no qsim-service fixture pins it, and that
repo's own spec example says "system". The first live integration run settles it. If it
is wrong the symptom is system_response_time is None plus a RuntimeWarning, and the fix
is this one line.
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_qsim_spec.py tests/test_qsim_measures.py -v && .venv/bin/python -m pytest -q`
Expected: all PASS in both files; full suite green.

- [ ] **Step 6: Commit**

```bash
git add qopt/qsim/spec.py qopt/qsim/measures.py tests/test_qsim_spec.py tests/test_qsim_measures.py
git commit -m "feat: request envelope with the closed measure list, plus measure extraction"
```

---

## Task 8: `SimulationAnalyzer` and the γ-conservation check

Assembles spec + client + measures into one `Analyzer`, adds the fail-fast stability pre-check, the seed policy, and the γ-conservation check.

**Files:**
- Create: `qopt/qsim/analyzer.py`
- Modify: `qopt/__init__.py` (export `QsimClient`, `SimulationAnalyzer`)
- Test: `tests/test_qsim_analyzer.py`

**Interfaces:**
- Consumes: `Analyzer`, `Evaluation` (Task 5); `build_request`, `extract` (Task 7); `QsimClient` (Task 6); `Station.sim_conservation_checked` (Task 4); `Station.gamma`, `Station.check_stable` (Task 1).
- Produces: `SimulationAnalyzer(network, client, *, seed=20260729, seed_policy="fixed", strict=False)` with `is_stochastic = True`, attribute `iteration`, and `evaluate(stations, S, *, fresh_seed=False) -> Evaluation`. Also `FRESH_SEED_OFFSET = 1_000_000`.

**The γ-conservation check (§6.8), and why it exists:** §4 makes `γ` derived — one write point in `Network.__init__`, then read by `allocate`, `min_feasible_budget`, and `zeta_from`. If the traffic solve and the emitted routing ever disagree, every number the optimizer produces is correct for a network that is *not* the one being simulated, and the closed-form topology tests cannot catch it: they check `solve_traffic` against analytic expectations, not against what `to_model_dict` actually serialized. Simulated throughput is an independent witness on exactly that, and it arrives on the same POST for free.

Policy is §7.2's, verbatim and unextended: a miss emits a `RuntimeWarning`, is recorded in `degraded`, and the run proceeds; `strict=True` raises `SimulationQualityError`. Not a hard failure, because a watchdog-truncated run can widen or bias throughput enough to miss legitimately, and halting an otherwise usable optimization for that would be worse than reporting it.

Fork-join stations are exempt because their throughput is the internal join station's number (qsim-service#8). Under `join: "all"` it *ought* to equal λ — one probe measured `0.985` against λ = 1.0 — but that is a single measurement, not a pinned upstream guarantee, and it is unverified beyond two branches.

**Seed policy (§6.5):** `"fixed"` uses common random numbers so `ΔS` reflects only real movement in `S`; `"vary"` adds the iteration count; `None` omits `seed` entirely. `fresh_seed=True` always uses `seed + FRESH_SEED_OFFSET` regardless of policy, and does **not** advance the iteration counter — the final evaluation is not a loop iteration.

- [ ] **Step 1: Write the failing test** — `tests/test_qsim_analyzer.py`

```python
import pytest

from conftest import FakeTransport
from qopt.exceptions import InstabilityError, SimulationQualityError
from qopt.network import Network, Route
from qopt.qsim.analyzer import FRESH_SEED_OFFSET, SimulationAnalyzer
from qopt.qsim.client import QsimClient
from qopt.station import ForkJoinStation, GG1Station


def _network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def _healthy(sim_response, **kwargs):
    """A response whose throughput brackets the derived gammas (0.6, 0.4, 0.5)."""
    return sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.6, "md1": 0.4, "fj": 0.5},
        system=1.16,
        **kwargs,
    )


def _analyzer(network, response, **kwargs):
    transport = FakeTransport((200, response))
    client = QsimClient("http://qsim.test", transport=transport)
    return SimulationAnalyzer(network, client, **kwargs), transport


S_OK = [3.0, 4.0, 5.0]


def test_is_stochastic():
    assert SimulationAnalyzer.is_stochastic is True


def test_evaluate_returns_sojourn_times_ci_and_extras(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.sojourn_times == [0.42, 0.29, 0.45]
    assert ev.ci == [(0.41, 0.43), (0.28, 0.30), (0.44, 0.46)]
    assert ev.degraded == []
    assert ev.extras["system_response_time"] == (1.16, (1.15, 1.17))
    assert ev.extras["seed"] == 20260729
    assert ev.extras["wallClockSeconds"] == 8.3
    assert len(transport.requests) == 1


def test_evaluate_sends_the_model_at_the_given_capacities(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    analyzer.evaluate(network.stations, S_OK)
    request = transport.requests[0]
    assert request["model"] == network.to_model_dict(S_OK)
    assert request["measures"] == [
        "response-time", "system-response-time", "throughput"
    ]
    assert request["stopping"]["maxWallClockSeconds"] == 120


def test_instability_is_caught_before_the_post(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    # mm1 needs S*mu > 0.6; 0.5 saturates it.
    with pytest.raises(InstabilityError):
        analyzer.evaluate(network.stations, [0.5, 4.0, 5.0])
    assert transport.requests == []          # no simulation time was spent


def test_fixed_seed_policy_repeats_one_seed(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response), seed=11)
    for _ in range(3):
        analyzer.evaluate(network.stations, S_OK)
    assert [r["seed"] for r in transport.requests] == [11, 11, 11]


def test_vary_seed_policy_advances_per_iteration(sim_response):
    network = _network()
    analyzer, transport = _analyzer(
        network, _healthy(sim_response), seed=11, seed_policy="vary"
    )
    for _ in range(3):
        analyzer.evaluate(network.stations, S_OK)
    assert [r["seed"] for r in transport.requests] == [11, 12, 13]


def test_none_seed_policy_omits_the_seed(sim_response):
    network = _network()
    analyzer, transport = _analyzer(
        network, _healthy(sim_response), seed_policy=None
    )
    analyzer.evaluate(network.stations, S_OK)
    assert "seed" not in transport.requests[0]


def test_fresh_seed_is_offset_and_does_not_advance_the_counter(sim_response):
    network = _network()
    analyzer, transport = _analyzer(network, _healthy(sim_response), seed=11,
                                    seed_policy="vary")
    analyzer.evaluate(network.stations, S_OK)              # seed 11, iteration -> 1
    analyzer.evaluate(network.stations, S_OK, fresh_seed=True)
    analyzer.evaluate(network.stations, S_OK)              # seed 12, not 13
    assert [r["seed"] for r in transport.requests] == [
        11, 11 + FRESH_SEED_OFFSET, 12
    ]


def test_invalid_seed_policy_rejected():
    network = _network()
    client = QsimClient("http://qsim.test", transport=FakeTransport((200, {})))
    with pytest.raises(ValueError, match="seed_policy"):
        SimulationAnalyzer(network, client, seed_policy="random")


def test_stations_must_be_the_networks_stations(sim_response):
    network = _network()
    analyzer, _ = _analyzer(network, _healthy(sim_response))
    other = [GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1")]
    with pytest.raises(ValueError, match="network stations"):
        analyzer.evaluate(other, [3.0])


# --- the gamma-conservation check (spec 6.8) --------------------------------

def test_conservation_miss_warns_and_records(sim_response):
    network = _network()
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.9, "md1": 0.4, "fj": 0.5},      # 0.9 CI excludes 0.6
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response)
    with pytest.warns(RuntimeWarning, match="excludes derived gamma"):
        ev = analyzer.evaluate(network.stations, S_OK)
    assert any("mm1" in d and "excludes derived gamma" in d for d in ev.degraded)
    assert ev.sojourn_times == [0.42, 0.29, 0.45]            # the run still proceeds


def test_conservation_miss_raises_under_strict(sim_response):
    network = _network()
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.9, "md1": 0.4, "fj": 0.5},
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response, strict=True)
    with pytest.raises(SimulationQualityError, match="excludes derived gamma"):
        analyzer.evaluate(network.stations, S_OK)


def test_forkjoin_throughput_never_flags_whatever_its_value(sim_response):
    network = _network()
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.6, "md1": 0.4, "fj": 99.0},     # nonsense at fj
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.degraded == []


def test_conservation_bracket_is_inclusive(sim_response):
    network = _network()
    # gamma sits exactly on the CI edge: mean 0.61, half-width 0.01 -> (0.60, 0.62).
    response = sim_response(
        sojourn={"mm1": 0.42, "md1": 0.29, "fj": 0.45},
        throughput={"mm1": 0.61, "md1": 0.4, "fj": 0.5},
        system=1.16,
    )
    analyzer, _ = _analyzer(network, response)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.degraded == []


def test_missing_throughput_bounds_are_treated_as_a_miss(sim_response):
    network = _network()
    response = _healthy(sim_response)
    for m in response["measures"]:
        if m["station"] == "mm1" and m["type"] == "throughput":
            m["lower"] = None
            m["upper"] = None
    analyzer, _ = _analyzer(network, response)
    with pytest.warns(RuntimeWarning, match="mm1"):
        ev = analyzer.evaluate(network.stations, S_OK)
    assert any("mm1" in d for d in ev.degraded)


def test_strict_also_raises_on_a_degraded_measure(sim_response):
    network = _network()
    analyzer, _ = _analyzer(network, _healthy(sim_response, completed=False),
                            strict=True)
    with pytest.raises(SimulationQualityError, match="completed=false"):
        analyzer.evaluate(network.stations, S_OK)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qsim_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.qsim.analyzer'`.

- [ ] **Step 3: Create `qopt/qsim/analyzer.py`**

```python
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
```

- [ ] **Step 4: Export from `qopt/__init__.py`**

```python
from qopt.qsim.analyzer import SimulationAnalyzer
from qopt.qsim.client import QsimClient
```

and add `"QsimClient"`, `"SimulationAnalyzer"` to `__all__`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_qsim_analyzer.py -v && .venv/bin/python -m pytest -q`
Expected: all PASS in the new file; full suite green.

- [ ] **Step 6: Commit**

```bash
git add qopt/qsim/analyzer.py qopt/__init__.py tests/test_qsim_analyzer.py
git commit -m "feat: SimulationAnalyzer with stability pre-check and gamma conservation"
```

---
## Task 9: `noise_floor`, the loop knobs, and the extended `Result`

The last core task. Wires the `Analyzer` seam into `Optimizer.run()`, adds the warm start, damping, CI-aware stopping, and the extended `Result` — while keeping `Optimizer(stations, budget)` bit-identical to today.

**Files:**
- Modify: `qopt/allocator.py` (append `noise_floor`)
- Modify: `qopt/optimizer.py` (whole file)
- Modify: `examples/mixed_network.py` (one line: `Optimizer(network, ...)`)
- Modify: `qopt/__init__.py` (export `noise_floor`)
- Test: `tests/test_noise_floor.py`
- Test: `tests/test_optimizer_loop.py`

**Interfaces:**
- Consumes: `AnalyticAnalyzer`, `Evaluation` (Task 5); `Network` (Task 3); `Station.zeta_from` (Task 1); `allocate` (existing).
- Produces: `noise_floor(stations, C, zeta_vec, dzeta) -> float`; `Optimizer(stations, budget, *, analyzer=None, tol=1e-9, max_iter=None, initial_zeta=None, damping=None, noise_kappa=1.0, final_evaluation=True, strict=False, warm_start=True)`; `Result` extended with `sojourn_ci=None`, `noise_floor=None`, `stop_reason="tol"`, `warm_start_iterations=0`, `degraded=<list>`, `system_response_time=None`, `sim_calls=0`.

**The perturbation direction matters (§6.4).** Eq 21 is invariant under *uniform positive scaling* of ζ: scaling every ζ by `k` multiplies both `numᵢ = √(ωᵢζᵢ/(cᵢµᵢ))` and `denom = Σ√(ω_jζ_jc_jµ_j⁻¹)` by `√k`, which cancels in the ratio. So perturbing all stations upward together is nearly a no-op, not a worst case. The worst case is **anti-correlated**: for each station `i`, evaluate `allocate` with component `i` at `ζᵢ+δζᵢ` and every other at `ζ_j−δζ_j`, plus the mirror. That is `2n` closed-form evaluations — negligible against one simulation run. (The same invariance is why an all-M/M/1 network converges in a single step: every ζ is identically 1, and uniform values are a fixed point of a scaling-invariant map.)

Perturbed ζ is clamped to `ZETA_FLOOR = 1e-12` before `allocate` takes its square root, because a wide CI can drive `ζ − δζ` negative.

**Backward compatibility, precisely.** `Optimizer(stations, budget)` defaults to `AnalyticAnalyzer` with `damping = 1.0`, `max_iter = 1000`, `ci = None` (so `noise_floor` is never computed and the stop threshold is plain `tol`). `AnalyticAnalyzer.evaluate` produces `st.sojourn_time(Si)` and `st.zeta_from(T, Si)` is `T * (Si*mu - gamma)` — the same float operations in the same order as today's `st.zeta(Si)`. Damping at exactly `1.0` takes an explicit branch that assigns `S_target` rather than blending, so no arithmetic is introduced. The result is bit-identical, which Step 4's test asserts with `==`.

**Two documented caveats (§6.6):**
- `seed_policy="fixed"` with `final_evaluation=False` reports metrics from the CRN sample path — biased numbers that look clean. This pairing emits a `RuntimeWarning` at construction.
- With `noise_kappa=0.0` against a stochastic analyzer, `converged=False` / `stop_reason="max_iter"` is the expected normal outcome, not a malfunction.

**`strict` division of labor:** `SimulationAnalyzer(strict=True)` fails fast, on the first degraded iteration. `Optimizer(strict=True)` raises at the *end* of the run if any degradation accumulated, so the whole audit trail is in the message. They compose; neither replaces the other.

- [ ] **Step 1: Write the failing noise-floor test** — `tests/test_noise_floor.py`

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_noise_floor.py -v`
Expected: FAIL — `ImportError: cannot import name 'noise_floor' from 'qopt.allocator'`.

- [ ] **Step 3: Append `noise_floor` to `qopt/allocator.py`**

```python
ZETA_FLOOR = 1e-12
"""Smallest zeta handed to `allocate`, which takes its square root."""


def noise_floor(stations, C, zeta_vec, dzeta):
    """How much of a capacity change is attributable to evaluation noise (spec 6.4).

    `allocate` is closed-form and pure, so this costs zero simulation calls: propagate
    each reported CI half-width h_i into zeta as dzeta_i = h_i * (S_i*mu_i - gamma_i),
    then measure the spread in S that a perturbation of that size can produce.

    The perturbation is ANTI-CORRELATED, not uniform. Eq 21 is invariant under uniform
    positive scaling of zeta, so moving every station up together is nearly a no-op
    rather than a worst case. For each station i we evaluate `allocate` with component i
    up and all others down, plus the mirror:

        noise_floor = max_i |S_i(zeta+) - S_i(zeta-)| / 2

    That is 2n closed-form evaluations, negligible against one simulation run.
    """
    n = len(zeta_vec)
    if n == 0 or all(d == 0.0 for d in dzeta):
        return 0.0
    worst = 0.0
    for i in range(n):
        up = [
            max(zeta_vec[k] + dzeta[k] if k == i else zeta_vec[k] - dzeta[k], ZETA_FLOOR)
            for k in range(n)
        ]
        down = [
            max(zeta_vec[k] - dzeta[k] if k == i else zeta_vec[k] + dzeta[k], ZETA_FLOOR)
            for k in range(n)
        ]
        S_up = allocate(stations, C, up)
        S_down = allocate(stations, C, down)
        worst = max(worst, abs(S_up[i] - S_down[i]) / 2.0)
    return worst
```

- [ ] **Step 4: Write the failing optimizer-loop test** — `tests/test_optimizer_loop.py`

```python
import math

import pytest

from qopt.allocator import min_feasible_budget
from qopt.analyzer import AnalyticAnalyzer, Analyzer, Evaluation
from qopt.network import Network, Route
from qopt.optimizer import Optimizer, Result
from qopt.station import ForkJoinStation, GG1Station


def _stations():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]


def _network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0)


class DeterministicFake(Analyzer):
    """Mirrors sojourn_time exactly, but declares itself stochastic.

    Lets the stochastic code path (warm start, damping, sim_calls, final evaluation) be
    exercised with zero randomness, so equivalence can be asserted bitwise.
    """

    is_stochastic = True

    def __init__(self, half_width=None):
        self.half_width = half_width
        self.calls = 0
        self.fresh_calls = 0

    def evaluate(self, stations, S, *, fresh_seed=False):
        self.calls += 1
        if fresh_seed:
            self.fresh_calls += 1
        T = [st.sojourn_time(Si) for st, Si in zip(stations, S)]
        ci = None
        if self.half_width is not None:
            ci = [(t - self.half_width, t + self.half_width) for t in T]
        return Evaluation(sojourn_times=T, ci=ci,
                          extras={"system_response_time": (sum(T), (0.0, 1.0))})


BUDGET = 15.600000000000001
LEGACY_S = [2.9601176145885644, 3.644844988735743, 3.017459891043565]
LEGACY_OBJECTIVE = 1.1669333832717816


# --- backward compatibility --------------------------------------------------

def test_default_construction_is_analytic_and_undamped():
    opt = Optimizer(_stations(), budget=BUDGET)
    assert isinstance(opt.analyzer, AnalyticAnalyzer)
    assert opt.damping == 1.0
    assert opt.max_iter == 1000


def test_analytic_defaults_reproduce_the_legacy_numbers_bitwise():
    result = Optimizer(_stations(), budget=BUDGET).run()
    assert result.capacities == LEGACY_S
    assert result.objective == LEGACY_OBJECTIVE
    assert result.iterations == 6
    assert result.converged is True
    assert result.stop_reason == "tol"
    assert result.sojourn_ci is None
    assert result.noise_floor is None
    assert result.warm_start_iterations == 0
    assert result.degraded == []
    assert result.sim_calls == 0


def test_result_new_fields_all_default():
    r = Result(capacities=[1.0], sojourn_times=[1.0], zeta=[1.0], objective=1.0,
               iterations=1, converged=True, residual=0.0)
    assert r.sojourn_ci is None
    assert r.noise_floor is None
    assert r.stop_reason == "tol"
    assert r.warm_start_iterations == 0
    assert r.degraded == []
    assert r.system_response_time is None
    assert r.sim_calls == 0


def test_result_degraded_default_is_per_instance():
    a = Result(capacities=[], sojourn_times=[], zeta=[], objective=0.0,
               iterations=0, converged=True, residual=0.0)
    b = Result(capacities=[], sojourn_times=[], zeta=[], objective=0.0,
               iterations=0, converged=True, residual=0.0)
    a.degraded.append("x")
    assert b.degraded == []


# --- Network as the first argument -------------------------------------------

def test_optimizer_accepts_a_network():
    network = _network()
    opt = Optimizer(network, budget=BUDGET)
    assert opt.network is network
    assert opt.stations == network.stations
    assert opt.run().capacities == LEGACY_S


def test_optimizer_still_accepts_a_bare_station_sequence():
    opt = Optimizer(_stations(), budget=BUDGET)
    assert opt.network is None
    assert len(opt.stations) == 3


# --- naive equivalence (spec 8, 6.6) ----------------------------------------

NAIVE_KNOBS = dict(warm_start=False, damping=1.0, noise_kappa=0.0, max_iter=1000)


def test_naive_equivalence_is_bit_identical():
    baseline = Optimizer(_stations(), budget=BUDGET).run()
    fake = DeterministicFake()
    simulated = Optimizer(
        _stations(), budget=BUDGET, analyzer=fake, **NAIVE_KNOBS
    ).run()
    assert simulated.capacities == baseline.capacities
    assert simulated.sojourn_times == baseline.sojourn_times
    assert simulated.zeta == baseline.zeta
    assert simulated.objective == baseline.objective
    assert simulated.iterations == baseline.iterations
    assert simulated.residual == baseline.residual
    assert simulated.converged == baseline.converged
    # One POST per iteration, plus the final evaluation (spec 6.3 cost model).
    assert simulated.sim_calls == simulated.iterations + 1
    assert fake.fresh_calls == 1


def test_naive_equivalence_without_a_final_evaluation():
    baseline = Optimizer(_stations(), budget=BUDGET).run()
    fake = DeterministicFake()
    simulated = Optimizer(
        _stations(), budget=BUDGET, analyzer=fake, final_evaluation=False, **NAIVE_KNOBS
    ).run()
    assert simulated.capacities == baseline.capacities
    assert simulated.iterations == baseline.iterations
    assert simulated.residual == baseline.residual
    assert simulated.sim_calls == simulated.iterations
    assert fake.fresh_calls == 0


# --- warm start --------------------------------------------------------------

def test_warm_start_costs_zero_simulation_calls_and_is_counted_separately():
    fake = DeterministicFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake, damping=1.0,
                       noise_kappa=0.0, max_iter=5).run()
    assert result.warm_start_iterations == 6        # the analytic pre-solve
    assert result.sim_calls == result.iterations + 1
    assert result.capacities == pytest.approx(LEGACY_S, rel=1e-9)


def test_warm_start_starts_the_loop_at_the_analytic_answer():
    fake = DeterministicFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake, damping=1.0,
                       noise_kappa=0.0, max_iter=20).run()
    # Already converged before the first simulated iteration, so it stops immediately.
    assert result.iterations == 1
    assert result.stop_reason == "tol"


def test_warm_start_off_skips_the_pre_solve():
    fake = DeterministicFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake,
                       **NAIVE_KNOBS).run()
    assert result.warm_start_iterations == 0


# --- damping -----------------------------------------------------------------

def test_stochastic_defaults_are_damped_and_capped():
    opt = Optimizer(_stations(), budget=BUDGET, analyzer=DeterministicFake())
    assert opt.damping == 0.5
    assert opt.max_iter == 20


def test_damping_slows_movement_but_reaches_the_same_point():
    result = Optimizer(_stations(), budget=BUDGET, analyzer=DeterministicFake(),
                       warm_start=False, damping=0.5, noise_kappa=0.0,
                       max_iter=500, tol=1e-12).run()
    assert result.capacities == pytest.approx(LEGACY_S, rel=1e-9)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_damping_validated(bad):
    with pytest.raises(ValueError, match="damping"):
        Optimizer(_stations(), budget=BUDGET, damping=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_noise_kappa_validated(bad):
    with pytest.raises(ValueError, match="noise_kappa"):
        Optimizer(_stations(), budget=BUDGET, noise_kappa=bad)


# --- CI-aware stopping (spec 6.4) -------------------------------------------

def test_stop_reason_flips_to_noise_floor_as_ci_widens():
    narrow = Optimizer(_stations(), budget=BUDGET,
                       analyzer=DeterministicFake(half_width=1e-15),
                       warm_start=False, damping=1.0, max_iter=200).run()
    assert narrow.stop_reason == "tol"

    wide = Optimizer(_stations(), budget=BUDGET,
                     analyzer=DeterministicFake(half_width=0.05),
                     warm_start=False, damping=1.0, max_iter=200).run()
    assert wide.stop_reason == "noise-floor"
    assert wide.noise_floor > wide.residual
    assert wide.converged is True


def test_kappa_zero_restores_naive_stopping():
    result = Optimizer(_stations(), budget=BUDGET,
                       analyzer=DeterministicFake(half_width=0.05),
                       warm_start=False, damping=1.0, noise_kappa=0.0,
                       max_iter=200).run()
    assert result.stop_reason == "tol"
    assert result.noise_floor is None


def test_ci_and_system_response_time_reach_the_result():
    result = Optimizer(_stations(), budget=BUDGET,
                       analyzer=DeterministicFake(half_width=0.01),
                       warm_start=False, damping=1.0, max_iter=200).run()
    assert len(result.sojourn_ci) == 3
    for (lo, hi), t in zip(result.sojourn_ci, result.sojourn_times):
        assert lo < t < hi
    assert result.system_response_time is not None


# --- degraded accounting and strict -----------------------------------------

class DegradingFake(DeterministicFake):
    def evaluate(self, stations, S, *, fresh_seed=False):
        ev = super().evaluate(stations, S, fresh_seed=fresh_seed)
        ev.degraded.append(f"call {self.calls}: synthetic degradation")
        return ev


def test_degraded_entries_accumulate_per_iteration():
    fake = DegradingFake()
    result = Optimizer(_stations(), budget=BUDGET, analyzer=fake, warm_start=False,
                       damping=1.0, noise_kappa=0.0, max_iter=1000).run()
    assert len(result.degraded) == fake.calls
    assert all("synthetic degradation" in d for d in result.degraded)


def test_strict_raises_at_the_end_with_the_whole_audit_trail():
    from qopt.exceptions import SimulationQualityError

    with pytest.raises(SimulationQualityError, match="synthetic degradation"):
        Optimizer(_stations(), budget=BUDGET, analyzer=DegradingFake(),
                  warm_start=False, damping=1.0, noise_kappa=0.0, strict=True,
                  max_iter=1000).run()


# --- caveat warnings (spec 6.6) ---------------------------------------------

class FixedSeedFake(DeterministicFake):
    seed_policy = "fixed"


def test_fixed_seed_without_a_final_evaluation_warns():
    with pytest.warns(RuntimeWarning, match="common random numbers"):
        Optimizer(_stations(), budget=BUDGET, analyzer=FixedSeedFake(),
                  final_evaluation=False)


def test_fixed_seed_with_a_final_evaluation_does_not_warn(recwarn):
    Optimizer(_stations(), budget=BUDGET, analyzer=FixedSeedFake())
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_max_iter_exhaustion_still_warns_and_reports_max_iter():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = Optimizer(_stations(), budget=BUDGET,
                           analyzer=DeterministicFake(), warm_start=False,
                           damping=1.0, noise_kappa=0.0, tol=0.0, max_iter=3).run()
    assert result.stop_reason == "max_iter"
    assert result.converged is False
    assert math.isfinite(result.residual)
```

- [ ] **Step 5: Rewrite `qopt/optimizer.py`** (whole file)

```python
"""Fixed-point optimization loop (paper Steps 0-5), analytic or simulation-backed."""

import math
import warnings
from dataclasses import dataclass, field

from qopt.allocator import allocate, min_feasible_budget, noise_floor
from qopt.analyzer import AnalyticAnalyzer
from qopt.exceptions import InfeasibleBudgetError, SimulationQualityError
from qopt.network import Network


@dataclass
class Result:
    """Outcome of an optimization run (lists aligned to the station order)."""

    capacities: list
    sojourn_times: list
    zeta: list
    objective: float
    iterations: int
    converged: bool
    residual: float  # final ||S_new - S||_inf; how close the last iterate came to tol

    # Simulation-path diagnostics. All defaulted, so analytic construction is unchanged.
    sojourn_ci: list = None            # per-station (lower, upper); None when analytic
    noise_floor: float = None          # final |delta S| attributable to noise (6.4)
    stop_reason: str = "tol"           # "tol" | "noise-floor" | "max_iter"
    warm_start_iterations: int = 0     # analytic iterations before the simulated phase
    degraded: list = field(default_factory=list)   # per-iteration quality audit (6.8, 7.2)
    system_response_time: object = None           # qsim diagnostic; not optimized
    sim_calls: int = 0                            # POSTs issued — the real cost meter


class Optimizer:
    """Drives the fixed-point iteration for the capacity allocation problem.

    Loop: allocate from an initial zeta guess, then repeatedly recompute zeta from the
    current capacities (eq 22) and re-allocate (eq 21) until the step falls below the
    stopping threshold or max_iter is reached.

    `Optimizer(stations, budget)` is bit-identical to the pre-simulation implementation:
    it defaults to AnalyticAnalyzer with damping 1.0 and max_iter 1000, and the
    CI-driven machinery stays inert because the analytic path reports no CI.
    """

    def __init__(self, stations, budget, *, analyzer=None, tol=1e-9, max_iter=None,
                 initial_zeta=None, damping=None, noise_kappa=1.0,
                 final_evaluation=True, strict=False, warm_start=True):
        if isinstance(stations, Network):
            self.network = stations
            self.stations = list(stations.stations)
        else:
            self.network = None
            self.stations = list(stations)
        self.budget = budget
        self.analyzer = AnalyticAnalyzer() if analyzer is None else analyzer
        self.tol = tol
        self.initial_zeta = initial_zeta
        self.final_evaluation = final_evaluation
        self.strict = strict
        self.warm_start = warm_start

        # Each simulated iteration is a full simulation run, so the caps differ by kind.
        stochastic = self.analyzer.is_stochastic
        self.max_iter = (20 if stochastic else 1000) if max_iter is None else max_iter
        self.damping = (0.5 if stochastic else 1.0) if damping is None else damping
        self.noise_kappa = noise_kappa

        if not math.isfinite(self.damping) or not 0.0 < self.damping <= 1.0:
            raise ValueError(
                f"damping must be a finite number in (0, 1], got {self.damping}"
            )
        if not math.isfinite(self.noise_kappa) or self.noise_kappa < 0:
            raise ValueError(
                f"noise_kappa must be a finite number >= 0, got {self.noise_kappa}"
            )

        if getattr(self.analyzer, "seed_policy", None) == "fixed" and not final_evaluation:
            warnings.warn(
                "seed_policy='fixed' with final_evaluation=False reports metrics from "
                "the common random numbers sample path: the loop converges crisply but "
                "the reported numbers are biased toward that one sample path. Set "
                "final_evaluation=True for an independently seeded final run (spec 6.5).",
                RuntimeWarning,
                stacklevel=2,
            )

    def run(self):
        stations = self.stations

        # Guard: budget must exceed the minimum needed for stability (eq 21 slack > 0).
        # `isfinite` first because NaN slips through every ordering comparison below.
        if not math.isfinite(self.budget):
            raise ValueError(f"budget must be a finite number, got {self.budget}")
        min_budget = min_feasible_budget(stations)
        if self.budget <= min_budget:
            raise InfeasibleBudgetError(
                f"budget {self.budget} <= minimum feasible {min_budget}"
            )

        # Guard: finite, strictly-positive initial zeta.
        if self.initial_zeta is None:
            zeta = [st.default_zeta for st in stations]
        else:
            zeta = list(self.initial_zeta)
            if len(zeta) != len(stations):
                raise ValueError("initial_zeta length must match number of stations")
        if not all(math.isfinite(z) and z > 0 for z in zeta):
            raise ValueError(
                f"initial zeta values must be finite and strictly positive, got {zeta}"
            )

        stochastic = self.analyzer.is_stochastic
        warm_start_iterations = 0
        if stochastic and self.warm_start:
            # The analytic pre-solve is deterministic and costs zero simulation calls,
            # so it is free and starts the expensive phase near the answer (spec 6.3).
            pre = Optimizer(
                stations, self.budget, tol=self.tol, initial_zeta=self.initial_zeta
            ).run()
            S = list(pre.capacities)
            warm_start_iterations = pre.iterations
        else:
            S = allocate(stations, self.budget, zeta)  # S^(1)

        degraded = []
        sim_calls = 0
        iterations = 0
        residual = math.inf
        floor = None
        stop_reason = "max_iter"
        evaluation = None

        for _ in range(self.max_iter):
            iterations += 1
            evaluation = self.analyzer.evaluate(stations, S)
            if stochastic:
                sim_calls += 1
            degraded.extend(evaluation.degraded)

            zeta = [
                st.zeta_from(T, Si)
                for st, T, Si in zip(stations, evaluation.sojourn_times, S)
            ]                                                    # eq 22
            S_target = allocate(stations, self.budget, zeta)      # eq 21

            floor = self._noise_floor(stations, S, zeta, evaluation.ci)
            if self.damping == 1.0:
                S_new = S_target       # explicit, so the analytic path adds no arithmetic
            else:
                theta = self.damping
                S_new = [
                    (1.0 - theta) * s + theta * t for s, t in zip(S, S_target)
                ]
            residual = max(abs(a - b) for a, b in zip(S_new, S))
            S = S_new

            threshold = self.tol
            if floor is not None:
                threshold = max(self.tol, self.noise_kappa * floor)
            if residual < threshold:
                stop_reason = (
                    "noise-floor"
                    if floor is not None and self.noise_kappa * floor > self.tol
                    else "tol"
                )
                break

        converged = stop_reason != "max_iter"
        if not converged:
            warnings.warn(
                f"Optimizer did not converge in {iterations} iterations "
                f"(max_iter={self.max_iter}, tol={self.tol}, final residual={residual:g}); "
                f"returned capacities are the last iterate and may be sub-optimal.",
                RuntimeWarning,
                stacklevel=2,
            )

        if stochastic:
            if self.final_evaluation or evaluation is None:
                # One more run at the converged S* with a fresh seed: those numbers are
                # the reported metrics, independent of the CRN sample path (spec 6.5).
                evaluation = self.analyzer.evaluate(stations, S, fresh_seed=True)
                sim_calls += 1
                degraded.extend(evaluation.degraded)
            # Otherwise the last loop iterate's numbers are reported as-is, which is what
            # final_evaluation=False asks for. They were measured at the pre-damping S.
        else:
            evaluation = self.analyzer.evaluate(stations, S)

        sojourn_times = list(evaluation.sojourn_times)
        zeta = [
            st.zeta_from(T, Si) for st, T, Si in zip(stations, sojourn_times, S)
        ]
        objective = sum(st.weight * T for st, T in zip(stations, sojourn_times))

        if self.strict and degraded:
            raise SimulationQualityError("; ".join(degraded))

        return Result(
            capacities=S,
            sojourn_times=sojourn_times,
            zeta=zeta,
            objective=objective,
            iterations=iterations,
            converged=converged,
            residual=residual,
            sojourn_ci=evaluation.ci,
            noise_floor=floor,
            stop_reason=stop_reason,
            warm_start_iterations=warm_start_iterations,
            degraded=degraded,
            system_response_time=evaluation.extras.get("system_response_time"),
            sim_calls=sim_calls,
        )

    def _noise_floor(self, stations, S, zeta, ci):
        """Propagate CI half-widths into zeta and measure the spread in S (spec 6.4)."""
        if ci is None or self.noise_kappa <= 0.0:
            return None
        dzeta = [
            0.5 * (upper - lower) * (Si * st.mu - st.gamma)
            for st, Si, (lower, upper) in zip(stations, S, ci)
        ]
        return noise_floor(stations, self.budget, zeta, dzeta)
```

- [ ] **Step 6: Point the example at the `Network` and export `noise_floor`**

In `examples/mixed_network.py`, change `Optimizer(network.stations, budget=budget)` to:

```python
    result = Optimizer(network, budget=budget).run()
```

In `qopt/__init__.py`:

```python
from qopt.allocator import allocate, min_feasible_budget, noise_floor
```

and add `"noise_floor"` to `__all__`.

- [ ] **Step 7: Run both new test files, the example, and the full suite**

Run:
```bash
.venv/bin/python -m pytest tests/test_noise_floor.py tests/test_optimizer_loop.py -v
.venv/bin/python examples/mixed_network.py
.venv/bin/python -m pytest -q
```
Expected: all PASS. The example prints `budget = 15.6000` and `objective (sum w*E[T]) = 1.166933` exactly as in Task 3. Full suite green, with all 48 original tests untouched.

- [ ] **Step 8: Commit**

```bash
git add qopt/allocator.py qopt/optimizer.py qopt/__init__.py examples/mixed_network.py tests/test_noise_floor.py tests/test_optimizer_loop.py
git commit -m "feat: noise-aware convergence, analyzer seam in the loop, extended Result"
```

---

## Task 10: The single-server simulated path end to end

Ships a working simulated path: a runnable example, the README update, and the first gated integration tests. Everything before this was unit-testable with no Java, no network, and no container; this is where the real service enters.

**Files:**
- Create: `examples/simulated_tandem.py`
- Create: `tests/test_integration_qsim.py`
- Modify: `README.md` (architecture diagram + Scope & limitations)
- Test: `tests/test_example_simulated.py` (create — asserts the example runs analytically without a service)

**Interfaces:**
- Consumes: everything from tasks 1–9.
- Produces: `examples.simulated_tandem.build_network() -> Network` and `main() -> Result | None`; the `QOPT_QSIM_URL` gating convention.

**Gating convention:** integration tests read `QOPT_QSIM_URL` and `pytest.skip` when it is unset, so the default suite stays fast and offline. Run them with `QOPT_QSIM_URL=http://localhost:8080 .venv/bin/python -m pytest tests/test_integration_qsim.py -v`. These are the only slow tests in the repo.

**Why a tandem example ships first (§9):** it isolates variability propagation with the fewest moving parts. A Poisson source feeds `M/D/1 → M/M/1` in series; the M/D/1's departure process is *not* Poisson, so the downstream station's true `cov_a ≠ 1` while the analytic path assumes the `cov_a` it was given. That divergence *is* variability propagation, demonstrated with no fork-join involved.

**The one open inference this task settles:** `measures.py` keys system-level measures on `station == ""`, inferred from `referenceNode=""` (spec §5.3 gotcha 2). No `qsim-service` fixture pins it and that repo's own spec example says `"system"`. The first live run here decides. If it is wrong, the symptom is `Result.system_response_time is None` plus a `RuntimeWarning` — and the fix is the one-line `SYSTEM_STATION` constant in `qopt/qsim/measures.py`. Record the outcome in the commit message either way.

- [ ] **Step 1: Write the failing tests** — `tests/test_integration_qsim.py`

```python
"""Integration tests against a live qsim-service.

Gated on QOPT_QSIM_URL and skipped by default, because they need the GPL service
running (typically its Docker image) and each takes seconds to minutes.

    QOPT_QSIM_URL=http://localhost:8080 .venv/bin/python -m pytest \
        tests/test_integration_qsim.py -v
"""

import os

import pytest

from qopt.network import Network, Route
from qopt.qsim.analyzer import SimulationAnalyzer
from qopt.qsim.client import QsimClient
from qopt.station import ForkJoinStation, GG1Station

QSIM_URL = os.environ.get("QOPT_QSIM_URL")

pytestmark = pytest.mark.skipif(
    not QSIM_URL, reason="set QOPT_QSIM_URL to run live qsim-service tests"
)

# Tight enough to be discriminating, loose enough to finish in seconds.
STOPPING = {
    "alpha": 0.05,
    "precision": 0.02,
    "minSamples": 50000,
    "maxSamples": 4000000,
    "maxWallClockSeconds": 180,
}


@pytest.fixture
def client():
    return QsimClient(QSIM_URL, stopping=STOPPING, preflight=True)


def test_health_responds(client):
    assert client.health()["status"] == "ok"


def test_mm1_simulated_ci_brackets_the_analytic_sojourn_time(client):
    """Spec 11 criterion 5: the actual validation of the idea, at one station."""
    station = GG1Station.mm1(mu=1.0, c=1.0, name="mm1")
    network = Network(
        [station],
        [Route(Network.SOURCE, "mm1"), Route("mm1", Network.SINK)],
        arrival_rate=1.0,
        name="mm1-bracket",
    )
    assert station.gamma == 1.0

    S = [2.0]                      # S*mu = 2.0, rho = 0.5
    analytic = station.sojourn_time(S[0])
    assert analytic == pytest.approx(1.0, rel=1e-12)      # 1/(S*mu - gamma)

    evaluation = SimulationAnalyzer(network, client).evaluate(network.stations, S)
    lower, upper = evaluation.ci[0]
    assert lower <= analytic <= upper, (
        f"simulated CI ({lower}, {upper}) does not bracket analytic {analytic}"
    )


def test_system_measure_key_inference_holds(client):
    """Settles spec 5.3 gotcha 2: is a system measure keyed on station ""?

    If this fails, change SYSTEM_STATION in qopt/qsim/measures.py to whatever the
    response actually carries and re-run.
    """
    station = GG1Station.mm1(mu=1.0, c=1.0, name="mm1")
    network = Network(
        [station],
        [Route(Network.SOURCE, "mm1"), Route("mm1", Network.SINK)],
        arrival_rate=1.0,
        name="system-measure-probe",
    )
    evaluation = SimulationAnalyzer(network, client).evaluate(network.stations, [2.0])
    assert evaluation.extras["system_response_time"] is not None, (
        "system-response-time did not come back under station '' — see "
        "qopt/qsim/measures.py::SYSTEM_STATION"
    )


def _mixed_network():
    """The spec 4.1.1 branching topology, whose derived gammas are (0.6, 0.4, 0.5)."""
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def test_gamma_conservation_holds_on_the_branching_network(client):
    """Spec 11 criterion 5b: solve_traffic and to_model_dict describe the same network.

    The branching topology is what makes this meaningful — a tandem chain would pass
    even with the source split wrong.
    """
    network = _mixed_network()
    assert network.gammas == {"mm1": 0.6, "md1": 0.4, "fj": 0.5}

    evaluation = SimulationAnalyzer(network, client).evaluate(
        network.stations, [3.0, 4.0, 5.0]
    )
    throughput = evaluation.extras["throughput"]
    for name, expected in (("mm1", 0.6), ("md1", 0.4)):
        mean, (lower, upper) = throughput[name]
        assert lower <= expected <= upper, (
            f"{name}: simulated throughput {mean} CI ({lower}, {upper}) "
            f"excludes derived gamma {expected}"
        )
    # No conservation degradation was recorded for the checked stations.
    assert not [d for d in evaluation.degraded if "excludes derived gamma" in d]


def test_optimizer_runs_against_the_live_service(client):
    """The whole loop, end to end: warm start, damped iterations, final fresh-seed run."""
    from qopt.allocator import min_feasible_budget
    from qopt.optimizer import Optimizer

    network = _mixed_network()
    budget = 6 * min_feasible_budget(network.stations)
    analyzer = SimulationAnalyzer(network, client)
    result = Optimizer(network, budget=budget, analyzer=analyzer, max_iter=6).run()

    assert result.sim_calls == result.iterations + 1
    assert result.warm_start_iterations > 0
    assert len(result.sojourn_ci) == 3
    assert result.stop_reason in ("tol", "noise-floor", "max_iter")
    for st, S in zip(network.stations, result.capacities):
        assert S * st.mu > st.gamma
    spent = sum(
        st.alloc_cost * S for st, S in zip(network.stations, result.capacities)
    )
    assert spent == pytest.approx(budget, rel=1e-9)
```

and `tests/test_example_simulated.py`

```python
def test_simulated_tandem_runs_analytically_without_a_service(monkeypatch):
    """The example must be runnable offline: it prints the analytic table and stops."""
    monkeypatch.delenv("QOPT_QSIM_URL", raising=False)
    from examples.simulated_tandem import build_network, main

    network = build_network()
    assert [st.name for st in network] == ["shape", "serve"]
    # A tandem chain carries lambda_0 through unchanged.
    assert [st.gamma for st in network] == [1.0, 1.0]

    result = main()
    assert result is not None
    assert result.sim_calls == 0            # no service, so the analytic path ran
    for st, S in zip(network, result.capacities):
        assert S * st.mu > st.gamma
```

- [ ] **Step 2: Run them to verify they fail (or skip)**

Run: `.venv/bin/python -m pytest tests/test_integration_qsim.py tests/test_example_simulated.py -v`
Expected: `test_integration_qsim.py` → **6 skipped** (`QOPT_QSIM_URL` unset — that is the correct default). `test_example_simulated.py` → FAIL with `ModuleNotFoundError: No module named 'examples.simulated_tandem'`.

- [ ] **Step 3: Create `examples/simulated_tandem.py`**

```python
"""Variability propagation with the fewest moving parts: M/D/1 -> M/M/1 in series.

A Poisson source at lambda_0 = 1.0 feeds a deterministic-service station, whose
departure process is NOT Poisson, which then feeds an exponential-service station:

    source (lambda_0 = 1.0) -> shape (M/D/1) -> serve (M/M/1) -> sink

The analytic path evaluates each station independently from the cov_a it was given, so
`serve` is analyzed as if its arrivals were Poisson. They are not: a deterministic
server smooths the stream it passes on. Simulating the whole network captures that
coupling, and the printed difference at `serve` is exactly the effect per-station
analysis cannot represent.

Set QOPT_QSIM_URL to compare against a live qsim-service; without it the example prints
the analytic table alone.
"""

import os

from qopt import (
    GG1Station,
    Network,
    Optimizer,
    QsimClient,
    Route,
    SimulationAnalyzer,
    min_feasible_budget,
)

BUDGET_MULTIPLE = 3.0


def build_network():
    stations = [
        GG1Station.md1(mu=1.0, c=1.0, name="shape"),
        GG1Station.mm1(mu=1.0, c=1.0, name="serve"),
    ]
    routes = [
        Route(Network.SOURCE, "shape"),
        Route("shape", "serve"),
        Route("serve", Network.SINK),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-simulated-tandem")


def _print_table(title, network, result):
    print(f"\n{title}")
    print(f"  budget = {result_budget(network):.4f}   stop_reason = {result.stop_reason}"
          f"   iterations = {result.iterations}   sim_calls = {result.sim_calls}")
    header = f"  {'station':10s} {'gamma':>7s} {'S*':>9s} {'E[T]':>9s} {'zeta':>9s}"
    if result.sojourn_ci is not None:
        header += f" {'E[T] 95% CI':>22s}"
    print(header)
    for i, (st, S, t, z) in enumerate(zip(
        network.stations, result.capacities, result.sojourn_times, result.zeta
    )):
        row = f"  {st.name:10s} {st.gamma:7.4f} {S:9.4f} {t:9.4f} {z:9.4f}"
        if result.sojourn_ci is not None:
            lower, upper = result.sojourn_ci[i]
            row += f"   ({lower:.4f}, {upper:.4f})"
        print(row)
    print(f"  objective (sum w*E[T]) = {result.objective:.6f}")


def result_budget(network):
    return BUDGET_MULTIPLE * min_feasible_budget(network.stations)


def main():
    print(__doc__.strip().split("\n\n")[0])

    network = build_network()
    budget = result_budget(network)
    analytic = Optimizer(network, budget=budget).run()
    _print_table("ANALYTIC (independent stations)", network, analytic)

    url = os.environ.get("QOPT_QSIM_URL")
    if not url:
        print("\nSet QOPT_QSIM_URL=http://localhost:8080 to add the simulated "
              "comparison. Analytic results only.")
        return analytic

    # A fresh Network: gamma is derived-only and cannot be rebound onto used stations.
    simulated_network = build_network()
    client = QsimClient(url, preflight=True)
    analyzer = SimulationAnalyzer(simulated_network, client)
    simulated = Optimizer(
        simulated_network, budget=budget, analyzer=analyzer
    ).run()
    _print_table("SIMULATED (whole network)", simulated_network, simulated)

    print("\nDIFFERENCE (simulated - analytic)")
    print(f"  {'station':10s} {'E[T] analytic':>14s} {'E[T] simulated':>15s} {'gap':>10s}")
    for st, a, s in zip(
        network.stations, analytic.sojourn_times, simulated.sojourn_times
    ):
        print(f"  {st.name:10s} {a:14.6f} {s:15.6f} {s - a:10.6f}")
    print("\n'shape' sees genuinely Poisson arrivals, so its cov_a = 1 is exact and the "
          "two paths should agree closely. 'serve' does not: its arrivals are the "
          "departures of a deterministic server, so any gap there is variability "
          "propagation, which per-station analysis cannot represent.")
    if simulated.degraded:
        print("\nDEGRADED")
        for entry in simulated.degraded:
            print(f"  - {entry}")
    return simulated


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the offline example test**

Run:
```bash
.venv/bin/python -m pytest tests/test_example_simulated.py -v
.venv/bin/python examples/simulated_tandem.py
```
Expected: PASS. The example prints the analytic table (`shape` and `serve` both at `gamma = 1.0000`) and the "Set QOPT_QSIM_URL" notice.

- [ ] **Step 5: Update `README.md`**

Two edits, both required by spec §9:

1. In the architecture block diagram, replace the dashed `future: simulation analyzer` box with the real path. Read the diagram first and match its existing box-drawing style; the new content is `qopt/analyzer.py` (the `Analyzer` seam) feeding either `AnalyticAnalyzer` or `qopt/qsim/` (`spec.py` → `client.py` → `measures.py`) over HTTP/JSON to `qsim-service`, and a `qopt/network.py` box supplying the topology and derived γ. Mark the `qsim-service` box as an external GPL v2 service reached only over HTTP.

2. In **Scope & limitations**, move network coupling from future work to supported:

```markdown
- **Network coupling is supported via simulation.** Per-station closed forms cannot
  capture how one station's departure process shapes the arrival variability of the
  stations downstream of it. `SimulationAnalyzer` obtains `E[T]` from a discrete-event
  simulation of the whole network (one `POST /simulate` per optimizer iteration) via
  [`qsim-service`](https://github.com/atantawi/qsim-service), so that coupling is
  captured directly. `qopt` speaks HTTP/JSON only and declares zero runtime
  dependencies; the service is GPL v2 and stays behind that boundary.
- **Single open chain only.** One customer chain enters from a source and departs to a
  sink, so every `γᵢ` is exogenously determined and fixed across iterations. Closed
  chains would make `λ` depend on `S` through throughput, moving eq 21's budget floor
  underneath the optimizer; multi-class networks need a per-class notion eq 21 does not
  have. Both are honest open limitations, not oversights.
- **Fork-join measures other than `response-time` are diagnostics only.** JMT defines
  just two fork-join region measures, so `utilization`, `queue-length`, and friends
  report join-station numbers at a fork-join node
  ([qsim-service#8](https://github.com/atantawi/qsim-service/issues/8)). Nothing
  outside `response-time` enters eq 22, so this constrains reporting, not results.
```

Also add a short **Simulated evaluation** usage block near the existing example section:

````markdown
### Simulated evaluation

```python
from qopt import (GG1Station, Network, Optimizer, QsimClient, Route,
                  SimulationAnalyzer, min_feasible_budget)

network = Network(
    [GG1Station.md1(mu=1.0, c=1.0, name="shape"),
     GG1Station.mm1(mu=1.0, c=1.0, name="serve")],
    [Route(Network.SOURCE, "shape"), Route("shape", "serve"),
     Route("serve", Network.SINK)],
    arrival_rate=1.0,
)                                        # gamma is derived from the topology
client = QsimClient("http://localhost:8080", preflight=True)
result = Optimizer(
    network,
    budget=3 * min_feasible_budget(network.stations),
    analyzer=SimulationAnalyzer(network, client),
).run()
print(result.capacities, result.sojourn_ci, result.sim_calls, result.stop_reason)
```

Runnable versions: `examples/simulated_tandem.py` and
`examples/simulated_mixed_network.py`. Both fall back to analytic-only output when
`QOPT_QSIM_URL` is unset.
````

- [ ] **Step 6: Run the full suite offline, then against a live service**

Run:
```bash
.venv/bin/python -m pytest -q
```
Expected: all pass, with the 6 tests in `tests/test_integration_qsim.py` **skipped** (`QOPT_QSIM_URL` unset). A skip there is the correct offline outcome, not a gap.

Then, with `qsim-service` running, in the background because it takes minutes:
```bash
QOPT_QSIM_URL=http://localhost:8080 .venv/bin/python -m pytest tests/test_integration_qsim.py -v
QOPT_QSIM_URL=http://localhost:8080 .venv/bin/python examples/simulated_tandem.py
```
Expected: 6 PASSED. If `test_system_measure_key_inference_holds` fails, fix `SYSTEM_STATION` in `qopt/qsim/measures.py` to the station name the response actually carries, re-run, and say so in the commit message.

- [ ] **Step 7: Commit**

```bash
git add examples/simulated_tandem.py tests/test_integration_qsim.py tests/test_example_simulated.py README.md
git commit -m "feat: simulated tandem example, live integration tests, README update"
```

State the live-run outcome in the commit body, including whether the `station: ""` inference for system measures held.

---

## Task 11: Fork-join validation

Last by choice, not by dependency. `ForkJoinStation.sim_node` already emits the node (Task 4) and `measures.extract` already reads its `response-time` (Task 7) — nothing here is gated upstream. It is sequenced last because it is the one path whose verification needs a live service *and* a non-trivial oracle.

**Files:**
- Create: `examples/simulated_mixed_network.py`
- Modify: `tests/test_integration_qsim.py` (append the fork-join oracles)
- Test: `tests/test_example_simulated.py` (append an offline check for the new example)

**Interfaces:**
- Consumes: everything from tasks 1–10.
- Produces: `examples.simulated_mixed_network.build_network() -> Network` and `main() -> Result | None`.

**The two oracles, and why each takes the shape it does (§8.2):**

| Oracle | Assertion | Why that shape |
|---|---|---|
| `system-response-time` in a fork-join-only network | **equality to `1e-9`** | Both numbers come from the same sample path, so it is an *identity*, not a statistical bracket. CI width is irrelevant; tightening or loosening precision cannot change the verdict. The sharpest available guard against a re-regression to join-anchoring, which read `0.0987` where the identity gives `0.2885` |
| `t_ul` at a **symmetric** `r = 1` fork-join | **CI brackets `t_ul`** | `t_ul` is exact for equal branch rates (verified: it reproduces `(12−ρ)/(8(µ−λ))` to the last bit), so the only discrepancy is sampling noise. Tighter precision strictly strengthens the test |
| max branch `E[T]` | **`sim >= bound`** | Rigorous for any branch configuration; a cheap always-true assertion, folded into both fork-join tests |

**The heterogeneous case is deliberately NOT an acceptance criterion.** A probe at λ = 1.0 with branch rates `µ = (5, 10)` gives `t_ul = 0.282906` against a simulated `0.288451` — a 1.9% gap. A bracket test there is self-defeating: `t_ul` is exact only for equal rates, so the gap is genuine approximation bias, not noise. Tighten the precision target below 1.9% to make the comparison discriminating and `t_ul` falls *outside* the CI, failing a correct run; leave it loose enough to bracket and the CI half-width exceeds the effect being measured, so the test passes regardless. **Do not re-add it as an assertion.** It belongs in the example's printed comparison, which is where this task puts it.

- [ ] **Step 1: Append the failing fork-join oracles to `tests/test_integration_qsim.py`**

```python
# --- fork-join oracles (spec 8.2, 11 criterion 5a) ---------------------------

FJ_STOPPING = dict(STOPPING, precision=0.01, minSamples=200000, maxWallClockSeconds=600)


@pytest.fixture
def fj_client():
    """A tighter, longer-running client: the fork-join oracles need sharper CIs."""
    return QsimClient(QSIM_URL, stopping=FJ_STOPPING, preflight=True)


def _fork_join_only_network(*, r, mu=1.0, arrival_rate=1.0, name="fj-only"):
    """src -> fj -> snk, where the fork-join station is the only service in the network."""
    station = ForkJoinStation(mu=mu, r=r, c1=1.0, c2=1.0, name="fj")
    return Network(
        [station],
        [Route(Network.SOURCE, "fj"), Route("fj", Network.SINK)],
        arrival_rate=arrival_rate,
        name=name,
    )


def _branch_lower_bound(station, S):
    """The slower branch's own M/M/1 mean — a rigorous lower bound on the FJ sojourn."""
    rates = (S * station.mu, S * station.r * station.mu)
    assert min(rates) > station.gamma, "branch saturated"
    return max(1.0 / (rate - station.gamma) for rate in rates)


def test_forkjoin_response_time_equals_system_response_time(fj_client):
    """Criterion 5a(i): an identity, because both come from the same sample path.

    Also the sharpest guard against a regression to join-anchoring, which measured
    0.0987 on a network where the identity gives 0.2885.
    """
    network = _fork_join_only_network(r=2.0, name="fj-identity")
    station = network.stations[0]
    assert station.gamma == 1.0

    S = [5.0]                       # branches at 5.0 and 10.0
    evaluation = SimulationAnalyzer(network, fj_client).evaluate(network.stations, S)
    fj_response_time = evaluation.sojourn_times[0]
    system, _ = evaluation.extras["system_response_time"]

    assert system == pytest.approx(fj_response_time, abs=1e-9), (
        f"fork-join response-time {fj_response_time} != system-response-time {system}; "
        f"the measure is probably anchored on the join station again"
    )
    assert fj_response_time >= _branch_lower_bound(station, S[0])


def test_symmetric_forkjoin_ci_brackets_t_ul(fj_client):
    """Criterion 5a(ii): r = 1 is where t_ul is exact, so bracketing is the right shape."""
    from qopt.forkjoin_approx import t_ul

    network = _fork_join_only_network(r=1.0, name="fj-symmetric")
    station = network.stations[0]

    S = [4.0]                       # both branches at 4.0, rho = 0.25
    expected = t_ul(station.gamma, S[0] * station.mu, S[0] * station.r * station.mu)
    # t_ul is exact for equal rates: (12 - rho) / (8 * (mu - lambda)).
    rho = station.gamma / (S[0] * station.mu)
    assert expected == pytest.approx(
        (12.0 - rho) / (8.0 * (S[0] * station.mu - station.gamma)), rel=1e-12
    )

    evaluation = SimulationAnalyzer(network, fj_client).evaluate(network.stations, S)
    simulated = evaluation.sojourn_times[0]
    lower, upper = evaluation.ci[0]
    assert lower <= expected <= upper, (
        f"simulated CI ({lower}, {upper}) does not bracket the exact t_ul {expected}"
    )
    assert simulated >= _branch_lower_bound(station, S[0])


def test_unsupported_measure_literal_is_rejected_by_the_live_service(client):
    """Pins spec 5.3: 'fork-join-response-time' is not a type, and qopt never emits it."""
    from qopt.exceptions import SimulationRequestError
    from qopt.qsim.spec import build_request

    network = _fork_join_only_network(r=2.0, name="fj-bad-measure")
    request = build_request(
        network, [5.0], seed=1, stopping=STOPPING,
        measures=("fork-join-response-time",),
    )
    with pytest.raises(SimulationRequestError):
        client.post_simulate(request)
```

and append to `tests/test_example_simulated.py`:

```python
def test_simulated_mixed_network_runs_analytically_without_a_service(monkeypatch):
    monkeypatch.delenv("QOPT_QSIM_URL", raising=False)
    from examples.simulated_mixed_network import build_network, main

    network = build_network()
    assert [st.name for st in network] == ["mm1", "md1", "fj"]
    assert [st.gamma for st in network] == [0.6, 0.4, 0.5]

    result = main()
    assert result is not None
    assert result.sim_calls == 0
    # Same analytic answer as examples/mixed_network.py, bitwise.
    assert result.capacities == [
        2.9601176145885644, 3.644844988735743, 3.017459891043565
    ]
    assert result.objective == 1.1669333832717816
```

- [ ] **Step 2: Run to verify they fail (or skip)**

Run: `.venv/bin/python -m pytest tests/test_integration_qsim.py tests/test_example_simulated.py -v`
Expected: integration tests → **9 skipped**. `test_simulated_mixed_network_runs_analytically_without_a_service` → FAIL with `ModuleNotFoundError: No module named 'examples.simulated_mixed_network'`.

- [ ] **Step 3: Create `examples/simulated_mixed_network.py`**

```python
"""The spec 4.1.1 network solved analytically and by simulation, side by side.

    source (lambda_0 = 1.0)
       +- 0.6 -> mm1 --+- 0.5 -> fj - 1.0 -> sink
       +- 0.4 -> md1 --+
                       +- 0.5 -> sink

gamma is DERIVED from this topology on both paths, so the printed difference is
attributable to variability propagation alone rather than to differing arrival rates.

A prediction, stated up front and then demonstrated:

  * mm1 and md1 receive a Poisson stream split by Bernoulli probabilities, which is
    still Poisson. Their cov_a = 1 is exactly right, so analytic and simulated should
    agree closely.
  * fj receives a thinned SUPERPOSITION of two departure streams, which is not Poisson
    — yet t_ul takes a Poisson arrival rate. So fj is precisely where the analytic
    approximation is unjustified and where simulation should visibly diverge.

This example also prints the HETEROGENEOUS t_ul-vs-simulation comparison, with the
simulated CI alongside each number so a reader can judge whether a gap is approximation
bias or sampling noise. It is printed and discussed rather than asserted: t_ul is exact
only for equal branch rates, so the heterogeneous gap is genuine bias, and any bracket
test on it either fails a correct run (tight precision) or passes regardless (loose
precision). See spec 8.2.

Set QOPT_QSIM_URL to add the simulated columns; without it the analytic table is printed
alone.
"""

import os

from qopt import (
    ForkJoinStation,
    GG1Station,
    Network,
    Optimizer,
    QsimClient,
    Route,
    SimulationAnalyzer,
    min_feasible_budget,
)
from qopt.forkjoin_approx import t_ul

BUDGET_MULTIPLE = 6.0


def build_network():
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1"),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj"),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6),
        Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5),
        Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5),
        Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def _print_table(title, network, result):
    print(f"\n{title}")
    print(f"  stop_reason = {result.stop_reason}   iterations = {result.iterations}"
          f"   sim_calls = {result.sim_calls}"
          f"   warm_start_iterations = {result.warm_start_iterations}")
    header = f"  {'station':6s} {'gamma':>7s} {'S*':>9s} {'E[T]':>9s} {'zeta':>9s}"
    if result.sojourn_ci is not None:
        header += f" {'E[T] 95% CI':>22s}"
    print(header)
    for i, (st, S, t, z) in enumerate(zip(
        network.stations, result.capacities, result.sojourn_times, result.zeta
    )):
        row = f"  {st.name:6s} {st.gamma:7.4f} {S:9.4f} {t:9.4f} {z:9.4f}"
        if result.sojourn_ci is not None:
            lower, upper = result.sojourn_ci[i]
            row += f"   ({lower:.6f}, {upper:.6f})"
        print(row)
    print(f"  objective (sum w*E[T]) = {result.objective:.6f}")
    if result.system_response_time is not None:
        mean, (lower, upper) = result.system_response_time
        print(f"  system response time = {mean:.6f} "
              f"CI ({lower:.6f}, {upper:.6f})   [diagnostic, not optimized]")


def main():
    print(__doc__.strip().split("\n\n")[0])

    network = build_network()
    budget = BUDGET_MULTIPLE * min_feasible_budget(network.stations)
    print(f"\nbudget = {budget:.4f}   derived gamma = "
          f"{tuple(st.gamma for st in network)}")

    analytic = Optimizer(network, budget=budget).run()
    _print_table("ANALYTIC (independent stations)", network, analytic)

    url = os.environ.get("QOPT_QSIM_URL")
    if not url:
        print("\nSet QOPT_QSIM_URL=http://localhost:8080 to add the simulated "
              "comparison. Analytic results only.")
        return analytic

    # A fresh Network: gamma is derived-only and cannot be rebound onto used stations.
    simulated_network = build_network()
    client = QsimClient(url, stopping={
        "alpha": 0.05, "precision": 0.02, "minSamples": 100000,
        "maxSamples": 4000000, "maxWallClockSeconds": 300,
    }, preflight=True)
    simulated = Optimizer(
        simulated_network, budget=budget,
        analyzer=SimulationAnalyzer(simulated_network, client),
    ).run()
    _print_table("SIMULATED (whole network)", simulated_network, simulated)

    print("\nDIFFERENCE (simulated - analytic), at each path's own S*")
    print(f"  {'station':6s} {'E[T] analytic':>14s} {'E[T] simulated':>15s} "
          f"{'gap':>10s} {'gap %':>8s}")
    for st, a, s in zip(
        network.stations, analytic.sojourn_times, simulated.sojourn_times
    ):
        print(f"  {st.name:6s} {a:14.6f} {s:15.6f} {s - a:10.6f} "
              f"{100.0 * (s - a) / a:7.2f}%")

    # The heterogeneous t_ul cross-check: printed and discussed, never asserted (8.2).
    fj = simulated_network.stations[-1]
    S_fj = simulated.capacities[-1]
    approximation = t_ul(fj.gamma, S_fj * fj.mu, S_fj * fj.r * fj.mu)
    measured = simulated.sojourn_times[-1]
    lower, upper = simulated.sojourn_ci[-1]
    half_width = 0.5 * (upper - lower)
    gap = measured - approximation
    print(f"\nFORK-JOIN: t_ul vs simulation at S* = {S_fj:.6f} "
          f"(branch rates {S_fj * fj.mu:.4f} and {S_fj * fj.r * fj.mu:.4f}, r = {fj.r:g})")
    print(f"  t_ul (heterogeneous, approximate) = {approximation:.6f}")
    print(f"  simulated                         = {measured:.6f} "
          f"CI ({lower:.6f}, {upper:.6f}), half-width {half_width:.6f}")
    print(f"  gap                               = {gap:+.6f} "
          f"({100.0 * gap / approximation:+.2f}%)")
    if abs(gap) > half_width:
        print("  The gap exceeds the CI half-width, so it is approximation bias, not "
              "noise: t_ul is exact only for equal branch rates (r = 1), and r "
              f"= {fj.r:g} here.")
    else:
        print("  The gap is within the CI half-width, so this run cannot separate "
              "approximation bias from sampling noise. Tighten `precision` to see it.")
    print("  This comparison is deliberately not an acceptance test — see spec 8.2.")

    if simulated.degraded:
        print("\nDEGRADED")
        for entry in simulated.degraded:
            print(f"  - {entry}")
    return simulated


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the offline test and the example**

Run:
```bash
.venv/bin/python -m pytest tests/test_example_simulated.py -v
.venv/bin/python examples/simulated_mixed_network.py
```
Expected: PASS. The example prints `budget = 15.6000   derived gamma = (0.6, 0.4, 0.5)`, the analytic table matching `examples/mixed_network.py` digit for digit, and the "Set QOPT_QSIM_URL" notice.

- [ ] **Step 5: Run the full suite offline**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, with the 9 tests in `tests/test_integration_qsim.py` skipped. All 48 original tests still untouched.

- [ ] **Step 6: Run the fork-join oracles against a live service**

With `qsim-service` running, in the background — `FJ_STOPPING` allows up to 600s per run:
```bash
QOPT_QSIM_URL=http://localhost:8080 .venv/bin/python -m pytest tests/test_integration_qsim.py -v
QOPT_QSIM_URL=http://localhost:8080 .venv/bin/python examples/simulated_mixed_network.py
```
Expected: 9 PASSED. Expected qualitative outcome from the example, which is the whole point of it: close agreement at `mm1` and `md1`, visible divergence at `fj`.

If `test_forkjoin_response_time_equals_system_response_time` fails, do **not** loosen the tolerance — it is an identity, so a mismatch means the measure is anchored on the wrong station. Check `qsim-service`'s `MeasureMapper.FORK_JOIN_STATION` and `JsimgWriter.expandedMeasureNode` against what spec §5.3 records, and file upstream.

- [ ] **Step 7: Verify the zero-dependency and licensing invariants one final time**

Run:
```bash
grep -n 'dependencies' pyproject.toml
grep -rn 'import' qopt/ | grep -vE 'from qopt|import (math|json|os|warnings|urllib|abc|dataclasses)|from (math|abc|dataclasses|urllib)'
```
Expected: `dependencies = []`; the second command prints nothing — every import in `qopt/` is stdlib or intra-package.

- [ ] **Step 8: Commit**

```bash
git add examples/simulated_mixed_network.py tests/test_integration_qsim.py tests/test_example_simulated.py
git commit -m "feat: fork-join validation oracles and the mixed-network simulation example"
```

State the live fork-join run's numbers in the commit body: the identity's two values, the symmetric bracket, and the heterogeneous gap the example printed.

---

## Acceptance criteria checklist (spec §11)

Verify each against a real command before calling the feature done.

- [ ] **1.** `Optimizer(stations, budget)` is bit-identical and the existing suite passes unmodified — Task 9 `test_analytic_defaults_reproduce_the_legacy_numbers_bitwise`; `git diff --stat main -- tests/test_station.py tests/test_allocator.py tests/test_optimizer.py tests/test_forkjoin_approx.py tests/test_smoke.py tests/test_example.py` is empty.
- [ ] **2.** `Network` derives γ for tandem, branching, and feedback topologies — Task 2 `tests/test_traffic.py`.
- [ ] **2a.** The §4.1.1 topology derives `(0.6, 0.4, 0.5)` and reproduces the analytic table bit-for-bit at budget 15.6 — Task 3 `test_derived_gamma_reproduces_the_legacy_result_bitwise`.
- [ ] **3.** `to_model_dict(S)` reproduces the golden fixture byte-for-byte — Task 4 `test_to_model_dict_matches_the_golden_fixture_byte_for_byte`.
- [ ] **4.** Naive-equivalence passes — Task 9 `test_naive_equivalence_is_bit_identical`.
- [ ] **5.** Live M/M/1 simulated CI brackets `1/(Sµ − γ)` — Task 10 `test_mm1_simulated_ci_brackets_the_analytic_sojourn_time`.
- [ ] **5a.** `ForkJoinStation.sim_node` emits both heterogeneous branches with `join: "all"` and inherits `SIM_MEASURE_TYPE == "response-time"` (Task 4); live identity, symmetric bracket, and slower-branch bound (Task 11).
- [ ] **5b.** `build_request` emits exactly the three measures (Task 7), and live throughput brackets the derived γ at every conservation-checked station of the §4.1.1 network (Task 10). Miss warns and records; `strict=True` raises; fork-join exempt (Task 8).
- [ ] **6.** `qopt` still declares zero runtime dependencies — Task 11 Step 7.
- [ ] **7.** Every §4.2 validation row (Task 3) and every §7.1 exception branch (Tasks 6, 7, 8) has a test.

## Known open items carried into implementation

- **`station: ""` for system measures is an inference**, not a verified fact (§5.3 gotcha 2). `measures.py` keys on `""`; Task 10 Step 6 settles it on the first live run. If wrong, the symptom is `Result.system_response_time is None` plus a `RuntimeWarning`, Task 11's identity test fails outright, and the fix is the one-line `SYSTEM_STATION` constant.
- **Fork-join throughput is exempt from the γ-conservation check** pending [qsim-service#8](https://github.com/atantawi/qsim-service/issues/8). Under `join: "all"` it *ought* to equal λ and one probe measured `0.985` against λ = 1.0, but that is one measurement, not an upstream guarantee, and it is unverified beyond two branches. When #8 lands, deleting `ForkJoinStation.sim_conservation_checked = False` is the whole change.
- **`examples/mixed_network.py`'s station labels change** (`"ingest (M/M/1)"` → `mm1`, etc.) because §4.2 forbids spaces and parentheses in names that become JSON node names and DOT identifiers. Every printed *number* is unchanged; spec §9's "byte-identical output" cannot be met literally, and the regression test compares numbers.
- **Tracking issue [#2](https://github.com/atantawi/quantum-optimizer/issues/2)** covers this implementation. **PR [#3](https://github.com/atantawi/quantum-optimizer/pull/3)**'s body describes only its first two commits; the 2026-07-30 revision (closed measure list, γ-conservation check, oracle shapes) is not reflected there. Worth folding in before the implementation PR references it.


