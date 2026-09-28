# Measured Arrival `cov_a` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the simulated path, measure each slope-mode G/G/1 station's arrival coefficient of variation from qsim-service's `interarrival-time` measure and let `phi` use it for the evaluation that measured it.

**Architecture:** A `cov_a=None` keyword threads from `Evaluation.arrival_cov` through `Station.zeta_from` → `phi` → `GG1Station.sojourn_time`/`dT_dS`. It is passed only when a value was measured, so every path that measures nothing executes today's calls. `SimulationAnalyzer` requests the measure only when some station has `uses_measured_cov_a`, and the optimizer reads the override at four points (loop ζ, noise floor, shape check, final ζ/φ).

**Tech Stack:** Python ≥ 3.10, stdlib only (zero runtime dependencies), pytest. qsim-service (Java, GPL, reached only over HTTP) at `8e7358b`+ for live runs.

**Spec:** `docs/superpowers/specs/2026-09-25-measured-arrival-cov-design.md`

## Global Constraints

- Zero runtime dependencies; `qopt` imports only the stdlib.
- The level path and the analytic path stay **bit-for-bit** unchanged: the existing 554-test suite must stay green without edits to any existing assertion.
- A request with no `uses_measured_cov_a` station, or with `measure_cov_a=False`, is **byte-identical** to today's.
- `secondMoments` is **never** sent.
- The `cov_a` keyword is passed to a station method **only when it is not `None`**.
- The constructor `cov_a` is **never written** by this feature.
- Measuring needs qsim-service `8e7358b` or later; no silent retry without the measure.
- Run tests with `.venv/bin/python -m pytest`. Live tests skip without `QOPT_QSIM_URL`; run them with Bash `run_in_background`.
- Mutation checks: after reverting code to watch a test fail, `find . -name __pycache__ -prune -exec rm -rf {} +` before re-running, and again after restoring (a same-size edit can reuse stale bytecode).
- Never `git stash -u` in this repo (three local-only doc directories are untracked).
- Commit messages end with `Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>`.

## Review Focus

1. **A measurement that disappears on a later evaluation** — a station measured at iteration 1 and unusable at iteration 2 must be priced at its constructor `cov_a` at iteration 2, not at the stale value. Pinned in Task 4 (`test_a_missing_final_measurement_is_not_filled_from_the_loop`).
2. **A user subclass whose `phi(self, S)` predates the keyword** running on a stochastic path where *other* stations are measured — it must never receive `cov_a=`. Pinned in Task 4 (`test_an_old_signature_phi_override_survives_a_measured_run`).
3. **A pathologically small interarrival mean** (`1e-200`) — `mean**2` underflows to 0 and a naive `variance/mean**2` raises `ZeroDivisionError` mid-run. It must degrade to `None`. Pinned in Task 2's parametrized table.
4. **A measured `cov_a = 0` at a `cov_s = 0` station** (`k = 0`) — the analytic `k == 0` branch must be taken from the *override's* `k`, giving `phi == 1 - rho`. Pinned in Task 1 (`test_a_measured_zero_cov_takes_the_deterministic_branch`).
5. **An unrelated 400 while measuring** — the version-floor hint must appear only for the `unsupported measure type: 'interarrival-time'` rejection, never on another bad request. Pinned in Task 3 (`test_an_unrelated_request_error_gets_no_version_hint`).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `qopt/station.py` | modify | `reads_arrival_cov`, `uses_measured_cov_a`, `cov_a=` on `phi`/`zeta_from`, and on `GG1Station.sojourn_time`/`dT_dS` |
| `qopt/analyzer.py` | modify | `Evaluation.arrival_cov` |
| `qopt/qsim/spec.py` | modify | `COV_A_MEASURE` |
| `qopt/qsim/measures.py` | modify | `extract_arrival_cov` |
| `qopt/qsim/analyzer.py` | modify | `measure_cov_a`, trigger, version-floor hint |
| `qopt/optimizer.py` | modify | four plumbing points, `Result.arrival_cov`, shape-check message |
| `tests/test_arrival_cov.py` | create | station + optimizer tests for this feature |
| `tests/test_qsim_measures.py` | modify | extraction table |
| `tests/test_qsim_analyzer.py` | modify | request trigger, hint, strict |
| `tests/test_integration_qsim.py` | modify | two live tests |
| `README.md`, `qopt/zeta.py`, slope spec §8.6 | modify | docs ripple |

---

### Task 1: Station accepts a measured `cov_a`

**Files:**
- Modify: `qopt/station.py` (class attributes near line 71; `phi` at ~273; `zeta_from` at ~294; `GG1Station` at ~514-596)
- Create: `tests/test_arrival_cov.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Station.reads_arrival_cov: bool` (class attr, `False`); `GG1Station.reads_arrival_cov = True`.
  - `Station.uses_measured_cov_a -> bool` (property): `self.reads_arrival_cov and self._zeta_mode == ZETA_SLOPE`.
  - `Station.phi(S, *, cov_a=None) -> float`; `Station.zeta_from(T, S, *, cov_a=None) -> float`.
  - `GG1Station.sojourn_time(S, *, cov_a=None)`, `GG1Station.dT_dS(S, *, cov_a=None)`, `GG1Station.phi(S, *, cov_a=None)`.
  - `ForkJoinStation` inherits `reads_arrival_cov = False` and the ignoring base `phi`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arrival_cov.py`:

```python
"""Measured arrival cov_a: station arithmetic and optimizer plumbing.

Spec: docs/superpowers/specs/2026-09-25-measured-arrival-cov-design.md
"""

import math

import pytest

from qopt.station import ForkJoinStation, GG1Station
from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE


def _gg1(cov_a, mode=ZETA_SLOPE, cov_s=0.5):
    return GG1Station(0.6, 1.5, c=2.0, cov_a=cov_a, cov_s=cov_s, name="g", zeta_mode=mode)


S_GRID = [0.5, 0.8, 1.2, 3.0]          # gamma/mu = 0.4, so every S here is stable


# --- station arithmetic -------------------------------------------------------

@pytest.mark.parametrize("S", S_GRID)
@pytest.mark.parametrize("c", [0.0, 0.3, 1.0, 2.5])
def test_a_measured_cov_a_prices_exactly_like_a_station_built_with_it(S, c):
    # Exact, not approx: the override must run the SAME expression on the measured value,
    # so a station built with cov_a=c is its bit-for-bit reference.
    built, measured = _gg1(c), _gg1(1.7)
    assert measured.sojourn_time(S, cov_a=c) == built.sojourn_time(S)
    assert measured.dT_dS(S, cov_a=c) == built.dT_dS(S)
    assert measured.phi(S, cov_a=c) == built.phi(S)
    assert measured.zeta_from(0.9, S, cov_a=c) == built.zeta_from(0.9, S)


def test_the_override_moves_slope_zeta():
    # Anti-vacuity for the test above: if the override were ignored, 1.7 and 0.2 would
    # still agree there only by coincidence -- they must not.
    st = _gg1(1.7)
    for S in S_GRID:
        assert st.zeta_from(0.9, S, cov_a=0.2) != st.zeta_from(0.9, S)


def test_the_override_does_not_persist_on_the_station():
    st = _gg1(1.7)
    before = [st.phi(S) for S in S_GRID]
    for S in S_GRID:
        st.phi(S, cov_a=0.2)
        st.zeta_from(0.9, S, cov_a=0.2)
        st.sojourn_time(S, cov_a=0.2)
    assert st.cov_a == 1.7
    assert [st.phi(S) for S in S_GRID] == before


def test_the_level_arm_ignores_the_override():
    st = _gg1(1.7, mode=ZETA_LEVEL)
    for S in S_GRID:
        x = S * st.mu - st.gamma
        assert st.zeta_from(0.9, S, cov_a=0.2) == 0.9 * x


def test_a_measured_zero_cov_takes_the_deterministic_branch():
    # cov_s = 0 and a measured cov_a = 0 give k == 0 from the OVERRIDE, although the
    # constructor k is 0.5. phi must then be exactly the k == 0 value, 1 - rho.
    st = _gg1(1.0, cov_s=0.0)
    for S in S_GRID:
        rho = st.gamma / (S * st.mu)
        assert st.phi(S, cov_a=0.0) == pytest.approx(1.0 - rho, rel=1e-12)
        assert st.sojourn_time(S, cov_a=0.0) == 1.0 / (S * st.mu)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, -math.inf])
def test_an_invalid_override_is_rejected_like_the_constructor_argument(bad):
    st = _gg1(1.7)
    with pytest.raises(ValueError, match="cov_a must be a finite number >= 0"):
        st.phi(1.0, cov_a=bad)
    with pytest.raises(ValueError, match="cov_a must be a finite number >= 0"):
        st.sojourn_time(1.0, cov_a=bad)


def test_a_fork_join_station_ignores_the_override():
    fj = ForkJoinStation(0.5, 1.0, r=2.0, c1=1.0, c2=1.0, name="fj", zeta_mode=ZETA_SLOPE)
    for S in [0.8, 1.5, 3.0]:
        assert fj.phi(S, cov_a=0.3) == fj.phi(S)
        assert fj.zeta_from(0.9, S, cov_a=0.3) == fj.zeta_from(0.9, S)


def test_only_a_slope_station_that_reads_cov_a_uses_a_measurement():
    assert GG1Station.reads_arrival_cov is True
    assert ForkJoinStation.reads_arrival_cov is False
    assert _gg1(1.0).uses_measured_cov_a is True
    assert _gg1(1.0, mode=ZETA_LEVEL).uses_measured_cov_a is False
    fj = ForkJoinStation(0.5, 1.0, r=2.0, c1=1.0, c2=1.0, zeta_mode=ZETA_SLOPE)
    assert fj.uses_measured_cov_a is False


def test_zeta_from_does_not_pass_the_keyword_when_there_is_no_measurement():
    # A user subclass written before the keyword existed must keep working on every
    # path that never measures.
    class Legacy(GG1Station):
        def phi(self, S):
            return super().phi(S)

    st = Legacy(0.6, 1.5, c=2.0, cov_a=1.7, cov_s=0.5, zeta_mode=ZETA_SLOPE)
    assert st.zeta_from(0.9, 1.0) == _gg1(1.7).zeta_from(0.9, 1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_arrival_cov.py -v`
Expected: FAIL — `TypeError: ... got an unexpected keyword argument 'cov_a'` and `AttributeError: ... 'reads_arrival_cov'`.

- [ ] **Step 3: Implement**

In `qopt/station.py`, after the `DOT_SHAPE` attribute and its docstring in `Station`, add:

```python
    reads_arrival_cov = False
    """Does this station's analytic model read the arrival process's coefficient of variation?

    If it does, a MEASURED one can stand in for the constructor's: `phi` and `zeta_from`
    take it as `cov_a=` for one evaluation. False here because the base class has no model
    to put one in, and on ForkJoinStation because `t_ul` has none. A subclass that sets it
    must accept `cov_a` in `phi`, and in `sojourn_time` to take part in the shape check.
    """
```

After the `zeta_mode` property, add:

```python
    @property
    def uses_measured_cov_a(self):
        """Should a simulated evaluation measure this station's arrival `cov_a`?

        Only a slope-mode station reads `phi`, and only a station that `reads_arrival_cov`
        has a `phi` a measurement changes. The one predicate both SimulationAnalyzer's
        request trigger and `extract_arrival_cov` read, so they cannot disagree.
        """
        return self.reads_arrival_cov and self._zeta_mode == ZETA_SLOPE
```

Change `Station.phi`'s signature to `def phi(self, S, *, cov_a=None):` and append to its docstring:

```
        `cov_a` is a measured arrival coefficient of variation for ONE evaluation (see
        `reads_arrival_cov`). The base model has none to replace, so it is ignored here.
```

Change `Station.zeta_from`'s signature to `def zeta_from(self, T, S, *, cov_a=None):`, add to its docstring:

```
        `cov_a`, when not None, is the arrival cov measured by the evaluation that produced
        T; only the slope arm reads it, through `phi`.
```

and replace the line `phi = self.phi(S)` with:

```python
            # Passed only when there is one, so an override of `phi(self, S)` written before
            # this keyword existed keeps working on every path that measures nothing.
            phi = self.phi(S) if cov_a is None else self.phi(S, cov_a=cov_a)
```

In the `phi == 0.0` boundary branch's comment block, append one paragraph:

```python
                # `admits_full_utilization` reads the CONSTRUCTOR k even when `cov_a` is a
                # measurement. The simulated path, the only one that measures, never gets
                # here: its analyzer refuses `S*mu == gamma` before evaluating.
```

In `GG1Station`, add the class attribute and `_k` directly after the class docstring / before `__init__`:

```python
    reads_arrival_cov = True
```

and after `admits_full_utilization`:

```python
    def _k(self, cov_a):
        """k = (cov_a^2 + cov_s^2)/2, with a measured `cov_a` in place of the constructor's.

        None evaluates today's expression on the constructor value, so every default call
        is bit-for-bit what it was. A measurement is validated like the constructor
        argument: a NaN would otherwise pass every comparison and price silently.
        """
        if cov_a is None:
            cov_a = self.cov_a
        elif not math.isfinite(cov_a) or cov_a < 0:
            raise ValueError(f"cov_a must be a finite number >= 0, got {cov_a}")
        return (cov_a ** 2 + self.cov_s ** 2) / 2.0
```

In `GG1Station.sojourn_time`: signature `def sojourn_time(self, S, *, cov_a=None):`, replace `k = (self.cov_a ** 2 + self.cov_s ** 2) / 2.0` with `k = self._k(cov_a)`.
In `GG1Station.dT_dS`: signature `def dT_dS(self, S, *, cov_a=None):`, same replacement. Keep `_check_stable` before `_k` in both (instability outranks a bad override, as before).

Add after `dT_dS`:

```python
    def phi(self, S, *, cov_a=None):
        """Station.phi, with a measured `cov_a` when one is given.

        The same expression as the base, evaluated on the widened pair, so a station built
        with cov_a=c and one measured at c agree exactly (test_arrival_cov.py pins it).
        """
        if cov_a is None:
            return super().phi(S)
        x = S * self.mu - self.gamma
        return (-self.dT_dS(S, cov_a=cov_a) * x
                / (self.mu * self.sojourn_time(S, cov_a=cov_a)))
```

Check the base `phi` body is literally `x = S * self.mu - self.gamma` / `return -self.dT_dS(S) * x / (self.mu * self.sojourn_time(S))`; the override must mirror its operation order exactly or the exact-equality test fails.

- [ ] **Step 4: Run to verify they pass, and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_arrival_cov.py -v && .venv/bin/python -m pytest -q`
Expected: new tests PASS; suite `554 passed` plus the new ones. Any failure in an existing test means the default arithmetic moved — fix the code, don't edit the test.

- [ ] **Step 5: Mutation check**

Revert, one at a time, (a) `phi(S, cov_a=cov_a)` → `phi(S)` in `zeta_from`, (b) `self._k(cov_a)` → `self._k(None)` in `sojourn_time`, (c) the `_k` validation branch. Clear `__pycache__` each time; each must fail at least one new test. Restore, clear, re-run green.

- [ ] **Step 6: Commit**

```bash
git add qopt/station.py tests/test_arrival_cov.py
git commit -m "feat(station): accept a measured arrival cov_a for one evaluation

GG1Station.sojourn_time, dT_dS and phi, and Station.zeta_from, take a
cov_a keyword that stands in for the constructor value without writing
it. uses_measured_cov_a is the single predicate for which stations a
simulated run should measure.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

### Task 2: Extract a measured `cov_a` from a qsim response

**Files:**
- Modify: `qopt/analyzer.py` (the `Evaluation` dataclass)
- Modify: `qopt/qsim/spec.py` (after `MEASURES`)
- Modify: `qopt/qsim/measures.py` (new function after `extract`)
- Modify: `tests/test_qsim_measures.py` (append)

**Interfaces:**
- Consumes: `Station.uses_measured_cov_a` (Task 1).
- Produces:
  - `Evaluation.arrival_cov: list | None = None`.
  - `qopt.qsim.spec.COV_A_MEASURE = "interarrival-time"`.
  - `qopt.qsim.measures.extract_arrival_cov(response, stations, job_class) -> (list, list)` — `(arrival_cov, degraded)`, `arrival_cov[i]` a float or `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_qsim_measures.py` (add `import math`, `import warnings` at the top if absent):

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_qsim_measures.py -v -k "cov or arrival or scv or unusable or weak_measurement or tiny"`
Expected: FAIL — `ImportError: cannot import name 'extract_arrival_cov'`.

- [ ] **Step 3: Implement**

`qopt/analyzer.py` — add to the `Evaluation` docstring's field list:

```
        arrival_cov: measured arrival cov_a per station, aligned to the station order,
            each a float or None (not measured, or unusable -- priced at the constructor
            cov_a); the whole field None when nothing was measured.
```

and the field after `extras`:

```python
    arrival_cov: list | None = None
```

`qopt/qsim/spec.py` — after the `MEASURES` docstring:

```python
COV_A_MEASURE = "interarrival-time"
"""Per-station interarrival moments, from which a measured `cov_a` is computed.

Requested only when some station has `uses_measured_cov_a` (SimulationAnalyzer). Facts
from qsim-service's README at 8e7358b: it reports `mean`, `variance` and `stdDev` of the
interarrival time, so SCV = variance/mean**2; it reports NO confidence interval, because
JMT's is on the rate; it switches on JMT's per-sample logging for itself alone, roughly
25-30% wall clock plus temporary disk -- which is why `secondMoments` is never sent, since
that would put every other measure on the same cost; and on a fork-join node it is taken
at the fork. A service older than 8e7358b rejects it as an unsupported measure type.
"""
```

`qopt/qsim/measures.py` — add `import math` at the top, `from qopt.qsim.spec import COV_A_MEASURE` after the existing imports, and after `extract`:

```python
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
```

Check `qopt/qsim/spec.py` does not import from `measures.py` (it must not, or this is circular).

- [ ] **Step 4: Run to verify they pass, and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_qsim_measures.py -v && .venv/bin/python -m pytest -q`
Expected: PASS; no existing test changed.

- [ ] **Step 5: Mutation check**

Revert, one at a time: (a) `variance / mean / mean` → `variance / mean ** 2` (the `mean-tiny` row must fail with `ZeroDivisionError`); (b) drop the `mean <= 0` test; (c) `if not st.uses_measured_cov_a` → `if not st.reads_arrival_cov` (the level `mm1` must then fail `test_cov_a_is_the_square_root...`). Clear `__pycache__` each time; restore, clear, green.

- [ ] **Step 6: Commit**

```bash
git add qopt/analyzer.py qopt/qsim/spec.py qopt/qsim/measures.py tests/test_qsim_measures.py
git commit -m "feat(qsim): extract a measured arrival cov_a from interarrival-time

Evaluation gains arrival_cov. extract_arrival_cov reads qsim-service's
interarrival moments (8e7358b+) for each station that uses a measurement,
and degrades to None -- the constructor cov_a -- for a missing or
degenerate one.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

### Task 3: `SimulationAnalyzer` requests and returns the measurement

**Files:**
- Modify: `qopt/qsim/analyzer.py`
- Modify: `tests/test_qsim_analyzer.py` (append)

**Interfaces:**
- Consumes: `COV_A_MEASURE`, `MEASURES` (`qopt.qsim.spec`); `extract_arrival_cov` (Task 2); `Station.uses_measured_cov_a` (Task 1).
- Produces: `SimulationAnalyzer(network, client, *, seed=..., seed_policy=..., strict=False, measure_cov_a=True)`; `evaluate` returns `Evaluation(..., arrival_cov=list | None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_qsim_analyzer.py`:

```python
# --- measured arrival cov_a (spec 2026-09-25 §3.3) ------------------------------

from qopt.exceptions import SimulationRequestError
from qopt.qsim.spec import COV_A_MEASURE, MEASURES
from qopt.zeta import ZETA_LEVEL


def _slope_network(md1_mode=ZETA_SLOPE, fj_mode=ZETA_LEVEL):
    stations = [
        GG1Station.mm1(mu=1.0, c=2.0, name="mm1"),
        GG1Station.md1(mu=1.0, c=1.0, name="md1", zeta_mode=md1_mode),
        ForkJoinStation(mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fj", zeta_mode=fj_mode),
    ]
    routes = [
        Route(Network.SOURCE, "mm1", 0.6), Route(Network.SOURCE, "md1", 0.4),
        Route("mm1", "fj", 0.5), Route("mm1", Network.SINK, 0.5),
        Route("md1", "fj", 0.5), Route("md1", Network.SINK, 0.5),
        Route("fj", Network.SINK, 1.0),
    ]
    return Network(stations, routes, arrival_rate=1.0, name="qopt-mixed-network")


def _ia_entry(station, mean, variance):
    return {"station": station, "class": "jobs", "type": COV_A_MEASURE,
            "mean": mean, "lower": None, "upper": None, "success": True,
            "variance": variance, "stdDev": variance ** 0.5}


def _with_interarrival(response, **by_station):
    response = dict(response)
    response["measures"] = response["measures"] + [
        _ia_entry(name, mean, var) for name, (mean, var) in by_station.items()
    ]
    return response


def test_a_network_with_no_slope_gg1_sends_todays_request(sim_response):
    network = _network()                                  # every station level
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    ev = analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES)
    assert "secondMoments" not in transport.requests[0]
    assert ev.arrival_cov is None


def test_a_slope_fork_join_alone_does_not_trigger_the_measure(sim_response):
    network = _slope_network(md1_mode=ZETA_LEVEL, fj_mode=ZETA_SLOPE)
    analyzer, transport = _analyzer(network, _healthy(sim_response))
    analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES)


def test_a_slope_gg1_adds_interarrival_time_and_nothing_else(sim_response):
    network = _slope_network()
    response = _with_interarrival(_healthy(sim_response), md1=(2.5, 1.5625))
    analyzer, transport = _analyzer(network, response)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES) + [COV_A_MEASURE]
    assert "secondMoments" not in transport.requests[0]
    assert ev.arrival_cov == [None, 0.5, None]            # sqrt(1.5625 / 2.5**2)
    assert ev.degraded == []


def test_measure_cov_a_false_restores_todays_request(sim_response):
    network = _slope_network()
    analyzer, transport = _analyzer(network, _healthy(sim_response), measure_cov_a=False)
    ev = analyzer.evaluate(network.stations, S_OK)
    assert transport.requests[0]["measures"] == list(MEASURES)
    assert ev.arrival_cov is None


def test_a_missing_measurement_degrades_and_strict_raises(sim_response):
    network = _slope_network()
    analyzer, _ = _analyzer(network, _healthy(sim_response))      # no interarrival entry
    with pytest.warns(RuntimeWarning, match="md1"):
        ev = analyzer.evaluate(network.stations, S_OK)
    assert ev.arrival_cov == [None, None, None]
    assert any("md1" in d and "constructor cov_a" in d for d in ev.degraded)

    strict, _ = _analyzer(network, _healthy(sim_response), strict=True)
    with pytest.warns(RuntimeWarning), pytest.raises(SimulationQualityError, match="md1"):
        strict.evaluate(network.stations, S_OK)


_OLD_SERVICE = (400, {"error": "invalid request", "details": [
    "unsupported measure type: 'interarrival-time'; supported: [response-time, throughput]"
]})


def test_an_old_service_names_the_opt_out():
    network = _slope_network()
    client = QsimClient("http://qsim.test", transport=FakeTransport(_OLD_SERVICE))
    with pytest.raises(SimulationRequestError, match=r"measure_cov_a=False"):
        SimulationAnalyzer(network, client).evaluate(network.stations, S_OK)


def test_an_unrelated_request_error_gets_no_version_hint():
    network = _slope_network()
    other = (400, {"error": "invalid request", "details": ["node name 'x y' is unsafe"]})
    client = QsimClient("http://qsim.test", transport=FakeTransport(other))
    with pytest.raises(SimulationRequestError) as info:
        SimulationAnalyzer(network, client).evaluate(network.stations, S_OK)
    assert "measure_cov_a" not in str(info.value)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_qsim_analyzer.py -v -k "slope or cov or old_service or unrelated or todays"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'measure_cov_a'`, and the slope request lacks `interarrival-time`.

- [ ] **Step 3: Implement**

In `qopt/qsim/analyzer.py`:

Imports become:

```python
from qopt.analyzer import Analyzer, Evaluation
from qopt.exceptions import SimulationQualityError, SimulationRequestError
from qopt.qsim.measures import extract, extract_arrival_cov
from qopt.qsim.spec import COV_A_MEASURE, MEASURES, build_request
```

Add after `_SEED_POLICIES`:

```python
_OLD_SERVICE_REJECTION = f"unsupported measure type: '{COV_A_MEASURE}'"
"""What a qsim-service older than 8e7358b says to a request carrying the measure."""
```

`__init__` gains `measure_cov_a=True` (last keyword) and `self.measure_cov_a = measure_cov_a`. Add to the class docstring:

```
    `measure_cov_a=True` asks qsim for each slope G/G/1 station's interarrival moments
    whenever the network has one (`Station.uses_measured_cov_a`), so `phi` prices the
    arrival variability the station actually sees rather than its constructor `cov_a`. It
    costs roughly 25-30% wall clock and needs qsim-service 8e7358b+; False sends exactly
    the pre-measurement request and prices every station at its constructor `cov_a`.
```

Add a method:

```python
    def _measures_cov_a(self, stations):
        """Ask for interarrival moments only when some station will use them.

        For every other network the request -- and its cost -- is exactly what it was
        before the measure existed.
        """
        return self.measure_cov_a and any(st.uses_measured_cov_a for st in stations)
```

In `evaluate`, replace the `build_request(...)` / `post_simulate` lines with:

```python
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
```

After `degraded.extend(_conservation_misses(...))`:

```python
        arrival_cov = None
        if measure_cov_a:
            arrival_cov, cov_degraded = extract_arrival_cov(
                response, stations, self.network.job_class
            )
            degraded.extend(cov_degraded)
```

and pass `arrival_cov=arrival_cov` to the returned `Evaluation`. The strict check already sits after this and sees the new entries.

- [ ] **Step 4: Run to verify they pass, and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_qsim_analyzer.py -v && .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Revert, one at a time: (a) the trigger to `self.measure_cov_a` alone (`test_a_network_with_no_slope_gg1_sends_todays_request` must fail); (b) the hint's `_OLD_SERVICE_REJECTION in str(exc)` to `True` (`test_an_unrelated_request_error_gets_no_version_hint` must fail); (c) drop `degraded.extend(cov_degraded)` (strict test must fail). Clear `__pycache__` each time; restore, clear, green.

- [ ] **Step 6: Commit**

```bash
git add qopt/qsim/analyzer.py tests/test_qsim_analyzer.py
git commit -m "feat(qsim): request interarrival-time when a slope G/G/1 will use it

SimulationAnalyzer(measure_cov_a=True) adds the measure only when some
station has uses_measured_cov_a, so every other request is byte-identical.
An old service's rejection names the opt-out instead of retrying.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

### Task 4: The optimizer prices each evaluation at its own measurement

**Files:**
- Modify: `qopt/optimizer.py` (module helpers after imports; `Result` after `zeta_shape_flags`; loop at ~318-374; final block at ~546-586; `_noise_floor` at ~611)
- Modify: `tests/test_arrival_cov.py` (append)

**Interfaces:**
- Consumes: `Evaluation.arrival_cov` (Task 2); `Station.zeta_from(..., cov_a=)`, `Station.phi(..., cov_a=)` (Task 1).
- Produces:
  - `Result.arrival_cov: list` (default `[]`).
  - `Optimizer._noise_floor(stations, S, zeta, ci, arrival_cov=None)`.
  - module-private `_measured(evaluation, n) -> list` and `_zeta_from(st, T, S, cov_a)`, `_phi(st, S, cov_a)` in `qopt/optimizer.py` — Task 5 uses `arrival_cov` from the loop.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arrival_cov.py`:

```python
# --- optimizer plumbing ---------------------------------------------------------

import warnings

from qopt.allocator import min_feasible_budget, noise_floor
from qopt.analyzer import Analyzer, Evaluation
from qopt.optimizer import Optimizer

# zeta_shape_tol=None keeps these plumbing tests independent of Task 5's check, which
# (before Task 5) would compare against the constructor model and warn.
NAIVE = dict(warm_start=False, damping=1.0, noise_kappa=0.0, max_iter=1000,
             zeta_shape_tol=None)


def _pair(md1_cov_a=1.0):
    # md1 is cov_s = 0 with constructor cov_a = 1 (k = 0.5); a measured 0.3 gives k = 0.045,
    # far enough that phi -- and so the fixed point -- moves.
    return [
        GG1Station(0.6, 1.5, c=2.0, cov_a=md1_cov_a, cov_s=0.0, name="md1",
                   zeta_mode=ZETA_SLOPE),
        GG1Station.mm1(1.2, 3.0, c=0.5, name="mm1", zeta_mode=ZETA_SLOPE),
    ]


C = 2.0 * min_feasible_budget(_pair())


class CovFake(Analyzer):
    """A stochastic-declared analyzer whose E[T] is the station's model AT the measured
    cov_a, and which reports that cov_a: a simulator whose measurement is exactly right.

    `loop` is returned by every non-final evaluation (or `script[i]` for the i-th, the last
    entry repeating), `final` by the fresh-seeded one, so a test can tell which evaluation
    a Result field came from. `factor` scales E[T], a stand-in for a model-shape error.
    """

    is_stochastic = True

    def __init__(self, loop=None, final="same", *, script=None, half_width=None,
                 factor=1.0):
        self.loop, self.script = loop, script
        self.final = loop if final == "same" else final
        self.half_width, self.factor = half_width, factor
        self.calls, self.seen = 0, []

    def evaluate(self, stations, S, *, fresh_seed=False):
        if fresh_seed:
            cov = self.final
        elif self.script is not None:
            cov = self.script[min(self.calls, len(self.script) - 1)]
        else:
            cov = self.loop
        if not fresh_seed:
            self.calls += 1
            self.seen.append(list(S))
        covs = [None] * len(stations) if cov is None else cov
        T = [(st.sojourn_time(Si) if c is None else st.sojourn_time(Si, cov_a=c))
             * self.factor for st, Si, c in zip(stations, S, covs)]
        ci = None if self.half_width is None else [
            (t - self.half_width, t + self.half_width) for t in T]
        return Evaluation(sojourn_times=T, ci=ci,
                          arrival_cov=None if cov is None else list(cov))


def test_a_measured_run_is_the_analytic_run_of_the_measured_model():
    # Bitwise: the override path runs the same arithmetic as a station built with the
    # measured value (Task 1), so the whole loop must be identical. Fails if the LOOP zeta
    # (point 1) or the FINAL zeta/phi (point 4) ignores the measurement.
    measured = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None]), **NAIVE).run()
    reference = Optimizer(_pair(md1_cov_a=0.3), C,
                          analyzer=CovFake(loop=None), **NAIVE).run()
    assert measured.capacities == reference.capacities
    assert measured.iterations == reference.iterations
    assert measured.zeta == reference.zeta
    assert measured.zeta_phi == reference.zeta_phi
    # Anti-vacuity: the measurement really moved the answer.
    unmeasured = Optimizer(_pair(), C, analyzer=CovFake(loop=None), **NAIVE).run()
    assert measured.capacities != unmeasured.capacities


def test_result_fields_come_from_the_final_evaluation():
    res = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None], final=[0.8, None]),
                    **NAIVE).run()
    st, S0, T0 = _pair()[0], res.capacities[0], res.sojourn_times[0]
    assert res.arrival_cov == [0.8, None]
    assert res.zeta[0] == st.zeta_from(T0, S0, cov_a=0.8)
    assert res.zeta_phi[0] == st.phi(S0, cov_a=0.8)
    assert res.zeta_phi[0] != st.phi(S0, cov_a=0.3)        # a mix-up would be visible


def test_without_a_final_evaluation_the_last_loop_measurement_is_reported():
    res = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None], final=[0.8, None]),
                    final_evaluation=False, **NAIVE).run()
    assert res.arrival_cov == [0.3, None]


def test_a_missing_final_measurement_is_not_filled_from_the_loop():
    # Per evaluation, never cached: a station unmeasured in the final evaluation is priced
    # at its constructor cov_a there, even though the loop measured it.
    res = Optimizer(_pair(), C, analyzer=CovFake(loop=[0.3, None], final=[None, None]),
                    **NAIVE).run()
    st, S0, T0 = _pair()[0], res.capacities[0], res.sojourn_times[0]
    assert res.arrival_cov == [None, None]
    assert res.zeta[0] == st.zeta_from(T0, S0)


def test_a_measurement_that_vanishes_mid_run_is_priced_at_the_constructor_then():
    # Iteration 2 loses md1's measurement. Its capacities must be those of a run that
    # measured nothing at iteration 2, which a cache of the last good value would break.
    script = [[0.3, None], [None, None], [0.3, None]]
    fake = CovFake(script=script)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        Optimizer(_pair(), C, analyzer=fake, **dict(NAIVE, max_iter=3)).run()
    S2, S3 = fake.seen[1], fake.seen[2]
    # S3 = eq 21 at the zeta of iteration 2, which must be the UNMEASURED zeta at S2.
    from qopt.allocator import allocate
    st = _pair()
    zeta2 = [s.zeta_from(s.sojourn_time(Si), Si) for s, Si in zip(st, S2)]
    assert S3 == allocate(st, C, zeta2)


def test_the_noise_floor_goes_through_the_same_override():
    fake = CovFake(loop=[0.3, None], half_width=0.02)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = Optimizer(_pair(), C, analyzer=fake, warm_start=False, damping=0.5,
                        max_iter=1, zeta_shape_tol=None).run()
    st, S = _pair(), fake.seen[0]
    ev = CovFake(loop=[0.3, None], half_width=0.02).evaluate(st, S)
    zeta = [st[0].zeta_from(ev.sojourn_times[0], S[0], cov_a=0.3),
            st[1].zeta_from(ev.sojourn_times[1], S[1])]
    dzeta = [st[0].zeta_from(0.02, S[0], cov_a=0.3), st[1].zeta_from(0.02, S[1])]
    assert res.noise_floor == noise_floor(st, C, zeta, dzeta)
    unmeasured = [st[0].zeta_from(0.02, S[0]), dzeta[1]]
    assert res.noise_floor != noise_floor(st, C, zeta, unmeasured)   # anti-vacuity


def test_nothing_measured_leaves_arrival_cov_empty():
    assert Optimizer(_pair(), C).run().arrival_cov == []
    assert Optimizer(_pair(), C, analyzer=CovFake(loop=None), **NAIVE).run().arrival_cov == []


def test_an_old_signature_phi_override_survives_a_measured_run():
    class Legacy(GG1Station):
        reads_arrival_cov = False               # a user type qopt knows nothing about
        def phi(self, S):
            return super().phi(S)

    stations = [
        Legacy(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=0.0, name="legacy", zeta_mode=ZETA_SLOPE),
        GG1Station(1.2, 3.0, c=0.5, cov_a=1.0, cov_s=0.0, name="g", zeta_mode=ZETA_SLOPE),
    ]
    budget = 2.0 * min_feasible_budget(stations)
    res = Optimizer(stations, budget, analyzer=CovFake(loop=[None, 0.3]), **NAIVE).run()
    assert res.arrival_cov == [None, 0.3]
```

Note on `test_a_measurement_that_vanishes_mid_run_is_priced_at_the_constructor_then`: `damping=1.0` makes `S3` exactly eq 21's target. If `allocate`'s import path differs, it is `qopt.allocator.allocate(stations, budget, zeta)` (see `qopt/optimizer.py` imports).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_arrival_cov.py -v -k "run or final or noise or empty or survives or vanishes"`
Expected: FAIL — `Result` has no `arrival_cov`; capacities equal the unmeasured run's.

- [ ] **Step 3: Implement**

In `qopt/optimizer.py`, after the imports:

```python
def _measured(evaluation, n):
    """Per-station measured cov_a from `evaluation`, or n Nones when it measured nothing.

    Read from THE evaluation whose E[T] is being inverted, every time -- never carried
    over from an earlier one, because a downstream station's true cov_a moves with S.
    """
    return [None] * n if evaluation.arrival_cov is None else list(evaluation.arrival_cov)


def _zeta_from(st, T, S, cov_a):
    """`st.zeta_from`, passing `cov_a` only when one was measured, so a station type whose
    `zeta_from` predates the keyword keeps working wherever nothing is measured."""
    return st.zeta_from(T, S) if cov_a is None else st.zeta_from(T, S, cov_a=cov_a)


def _phi(st, S, cov_a):
    """`st.phi`, under the same rule as `_zeta_from`."""
    return st.phi(S) if cov_a is None else st.phi(S, cov_a=cov_a)
```

`Result` — after `zeta_shape_flags` and its docstring:

```python
    arrival_cov: list = field(default_factory=list)
    """Per-station arrival cov_a MEASURED by the evaluation `zeta` was recomputed from, or
    None where that station was priced at its constructor cov_a -- it uses no measurement
    (`Station.uses_measured_cov_a`), or its measurement was unusable. Empty when nothing was
    measured: the analytic path, `measure_cov_a=False`, or no station that uses one.

    Like `zeta`, it comes from the FINAL evaluation, which on a stochastic path is a
    different sample from the loop iterate that set the capacities: it describes the
    reported zeta, not the trajectory.
    """
```

Loop — immediately after `evaluation = self.analyzer.evaluate(stations, S)` and the snapshot lines, add:

```python
            arrival_cov = _measured(evaluation, len(stations))
```

Replace the eq-22 list and the `_noise_floor` call:

```python
            zeta = [
                _zeta_from(st, T, Si, c)
                for st, T, Si, c in zip(stations, evaluation.sojourn_times, S, arrival_cov)
            ]                                                    # eq 22
            S_target = allocate(stations, self.budget, zeta)      # eq 21

            floor = self._noise_floor(stations, S, zeta, evaluation.ci, arrival_cov)
```

Final block — replace the `zeta` and `zeta_phi` comprehensions:

```python
        arrival_cov = _measured(evaluation, len(stations))
        zeta = [
            _zeta_from(st, T, Si, c)
            for st, T, Si, c in zip(stations, sojourn_times, S, arrival_cov)
        ]
```

(keep the existing comment block that follows it, then)

```python
        zeta_phi = [
            _phi(st, Si, c) if st.zeta_mode == ZETA_SLOPE else 1.0
            for st, Si, c in zip(stations, S, arrival_cov)
        ]
```

Append one sentence to that comment block: `The measured cov_a is this same final evaluation's, for the same reason.` Pass `arrival_cov=[] if evaluation.arrival_cov is None else list(evaluation.arrival_cov),` to `Result(...)` after `zeta_shape_flags=`.

`_noise_floor`:

```python
    def _noise_floor(self, stations, S, zeta, ci, arrival_cov=None):
```

add to its docstring: `arrival_cov is the measurement the zeta was computed under; the half-width goes through the same override, or the propagation is not the linear map it assumes.` and replace the `dzeta` comprehension:

```python
        if arrival_cov is None:
            arrival_cov = [None] * len(stations)
        dzeta = [
            0.0 if entry is None else _zeta_from(st, 0.5 * (entry[1] - entry[0]), Si, c)
            for st, Si, entry, c in zip(stations, S, ci, arrival_cov)
        ]
```

- [ ] **Step 4: Run to verify they pass, and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_arrival_cov.py -v && .venv/bin/python -m pytest -q`
Expected: PASS; no existing test changed (in particular `test_analytic_defaults_reproduce_the_legacy_numbers_bitwise`).

- [ ] **Step 5: Mutation check — one point at a time**

Revert each of these alone, clearing `__pycache__` before each run; each must fail at least one new test:
1. loop `_zeta_from(st, T, Si, c)` → `st.zeta_from(T, Si)`;
2. `_noise_floor(..., arrival_cov)` → `_noise_floor(...)` (drop the argument);
3. final `_zeta_from(...)` → `st.zeta_from(T, Si)`;
4. final `_phi(st, Si, c)` → `st.phi(Si)`;
5. `_measured` returning a cached last non-`None` value (e.g. store on `self`) — `..._vanishes_mid_run_...` must fail.

Restore, clear, green. Record which test killed each mutation in the commit body.

- [ ] **Step 6: Commit**

```bash
git add qopt/optimizer.py tests/test_arrival_cov.py
git commit -m "feat(optimizer): price each evaluation at its own measured cov_a

Eq 22, the noise floor and the final zeta/phi read Evaluation.arrival_cov
from the evaluation whose E[T] they invert; Result.arrival_cov reports the
final one. Mutation-checked at all four points and against a cached value.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

Before committing, add one line per Step-5 mutation to the body (`Mutation N killed by test_...`), naming the test that actually failed.

---

### Task 5: The shape check compares against the measured model

**Files:**
- Modify: `qopt/optimizer.py` (the `zeta_shape_tol` block, ~320-366)
- Modify: `tests/test_arrival_cov.py` (append)

**Interfaces:**
- Consumes: the loop-local `arrival_cov` (Task 4); `GG1Station.sojourn_time(S, *, cov_a=)` (Task 1).
- Produces: no new API; a second message form in `Result.zeta_shape_flags`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arrival_cov.py`:

```python
from qopt.zeta import ZETA_SHAPE_TOL


def _mismatched():
    # Constructor cov_a = 2 at cov_s = 0 (k = 2); measured 0 (k = 0). E[T] under the two
    # differs by a factor 1 + 2*rho/(1-rho): far past the 25% tolerance at any load.
    return [
        GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=0.0, name="gg", zeta_mode=ZETA_SLOPE),
        GG1Station.mm1(1.2, 3.0, c=0.5, name="mm1"),
    ]


def test_a_correct_measurement_silences_a_check_the_constructor_would_fail():
    stations = _mismatched()
    budget = 2.0 * min_feasible_budget(stations)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Optimizer(stations, budget, analyzer=CovFake(loop=[0.0, None]),
                        **dict(NAIVE, zeta_shape_tol=ZETA_SHAPE_TOL)).run()
    assert res.zeta_shape_flags == []
    # Anti-vacuity: against the CONSTRUCTOR model this E[T] is far outside tolerance.
    st, Si, T = stations[0], res.capacities[0], res.sojourn_times[0]
    assert abs(T / st.sojourn_time(Si) - 1.0) > ZETA_SHAPE_TOL


def test_a_shape_error_at_the_measured_cov_names_the_measurement():
    stations = _mismatched()
    budget = 2.0 * min_feasible_budget(stations)
    with pytest.warns(RuntimeWarning, match="gg"):
        res = Optimizer(stations, budget,
                        analyzer=CovFake(loop=[0.0, None], factor=4.0),
                        **dict(NAIVE, zeta_shape_tol=ZETA_SHAPE_TOL)).run()
    assert len(res.zeta_shape_flags) == 1
    message = res.zeta_shape_flags[0]
    assert "measured arrival cov_a=0" in message
    assert "check the station's parameters" not in message   # not today's advice
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_arrival_cov.py -v -k "shape or silences"`
Expected: FAIL — the first raises the constructor-model `RuntimeWarning`; the second's message is today's.

- [ ] **Step 3: Implement**

In the shape-check loop, iterate `for st, T, Si, c in zip(stations, evaluation.sojourn_times, S, arrival_cov):`, replace `T_model = st.sojourn_time(Si)` with

```python
                    T_model = (st.sojourn_time(Si) if c is None
                               else st.sojourn_time(Si, cov_a=c))
```

and, inside the `if abs(...)` after `shape_checked.add(id(st))`, choose the message:

```python
                        disagreement = abs(T / T_model - 1.0) * 100
                        if c is None:
                            message = (<today's message, unchanged>)
                        else:
                            message = (
                                f"station {st.name!r}: measured E[T]={T:g} disagrees with "
                                f"its analytic model's {T_model:g} at the measured arrival "
                                f"cov_a={c:g} by {disagreement:.1f}%, above "
                                f"zeta_shape_tol={self.zeta_shape_tol:g}. The arrival "
                                f"variability is measured here, not assumed, so the "
                                f"disagreement is in the model's shape itself -- the G/G/1 "
                                f"approximation at this load -- and slope-calibrated zeta "
                                f"takes its phi from that shape."
                            )
```

Keep today's message text byte-identical (existing tests match on it). Update the block's leading comment: after "Comparing the two E[T] values tests exactly that assumption", add: "When the analyzer measured this station's cov_a, the model is evaluated AT the measurement, so the check then tests the approximation's shape rather than the constructor argument."

- [ ] **Step 4: Run to verify they pass, and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_arrival_cov.py tests/test_zeta.py -v && .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Revert `T_model` to `st.sojourn_time(Si)` alone; clear `__pycache__`; `test_a_correct_measurement_silences...` must fail. Restore, clear, green.

- [ ] **Step 6: Commit**

```bash
git add qopt/optimizer.py tests/test_arrival_cov.py
git commit -m "feat(optimizer): cross-check measured E[T] against the model at measured cov_a

With a measurement, zeta_shape_tol tests the G/G/1 approximation's shape
instead of the constructor cov_a, and its message says so.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

### Task 6: Live witnesses against qsim-service

**Files:**
- Modify: `tests/test_integration_qsim.py` (append)

**Interfaces:**
- Consumes: `SimulationAnalyzer(..., measure_cov_a=True)`, `Evaluation.arrival_cov` (Tasks 2-3).
- Produces: nothing.

- [ ] **Step 1: Confirm a live service at `8e7358b`+**

Run: `echo $QOPT_QSIM_URL; curl -s "$QOPT_QSIM_URL/health"`
If unset or unreachable, ask the user to start qsim-service built from `8e7358b`+ (their `modeling-analysis/qsim-service` checkout at `~/Projects/modeling-analysis/qsim-service`) and export `QOPT_QSIM_URL`. Do not rebuild or start it without asking. Without a service these tests skip — say so in the report rather than claiming them green.

- [ ] **Step 2: Write the tests**

Append to `tests/test_integration_qsim.py` (add `import math` and `from qopt.zeta import ZETA_SLOPE` to the imports):

```python
def _tandem(up, down, arrival_rate, name):
    return Network(
        [up, down],
        [Route(Network.SOURCE, up.name), Route(up.name, down.name),
         Route(down.name, Network.SINK)],
        arrival_rate=arrival_rate, name=name,
    )


def test_burke_an_mm1_feeds_its_successor_poisson_arrivals(client):
    # Burke: an M/M/1's departures are Poisson, so the successor's interarrival SCV is 1.
    # For iid exponential samples the SCV estimator's standard error is 2/sqrt(n) (delta
    # method: Var = 4/n). n >= minSamples/2 after warm-up discards is a conservative
    # floor, and the tolerance is 5 standard errors at that floor.
    up = GG1Station.mm1(mu=1.0, c=1.0, name="up")
    down = GG1Station.mm1(mu=1.0, c=1.0, name="down", zeta_mode=ZETA_SLOPE)
    network = _tandem(up, down, 0.5, "burke-tandem")
    ev = SimulationAnalyzer(network, client).evaluate(network.stations, [0.625, 1.0])
    assert ev.arrival_cov[0] is None                   # level station: not measured
    scv = ev.arrival_cov[1] ** 2
    tol = 5 * 2 / math.sqrt(STOPPING["minSamples"] / 2)
    assert abs(scv - 1.0) <= tol, (scv, tol)


def test_deterministic_service_smooths_the_successors_arrivals(client):
    # For an M/G/1 the MARGINAL interdeparture SCV is exactly 1 - rho^2 (1 - cs^2):
    # condition on the queue being empty after a departure. At cs = 0, rho = 0.9 that is
    # 0.19 (an independent 390k-sample Python simulation gave 0.194). Successive
    # interdepartures are correlated, so the iid error understates this estimate's; the
    # 0.05 band is a judgement, and the direction (< 0.5, well below Poisson's 1) is the
    # claim that matters for phi.
    up = GG1Station.md1(mu=1.0, c=1.0, name="up")
    down = GG1Station.mm1(mu=1.0, c=1.0, name="down", zeta_mode=ZETA_SLOPE)
    network = _tandem(up, down, 0.5, "smoothing-tandem")
    ev = SimulationAnalyzer(network, client).evaluate(network.stations, [0.5 / 0.9, 1.0])
    scv = ev.arrival_cov[1] ** 2
    assert scv < 0.5, scv
    assert abs(scv - 0.19) <= 0.05, scv
```

- [ ] **Step 3: Run in the background**

Run (Bash `run_in_background`): `QOPT_QSIM_URL=$QOPT_QSIM_URL .venv/bin/python -m pytest tests/test_integration_qsim.py -v -k "burke or smooths" 2>&1 | tail -20`
Expected: both PASS. If the Burke tolerance fails, print `samplesAnalyzed` from the raw response before touching the tolerance; if the smoothing test misses 0.19 by more than the band, report the measured value — do not widen the band to pass.

- [ ] **Step 4: Sweep for hung children**

Run: `ps -o pid,ppid,etime,command -A | grep -i pytest | grep -v grep`
Expected: none left after the run.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_qsim.py
git commit -m "test(integration): witness measured arrival cov_a against live qsim

Burke (M/M/1 output is Poisson, SCV 1) and deterministic-service
smoothing (exact marginal interdeparture SCV 1 - rho^2 = 0.19).

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

### Task 7: Documentation ripple

**Files:**
- Modify: `README.md` (the "One caveat worth reading" paragraph, ~114-135; the simulation section near `SimulationAnalyzer`)
- Modify: `qopt/zeta.py` (module docstring; `ZETA_SHAPE_TOL` docstring)
- Modify: `qopt/station.py` (`Station.phi` docstring's override advice)
- Modify: `docs/superpowers/specs/2026-09-22-slope-calibrated-zeta-design.md` (§8.6)
- Modify: `docs/superpowers/specs/2026-09-25-measured-arrival-cov-design.md` (Status line)

**Interfaces:** none.

- [ ] **Step 1: README caveat paragraph**

Replace "it is never sent to the simulator and never measured back, so under slope calibration it must describe the arrival process the station *actually* sees, internal traffic included." with:

```
it is never sent to the simulator. **On a simulated run it is measured back**:
`SimulationAnalyzer` asks qsim-service for each slope-mode `GG1Station`'s interarrival
moments and prices `phi` at the measured `cov_a` for that evaluation
(`Result.arrival_cov` reports the final one). That needs qsim-service `8e7358b` or later
and costs roughly 25–30% wall clock; `SimulationAnalyzer(measure_cov_a=False)` turns it
off. **On the analytic path, or with measurement off,** `cov_a` must describe the arrival
process the station *actually* sees, internal traffic included.
```

Keep the rest of the paragraph (the `√(2 − cov_s²)` rule applies to the analytic / opt-out case). Replace its final sentence "If you know the true arrival variability but cannot express it as a `cov_a`, override `phi(S)` on a subclass." with "With a measurement, the cross-check evaluates the model *at* the measured `cov_a`, so a flag then means the G/G/1 approximation's shape disagrees, not the constructor argument. If you know the true arrival variability but cannot express it as a `cov_a`, override `phi(S, *, cov_a=None)` on a subclass."

- [ ] **Step 2: `qopt/zeta.py`**

Module docstring: after "why the SIMULATED path needs phi from the analytic model while E[T] stays measured." add "On that path `cov_a` is measured too (qsim-service `interarrival-time`, spec 2026-09-25), so phi's model sees the arrival variability each station actually gets."
`ZETA_SHAPE_TOL` docstring: add a paragraph: "When the analyzer measured a station's `cov_a`, the analytic side is evaluated AT the measurement, so the check then bounds the G/G/1 approximation's own error rather than a wrong constructor argument."

- [ ] **Step 3: `Station.phi` docstring**

Replace "Overridable: a user who knows the true arrival variability but cannot express it as a constructor `cov_a` should override this rather than reach for a new API." with "Overridable: a user who knows the true arrival variability but cannot express it as a constructor `cov_a` should override this rather than reach for a new API -- on the analytic path, or with `SimulationAnalyzer(measure_cov_a=False)`; a simulated run otherwise measures it. An override should accept `cov_a=None` if its class sets `reads_arrival_cov`."

- [ ] **Step 4: Slope spec §8.6 and this spec's status**

In `docs/superpowers/specs/2026-09-22-slope-calibrated-zeta-design.md` §8.6, append: "**Consumed 2026-09-25:** qsim-service#15 shipped as `interarrival-time` (`8e7358b`); see `2026-09-25-measured-arrival-cov-design.md`." In this feature's spec set `Status: **Implemented on feat/measured-arrival-cov.**`

- [ ] **Step 5: Figure-vs-source grep**

Run: `grep -n "8e7358b\|25–30%\|25-30%\|0.19\|measure_cov_a" README.md qopt/*.py qopt/qsim/*.py docs/superpowers/specs/2026-09-25-measured-arrival-cov-design.md`
Check every hit against its source (qsim-service README for the cost and version; Task 6 for 0.19). Also grep the new prose for exactness words (`exactly`, `always`, `never`, `bit-for-bit`) and confirm each is backed by a test.

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

```bash
git add README.md qopt/zeta.py qopt/station.py docs/superpowers/specs/2026-09-22-slope-calibrated-zeta-design.md docs/superpowers/specs/2026-09-25-measured-arrival-cov-design.md
git commit -m "docs: record that the simulated path measures arrival cov_a

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```
