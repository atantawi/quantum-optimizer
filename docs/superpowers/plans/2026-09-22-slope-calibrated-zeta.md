# Slope-calibrated ζ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user select, per station, whether ζ is calibrated to the *level* of that station's sojourn-time curve (`ζ = T·x`, eq 22, today's default) or to its *slope* (`ζ = φ·T·x`), which makes the optimizer's fixed point the coupled optimum exactly instead of an approximation of it.

**Architecture:** One new module holds the two mode constants and their validator. Stations gain a keyword-only `zeta_mode=`, a derivative hook `dT_dS(S)` (central difference in the base class, closed forms on `GG1Station` and `ForkJoinStation`), and `phi(S)`. A single branch in `Station.zeta_from` — the one seam where eq 22 is inverted — multiplies by φ in slope mode. `Result` reports ζ-as-used plus φ and the per-station mode. Nothing in `allocate`, `noise_floor`, or `Network` changes.

**Tech Stack:** Python ≥ 3.10, standard library only (`math`, `dataclasses`, `abc`, `warnings`). pytest for tests. No new dependencies — qopt ships zero runtime dependencies by design.

**Spec:** `docs/superpowers/specs/2026-09-22-slope-calibrated-zeta-design.md`

**Analysis the spec rests on:** `docs/slope-calibrated-zeta/findings.md` and `docs/slope-calibrated-zeta/probe-output.txt`. Section numbers in this plan refer to those files as committed; the reference objectives Task 8 asserts are in **probe-output.txt section 3**.

## Global Constraints

- **Zero runtime dependencies.** `pyproject.toml` has `dependencies = []` and that is policy, not accident. Standard library only in `qopt/`. pytest is a dev dependency and belongs only in `tests/`.
- **Python ≥ 3.10** (`requires-python = ">=3.10"`).
- **The default path stays bit-for-bit identical.** The default is `ZETA_LEVEL`; the level arm of `zeta_from` must compute the same operations in the same order as today's `T * (S * self.mu - self.gamma)`. Level-mode assertions use `==`, never `pytest.approx`.
- **Do not modify the paper.** `docs/analysis.pdf` is not edited, and no task changes it. qopt documents a deliberate divergence from eq 22 instead.
- **Baseline suite: 464 passed, 10 skipped in ~0.55s** (`python3 -m pytest -q` from the repo root, no `PYTHONPATH` needed). Every task ends with the full suite green, with 464 as a floor that only grows.
- **Every new assertion is mutation-checked**, and every mutation/restore cycle clears `__pycache__`. A same-second, same-size restore otherwise reuses the perturbed `.pyc` and both runs report identical counts, which looks like a vacuous test but is a broken harness.
- **No fenced code block in `docs/slope-calibrated-zeta/findings.md` may change.** `docs/quotes-selfcheck.py` requires every fenced quote there to match `probe-output.txt` verbatim. Task 10 edits prose only.
- **`docs/audit-selfcheck.py` needs `PYTHONPATH=.`** to run (`PYTHONPATH=. python3 docs/audit-selfcheck.py`). This is pre-existing.
- **Commit messages** end with `Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>`.

## Review Focus

Five things the spec implies but whose failure no task's own happy-path tests would catch. Each has a test assigned to the task that owns the code.

1. **The probe's reference networks carry non-unit `weight`s (`w=2.0`, `1.5`), and they are load-bearing in every objective number Task 8 asserts.** An implementer porting the station specs who drops the third positional argument gets numbers that miss by percent, and the tempting fix is to loosen the tolerance rather than restore the weights. eq 21 multiplies `w·ζ` under one square root, so a weight error and a φ error are indistinguishable from the objective alone. → Task 8, Step 1 asserts the *level* column first, which today's code already reproduces, so a port error fails before slope mode is ever exercised.
2. **`dT_dS` at or below the stability boundary must raise, not return a finite number.** The step `h = 1e-7*(S − γ/μ)` is zero at `S = γ/μ` and *negative* below it — where `S − h` is the *more* stable side, so only the `S + h` evaluation raises. A sign slip that evaluates `S − h` first, or an `abs(h)`, would silently return a derivative for an unstable station. → Task 3, Step 6.
3. **`phi` on a station whose γ is not yet bound** (constructed bare for a `Network`) must raise the canonical unbound-gamma `ValueError` naming the station, not `AttributeError` or a number computed from a `None`. → Task 3, Step 8.
4. **A legitimately tiny ζ must survive `allocate`.** A `cov_a = cov_s = 0` station has `φ = 1 − ρ` exactly, reaching ~1e-6 at extreme load, so `ζ_slope` can land near 1e-6 while `ζ_level` is ~1. `ZETA_FLOOR = 1e-12` is local to `noise_floor` and clamps nothing in `allocate`, which requires only finite and `> 0`. The run must complete, not raise and not silently clamp. → Task 6, Step 8.
5. **Slope mode on an all-M/M/1 network must be bit-for-bit identical to level mode.** φ ≡ 1 identically there, so any difference means the slope arm is not reducing to the identity — the single cheapest witness that φ is right, and it fails loudly if a closed form carries a stray factor. → Task 8, Step 7.

---

## File Structure

| File | Responsibility |
|---|---|
| `qopt/zeta.py` | **new.** The two mode constants, the cross-check default tolerance, and `resolve_zeta_mode`. Separate from `station.py` (491 lines) because the constants need the explanation of what level and slope calibration *mean* beside them, which is not station plumbing. Mirrors how `forkjoin_policy.py` holds the `R_STAR_*` constants. |
| `qopt/station.py` | `zeta_mode` plumbing, `dT_dS`, `phi`, and the one `zeta_from` branch. |
| `qopt/optimizer.py` | Three new `Result` fields, φ reporting, and the measured-vs-analytic cross-check. |
| `qopt/__init__.py` | Four new exports. |
| `tests/test_zeta.py` | **new.** Everything about ζ calibration: the φ invariants, both closed forms, the seam, reporting, end-to-end objectives, and the cross-check. One file because these all test one feature and change together. |
| `README.md`, `docs/slope-calibrated-zeta/*`, `docs/paper-map.md` | Documentation ripple. |

---

## Task 1: The mode constants and their validator

**Files:**
- Create: `qopt/zeta.py`
- Modify: `qopt/__init__.py`
- Test: `tests/test_zeta.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `ZETA_LEVEL = "level"`, `ZETA_SLOPE = "slope"`, `ZETA_MODES = (ZETA_LEVEL, ZETA_SLOPE)`, `ZETA_SHAPE_TOL = 0.25`, and `resolve_zeta_mode(mode) -> str`. Every later task imports these from `qopt.zeta`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_zeta.py`:

```python
"""ζ calibration: level (eq 22) and slope (docs/slope-calibrated-zeta/)."""

import math

import pytest

from qopt.zeta import (
    ZETA_LEVEL,
    ZETA_MODES,
    ZETA_SHAPE_TOL,
    ZETA_SLOPE,
    resolve_zeta_mode,
)


def test_the_two_modes_are_distinct_strings():
    assert ZETA_LEVEL == "level"
    assert ZETA_SLOPE == "slope"
    assert ZETA_MODES == (ZETA_LEVEL, ZETA_SLOPE)


def test_none_resolves_to_the_level_default():
    # The default must be the incumbent: an unspecified mode is eq 22, always.
    assert resolve_zeta_mode(None) == ZETA_LEVEL


def test_each_mode_resolves_to_itself():
    for mode in ZETA_MODES:
        assert resolve_zeta_mode(mode) == mode


def test_an_unknown_mode_is_rejected_and_the_message_names_the_alternatives():
    with pytest.raises(ValueError) as excinfo:
        resolve_zeta_mode("sloped")
    message = str(excinfo.value)
    assert "sloped" in message
    assert ZETA_LEVEL in message and ZETA_SLOPE in message


def test_a_non_string_mode_is_rejected():
    # A caller reaching for a bool or a number is confusing this with a flag.
    for bad in (True, 1, 1.0, [], object()):
        with pytest.raises(ValueError):
            resolve_zeta_mode(bad)


def test_the_shape_tolerance_default_is_a_fraction_not_a_percentage():
    # 0.25 means 25%. A value > 1 would mean the cross-check never fires.
    assert ZETA_SHAPE_TOL == 0.25
    assert math.isfinite(ZETA_SHAPE_TOL) and 0.0 < ZETA_SHAPE_TOL < 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'qopt.zeta'`.

- [ ] **Step 3: Write the module**

Create `qopt/zeta.py`:

```python
"""How ζ is calibrated to a station's sojourn-time curve.

eq 22 calibrates ζ to the curve's LEVEL: ζ = T*x with x = S*mu - gamma, so the surrogate
T_hat = zeta/x passes through the true (S, T) point. That is qopt's incumbent and the
default here.

eq 21, though, reads the surrogate only through its DERIVATIVE -- it water-fills on
marginal returns, and never evaluates T_hat itself. So the level calibration spends the
single free parameter on the one quantity the allocator does not look at. Calibrating the
SLOPE instead,

    zeta = phi * T * x = x**2 * |dT/dx|,   phi = |dT/dS| * x / (mu * T)

makes eq 21's stationarity condition hold on the TRUE slope at the current point, which
turns the loop's fixed point into the coupled optimum exactly rather than an approximation
of it. phi is the elasticity of E[T] in spare capacity, -d log T / d log x.

phi == 1 exactly for M/M/1, which is why the two calibrations were never distinguished:
it is the station type most of qopt's suite uses. phi = 1 - rho exactly for a cov = 0
station, and reaches 1.64 for G/G/1 with cov = 5.

This is a deliberate divergence from eq 22, not an amendment to it -- the paper is
unchanged. See docs/slope-calibrated-zeta/findings.md for the derivation, the measured
payoff, and why the SIMULATED path needs phi from the analytic model while E[T] stays
measured.
"""

ZETA_LEVEL = "level"
"""eq 22: ζ = T*x. Calibrates the surrogate's level. The default."""

ZETA_SLOPE = "slope"
"""ζ = phi*T*x: calibrates the surrogate's slope, the quantity eq 21 actually reads."""

ZETA_MODES = (ZETA_LEVEL, ZETA_SLOPE)
"""Every accepted calibration, in the order a message should list them."""

ZETA_SHAPE_TOL = 0.25
"""Default tolerance for the measured-vs-analytic E[T] cross-check (spec section 8.4).

Under slope calibration phi comes from the station's ANALYTIC model, so a badly wrong
model parameter -- `cov_a` describing an arrival process the station does not see -- buys
a converged, plausible, quietly suboptimal answer with no symptom. Comparing measured
against analytic E[T] tests exactly that assumption, and amplifies it: `cov_a = 3` where
the truth is 1 at rho = 0.67 is a 24% error in phi but a 268% error in E[T].

0.25 sits between the two scales this has been measured at: legitimate analytic-vs-
simulated disagreement is within +/-1.1% across the 14 stations of
docs/qcsc-example/live-run.log, while a wrong `cov_a` is hundreds of percent. It is
deliberately configurable, because that evidence contains no G/G/1 with cov != 1 -- the
very station type slope calibration most benefits.
"""


def resolve_zeta_mode(mode):
    """Validate a ζ calibration selection, mapping None to the default.

    Validation happens once, at construction, so `zeta_from` never has to consider an
    unknown mode: it branches on ZETA_SLOPE and treats anything else as level.

    Mirrors `forkjoin_policy.resolve_r_star`: a string constant, validated up front, named
    in the error message rather than left to the caller to guess.
    """
    if mode is None:
        return ZETA_LEVEL
    # `is not str` rather than a truthiness or equality test: True == 1 and both would
    # otherwise slip past an `in ZETA_MODES` check only by luck of not being "level".
    if type(mode) is not str or mode not in ZETA_MODES:
        raise ValueError(
            f"zeta_mode must be one of {ZETA_MODES!r} (or None for "
            f"{ZETA_LEVEL!r}), got {mode!r}"
        )
    return mode
```

- [ ] **Step 4: Add the exports**

In `qopt/__init__.py`, add the import after the `from qopt.station import ...` line:

```python
from qopt.zeta import ZETA_LEVEL, ZETA_MODES, ZETA_SHAPE_TOL, ZETA_SLOPE, resolve_zeta_mode
```

and add to `__all__`, immediately after `"optimal_ray",`:

```python
    "ZETA_LEVEL",
    "ZETA_SLOPE",
    "ZETA_MODES",
    "ZETA_SHAPE_TOL",
    "resolve_zeta_mode",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 6 passed.

Run: `python3 -m pytest -q`
Expected: 470 passed, 10 skipped.

- [ ] **Step 6: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/zeta.py")
s = p.read_text()
# Mutation: accept anything, the way a bare attribute assignment would.
s = s.replace("    if type(mode) is not str or mode not in ZETA_MODES:",
              "    if False:", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q          # EXPECT: 2 failed (unknown mode, non-string)
git checkout -- qopt/zeta.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q          # EXPECT: 6 passed
```

If the two runs report the same counts, the harness is broken, not the test — `__pycache__` was not cleared.

- [ ] **Step 7: Commit**

```bash
git add qopt/zeta.py qopt/__init__.py tests/test_zeta.py
git commit -m "feat: add the zeta calibration mode constants and their validator

qopt/zeta.py holds ZETA_LEVEL (eq 22, the default) and ZETA_SLOPE, plus
resolve_zeta_mode, which mirrors forkjoin_policy.resolve_r_star: a string
constant validated once at construction and named in the error message.

No behaviour change yet -- nothing consumes these.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 2: Thread `zeta_mode` through the station hierarchy

**Files:**
- Modify: `qopt/station.py` (`Station.__init__` ~:52, `SingleServerStation.__init__` ~:208, `GG1Station.__init__` ~:231, `GG1Station.mm1`/`md1` ~:248/:253, `ForkJoinStation.__init__` ~:316)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: `resolve_zeta_mode`, `ZETA_LEVEL`, `ZETA_SLOPE` from `qopt.zeta`.
- Produces: keyword-only `zeta_mode=ZETA_LEVEL` on `Station`, `SingleServerStation`, `GG1Station`, `GG1Station.mm1`, `GG1Station.md1`, `ForkJoinStation`; the instance attribute `_zeta_mode`; and a read-only property `Station.zeta_mode`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zeta.py`:

```python
from qopt.station import ForkJoinStation, GG1Station, Station


def _every_station_kind(**kwargs):
    """One instance of each concrete station type, all constructor kwargs forwarded."""
    return [
        GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0, name="gg1", **kwargs),
        GG1Station.mm1(0.6, 1.5, c=2.0, name="mm1", **kwargs),
        GG1Station.md1(0.6, 1.5, c=2.0, name="md1", **kwargs),
        ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, name="fj", **kwargs),
        ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                        name="fj-tuned", **kwargs),
    ]


def test_every_station_kind_defaults_to_level():
    for st in _every_station_kind():
        assert st.zeta_mode == ZETA_LEVEL, st.name


def test_every_station_kind_accepts_slope():
    for st in _every_station_kind(zeta_mode=ZETA_SLOPE):
        assert st.zeta_mode == ZETA_SLOPE, st.name


def test_an_unknown_mode_is_rejected_at_construction_by_every_kind():
    for ctor in (
        lambda **kw: GG1Station(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=1.0, **kw),
        lambda **kw: GG1Station.mm1(0.6, 1.5, c=2.0, **kw),
        lambda **kw: GG1Station.md1(0.6, 1.5, c=2.0, **kw),
        lambda **kw: ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, **kw),
    ):
        with pytest.raises(ValueError, match="zeta_mode"):
            ctor(zeta_mode="slopes")


def test_zeta_mode_is_read_only():
    # Read-only for the reason ForkJoinStation.policy is: nothing should change a
    # calibration mid-run, when some iterations have already been priced the other way.
    st = GG1Station.mm1(0.6, 1.5, c=2.0)
    with pytest.raises(AttributeError):
        st.zeta_mode = ZETA_SLOPE


def test_zeta_mode_is_keyword_only():
    # Positional would collide with `name` and with every subclass's own signature.
    with pytest.raises(TypeError):
        GG1Station(0.6, 1.5, 1.0, ZETA_SLOPE)  # type: ignore[misc]


def test_the_mode_survives_a_forkjoin_retune_and_reset():
    # retune/reset_policy rewrite mu, r and r_star. The calibration is not policy state
    # and must not be touched by either.
    st = ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                         zeta_mode=ZETA_SLOPE)
    st.retune(3.0)
    assert st.zeta_mode == ZETA_SLOPE
    st.reset_policy()
    assert st.zeta_mode == ZETA_SLOPE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'zeta_mode'`.

- [ ] **Step 3: Add the plumbing**

In `qopt/station.py`, extend the import block at the top:

```python
from qopt.zeta import ZETA_LEVEL, ZETA_SLOPE, resolve_zeta_mode
```

`Station.__init__` — change the signature and add one line at the end of the body:

```python
    def __init__(self, gamma=None, mu=None, weight=1.0, *, name=None,
                 zeta_mode=ZETA_LEVEL):
```

after `self.name = name`:

```python
        # Validated here so `zeta_from` can branch on ZETA_SLOPE alone and treat
        # anything else as level -- an unknown mode cannot reach the hot path.
        self._zeta_mode = resolve_zeta_mode(zeta_mode)
```

Add the property immediately after the `gamma` property:

```python
    @property
    def zeta_mode(self):
        """Which ζ calibration this station uses (a qopt.ZETA_* constant).

        Read-only and fixed at construction, for the reason `ForkJoinStation.policy` is:
        a run prices iterations against a calibration, so changing it mid-run would leave
        a converged ζ that no single rule produced.
        """
        return self._zeta_mode
```

`SingleServerStation.__init__`:

```python
    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, name=None,
                 zeta_mode=ZETA_LEVEL):
        super().__init__(gamma, mu, weight, name=name, zeta_mode=zeta_mode)
```

`GG1Station.__init__`:

```python
    def __init__(self, gamma=None, mu=None, weight=1.0, *, c, cov_a, cov_s, name=None,
                 zeta_mode=ZETA_LEVEL):
        super().__init__(gamma, mu, weight, c=c, name=name, zeta_mode=zeta_mode)
```

`GG1Station.mm1` and `GG1Station.md1`:

```python
    @classmethod
    def mm1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None,
            zeta_mode=ZETA_LEVEL):
        """M/M/1 preset (cov_a = cov_s = 1); zeta is identically 1.

        phi is identically 1 too, so `zeta_mode` makes no difference on this station --
        the two calibrations agree exactly. It is accepted so a network can be switched
        wholesale without special-casing its M/M/1 members.
        """
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=1.0, name=name,
                   zeta_mode=zeta_mode)

    @classmethod
    def md1(cls, gamma=None, mu=None, weight=1.0, *, c, name=None,
            zeta_mode=ZETA_LEVEL):
        """M/D/1 preset (cov_a = 1, cov_s = 0); zeta = 1 - rho/2."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=0.0, name=name,
                   zeta_mode=zeta_mode)
```

`ForkJoinStation.__init__` — signature and the `super().__init__` call:

```python
    def __init__(self, gamma=None, mu=None, weight=1.0, *, r, c1, c2, r_star=None,
                 name=None, zeta_mode=ZETA_LEVEL):
```

```python
        super().__init__(gamma, mu if mu is None else mu * k, weight, name=name,
                         zeta_mode=zeta_mode)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 12 passed.

Run: `python3 -m pytest -q`
Expected: 476 passed, 10 skipped. **No existing test may change**, since nothing yet reads `_zeta_mode`.

- [ ] **Step 5: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 1: drop the validation, so a bad mode is stored silently.
s = s.replace("self._zeta_mode = resolve_zeta_mode(zeta_mode)",
              "self._zeta_mode = zeta_mode", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: 1 failed (rejected at construction)
git checkout -- qopt/station.py

python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 2: make the property writable, which is how a plain attribute would behave.
s = s.replace("    @property\n    def zeta_mode(self):", "    def _unused(self):\n        pass\n\n    def zeta_mode(self):", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: failures (mode is a bound method, not a string)
git checkout -- qopt/station.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: 476 passed, 10 skipped
```

- [ ] **Step 6: Commit**

```bash
git add qopt/station.py tests/test_zeta.py
git commit -m "feat: accept a per-station zeta_mode, defaulting to level

Keyword-only and read-only, threaded through every concrete station type
including the mm1/md1 presets. NOT named `zeta=`: Station.zeta(S) is already a
method, so an attribute of that name would shadow it.

Validated once at construction, so zeta_from can branch on ZETA_SLOPE alone and
treat anything else as level. Nothing reads the mode yet -- this commit is pure
plumbing and the suite is unchanged.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 3: `dT_dS` central difference and `phi` on the base class

**Files:**
- Modify: `qopt/station.py` (add `_FD_STEP` near the top; add `dT_dS` and `phi` to `Station`, immediately before `zeta_from`)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: `Station.sojourn_time`, `Station.gamma`, `Station.mu` from Task 2's hierarchy.
- Produces: `Station.dT_dS(S) -> float` (negative for every station type shipped) and `Station.phi(S) -> float`. Tasks 4 and 5 override `dT_dS`; Task 6 calls `phi`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zeta.py`:

```python
from qopt.exceptions import InstabilityError


def test_phi_is_identically_one_for_mm1():
    # THE invariant that explains why the two calibrations were never distinguished:
    # M/M/1 is the station type most of the suite uses, and there eq 22 is already
    # slope-correct. Any closed form that breaks this is wrong.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    for S in (0.61, 0.8, 1.0, 2.5, 10.0, 1e4):
        assert st.phi(S) == pytest.approx(1.0, abs=1e-9), S


def test_phi_is_one_minus_rho_for_a_zero_cov_station():
    # cov_a = cov_s = 0 gives E[T] = 1/m with no queueing term, so spare capacity buys
    # almost nothing and phi collapses toward 0 -- exactly 1 - rho.
    st = GG1Station(0.6, 1.0, c=1.0, cov_a=0.0, cov_s=0.0)
    for S in (0.7, 1.0, 2.0, 12.0):
        rho = st.gamma / (S * st.mu)
        assert st.phi(S) == pytest.approx(1.0 - rho, abs=1e-8), S


def test_dt_ds_is_negative_for_every_station_kind():
    # More capacity cannot lengthen the sojourn time. A positive derivative would make
    # phi negative and eq 21's sqrt(zeta) complex.
    for st in _every_station_kind():
        assert st.dT_dS(3.0) < 0.0, st.name


def test_phi_is_strictly_positive_for_every_station_kind():
    for st in _every_station_kind():
        assert st.phi(3.0) > 0.0, st.name


def test_the_finite_difference_default_serves_a_subclass_with_no_closed_form():
    # The base implementation is deliberately concrete, not abstract, so slope
    # calibration works for a user station whose derivative qopt has never seen.
    class QuadraticStation(Station):
        """E[T] = 1/x**2 -- not a queue qopt ships, which is the point."""

        def sojourn_time(self, S):
            m = S * self.mu
            self._check_stable(m)
            return 1.0 / (m - self.gamma) ** 2

        def sim_node(self, S, job_class):
            raise NotImplementedError

        @property
        def alloc_cost(self):
            return 1.0

        @property
        def default_zeta(self):
            return 1.0

    st = QuadraticStation(gamma=0.5, mu=1.0)
    # T = x**-2 so dT/dx = -2 x**-3 and phi = -dT/dS * x/(mu T) = 2, at every S.
    for S in (0.6, 1.0, 4.0):
        assert st.phi(S) == pytest.approx(2.0, rel=1e-6), S


def test_dt_ds_refuses_an_unstable_capacity():
    # Review Focus 2. At S == gamma/mu the step h is 0; below it h is NEGATIVE, so
    # S - h is the MORE stable side and only S + h raises. An abs(h), or evaluating
    # S - h first and returning early, would hand back a derivative for a station that
    # has no sojourn time at all.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    boundary = st.gamma / st.mu           # 0.6
    for S in (boundary, boundary * 0.5, boundary - 1e-12):
        with pytest.raises(InstabilityError):
            st.dT_dS(S)
        with pytest.raises(InstabilityError):
            st.phi(S)


def test_the_finite_difference_step_stays_inside_the_stability_region():
    # Scaling h to SPARE CAPACITY rather than to S is what guarantees S - h > gamma/mu.
    # A fixed step would fall off the boundary for a station run close to it.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    assert st.dT_dS(0.6 + 1e-9) < 0.0          # a hair above the boundary, still fine


def test_phi_on_an_unbound_gamma_names_the_station():
    # Review Focus 3. A station built bare for a Network has no gamma until bind_gamma.
    # The canonical ValueError must survive -- not AttributeError, and certainly not a
    # number computed from a None.
    st = GG1Station(mu=1.0, c=1.0, cov_a=1.0, cov_s=1.0, name="unbound")
    with pytest.raises(ValueError, match="unbound"):
        st.phi(2.0)
    with pytest.raises(ValueError, match="unbound"):
        st.dT_dS(2.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: FAIL — `AttributeError: 'GG1Station' object has no attribute 'phi'`.

- [ ] **Step 3: Implement the two hooks**

In `qopt/station.py`, after the `distribution_dict` function, add:

```python
_FD_STEP = 1e-7
"""Central-difference step for `Station.dT_dS`, as a FRACTION OF SPARE CAPACITY.

Scaled to spare capacity rather than to S, which is what keeps `S - h` strictly inside the
stability region for a station run arbitrarily close to its boundary. 1e-7 is near the
cube-root-of-epsilon optimum for a central difference and is measured to agree with the
closed forms to 6.7e-09 relative (docs/slope-calibrated-zeta/probe-output.txt section 1).
"""
```

Add to `Station`, immediately before `zeta_from`:

```python
    def dT_dS(self, S):
        """dE[T]/dS at capacity S -- negative, since capacity cannot lengthen a queue.

        Only slope-calibrated ζ reads this, so a level-mode station never pays for it.

        Deliberately concrete rather than abstract: slope calibration then works for ANY
        station, including a user subclass qopt has never seen, and no existing subclass
        breaks. `GG1Station` and `ForkJoinStation` override it with closed forms, which
        this default is tested against.

        The step is a fraction of SPARE CAPACITY, `S - gamma/mu`, which buys two things a
        fixed step does not: `S - h` is inside the stability region by construction for
        any stable S, and at `S == gamma/mu` the step is 0 so `sojourn_time` raises
        InstabilityError rather than this dividing by zero. Below the boundary h is
        negative, and the `S + h` evaluation is the one that raises -- so do not reorder
        these two calls or take an absolute value.

        For a fork-join this differences along the station's FIXED CURRENT RAY, because
        `sojourn_time` scales both servers with S. That is the radial derivative slope
        calibration needs (see ForkJoinStation.dT_dS), so the default cannot accidentally
        reproduce the `forkjoin_policy._dt_dm1` defect.
        """
        h = _FD_STEP * (S - self.gamma / self.mu)
        return (self.sojourn_time(S + h) - self.sojourn_time(S - h)) / (2.0 * h)

    def phi(self, S):
        """Elasticity of E[T] in spare capacity: -d log T / d log x, with x = S*mu - gamma.

        The ratio of the true slope to the one eq 22's level calibration implies, so
        `phi == 1` means eq 22 is already slope-correct at S and the two calibrations
        agree. Identically 1 for M/M/1; exactly `1 - rho` for a cov = 0 station; up to
        1.64 for G/G/1 with cov = 5.

        Uses this station's ANALYTIC `sojourn_time`, always -- including on the simulated
        path, where `zeta_from` receives a MEASURED E[T] and this supplies only the shape.
        That hybrid is load-bearing in both directions (spec section 4.4): a fully
        analytic slope ζ cancels T and would cut the simulator out of the allocation
        entirely, while a secant slope from consecutive loop iterates degenerates to 0/0
        as they converge. It also costs no simulation calls.

        Overridable: a user who knows the true arrival variability but cannot express it
        as a constructor `cov_a` should override this rather than reach for a new API.
        """
        x = S * self.mu - self.gamma
        return -self.dT_dS(S) * x / (self.mu * self.sojourn_time(S))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 20 passed.

Run: `python3 -m pytest -q`
Expected: 484 passed, 10 skipped.

- [ ] **Step 5: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 1: absolute-value the step, which hides an unstable S (Review Focus 2).
s = s.replace("        h = _FD_STEP * (S - self.gamma / self.mu)",
              "        h = abs(_FD_STEP * (S - self.gamma / self.mu))", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: test_dt_ds_refuses_an_unstable_capacity fails
git checkout -- qopt/station.py

python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 2: drop the x factor from phi -- a dimensional error that leaves phi
# plausible-looking and positive.
s = s.replace("        return -self.dT_dS(S) * x / (self.mu * self.sojourn_time(S))",
              "        return -self.dT_dS(S) / (self.mu * self.sojourn_time(S))", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: the phi invariants fail
git checkout -- qopt/station.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: 484 passed, 10 skipped
```

- [ ] **Step 6: Commit**

```bash
git add qopt/station.py tests/test_zeta.py
git commit -m "feat: add dT_dS and phi hooks to Station

dT_dS is a central difference in the base class -- concrete, not abstract, so
slope calibration works for a user subclass qopt has never seen. The step is a
fraction of SPARE capacity, which keeps S-h inside the stability region for a
station run arbitrarily close to its boundary and makes an unstable S raise
rather than return a number.

phi is the elasticity of E[T] in spare capacity, and reads this station's
ANALYTIC sojourn_time even when E[T] is measured: a fully analytic slope zeta
would cancel T and cut the simulator out of the allocation entirely.

Pinned: phi == 1 identically for M/M/1 -- the invariant that explains why the
two calibrations were never distinguished -- and phi == 1 - rho for cov = 0.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 4: `GG1Station.dT_dS` closed form

**Files:**
- Modify: `qopt/station.py` (add `dT_dS` to `GG1Station`, after `sojourn_time` ~:246)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: `Station.dT_dS` (the default it overrides and is tested against), `Station._check_stable`.
- Produces: `GG1Station.dT_dS(S)`, same signature and sign convention as the base.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zeta.py`:

```python
def _central_difference(st, S, frac=1e-7):
    """The base-class difference, computed independently of whatever dT_dS now does."""
    h = frac * (S - st.gamma / st.mu)
    return (st.sojourn_time(S + h) - st.sojourn_time(S - h)) / (2.0 * h)


def test_gg1_closed_form_matches_the_central_difference():
    # Over loads AND coefficients of variation: the closed form has a k*gamma*(2m-gamma)
    # term whose sign and grouping a single (cov, load) pair cannot pin.
    for cov_a in (0.0, 0.5, 1.0, 2.0, 5.0):
        for cov_s in (0.0, 1.0, 3.0):
            st = GG1Station(0.6, 1.5, c=1.0, cov_a=cov_a, cov_s=cov_s)
            for S in (0.45, 0.6, 1.0, 3.0, 20.0):
                assert st.dT_dS(S) == pytest.approx(
                    _central_difference(st, S), rel=1e-6
                ), (cov_a, cov_s, S)


def test_gg1_phi_is_exactly_one_for_mm1_under_the_closed_form():
    # Now abs=1e-12 rather than the difference-limited 1e-9 of the fd default: the
    # closed form should make this invariant hold to machine precision.
    st = GG1Station.mm1(0.6, 1.0, c=2.0)
    for S in (0.61, 1.0, 2.5, 100.0, 1e6):
        assert st.phi(S) == pytest.approx(1.0, abs=1e-12), S


def test_gg1_phi_is_exactly_one_minus_rho_for_cov_zero_under_the_closed_form():
    st = GG1Station(0.6, 1.0, c=1.0, cov_a=0.0, cov_s=0.0)
    for S in (0.7, 1.0, 2.0, 50.0):
        rho = st.gamma / (S * st.mu)
        assert st.phi(S) == pytest.approx(1.0 - rho, rel=1e-12), S


def test_gg1_phi_matches_the_documented_sensitivity_table():
    # Spec section 8.2, and this test is that table's source of record. gamma = 0.6,
    # cov_s = 1, phi read at three loads for a true and a mis-specified cov_a.
    #
    # It is here because under slope calibration `cov_a` stops being decorative on the
    # simulated path -- it is never sent to qsim and never measured back, so it enters
    # the allocation only through phi. Understating it drives phi toward 1 and degrades
    # to level calibration; OVERSTATING it can land worse than the incumbent.
    rows = [
        # rho,  cov_a, expected phi
        (0.30, 1.0, 1.000000),
        (0.30, 3.0, 1.381818),
        (0.67, 1.0, 1.000000),
        (0.67, 3.0, 1.240030),
        (0.95, 1.0, 1.000000),
        (0.95, 3.0, 1.039697),
        (0.67, 2.0, 1.165419),
    ]
    gamma = 0.6
    for rho, cov_a, expected in rows:
        st = GG1Station(gamma, 1.0, c=1.0, cov_a=cov_a, cov_s=1.0)
        S = gamma / rho / st.mu          # m = gamma/rho
        assert st.phi(S) == pytest.approx(expected, rel=1e-6), (rho, cov_a)


def test_a_wrong_cov_a_moves_expected_sojourn_time_far_more_than_it_moves_phi():
    # Why the measured-vs-analytic E[T] cross-check (Task 9) is the right detector: the
    # symptom is roughly ten times larger than the defect it indicates.
    S = 0.6 / 0.67
    truth = GG1Station(0.6, 1.0, c=1.0, cov_a=1.0, cov_s=1.0)
    wrong = GG1Station(0.6, 1.0, c=1.0, cov_a=3.0, cov_s=1.0)
    phi_error = wrong.phi(S) / truth.phi(S) - 1.0
    t_error = wrong.sojourn_time(S) / truth.sojourn_time(S) - 1.0
    assert phi_error == pytest.approx(0.240, abs=0.005)
    assert t_error == pytest.approx(2.68, abs=0.02)
    assert t_error > 10 * phi_error
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: `test_gg1_phi_is_exactly_one_for_mm1_under_the_closed_form` and the `rel=1e-12` cov-zero test FAIL (the fd default is only good to ~1e-9); the sensitivity-table test passes already, since φ is correct either way — it is pinning the *numbers*, not the implementation.

- [ ] **Step 3: Implement the closed form**

Add to `GG1Station`, immediately after `sojourn_time`:

```python
    def dT_dS(self, S):
        """Closed form of the Allen-Cunneen derivative.

            E[T] = 1/m + k*gamma/(m*x),   m = S*mu,  x = m - gamma,  k = (cov_a^2+cov_s^2)/2

        differentiated in S:

            dT/dS = -mu * [ 1/m^2 + k*gamma*(2m - gamma)/(m*x)^2 ]

        Every term is negative, so no sign can cancel silently. At k = 1 this gives
        phi == 1 exactly, which is the M/M/1 invariant; at k = 0 it gives phi = 1 - rho.
        """
        m = S * self.mu
        self._check_stable(m)
        x = m - self.gamma
        k = (self.cov_a ** 2 + self.cov_s ** 2) / 2.0
        return -self.mu * (
            1.0 / m ** 2 + k * self.gamma * (2.0 * m - self.gamma) / (m * x) ** 2
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 25 passed.

Run: `python3 -m pytest -q`
Expected: 489 passed, 10 skipped.

- [ ] **Step 5: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation: (2m - gamma) -> (2m + gamma), a plausible sign slip that leaves the
# derivative negative and the magnitude close at low load.
s = s.replace("1.0 / m ** 2 + k * self.gamma * (2.0 * m - self.gamma) / (m * x) ** 2",
              "1.0 / m ** 2 + k * self.gamma * (2.0 * m + self.gamma) / (m * x) ** 2", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: the fd-agreement and phi invariants fail
git checkout -- qopt/station.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: 25 passed
```

- [ ] **Step 6: Commit**

```bash
git add qopt/station.py tests/test_zeta.py
git commit -m "feat: closed-form dT_dS for GG1Station

Differentiates the Allen-Cunneen mean-value form in S. Checked against the
base-class central difference over five coefficients of variation and five
loads, and it makes the two phi boundary cases exact rather than
difference-limited: phi == 1 for M/M/1 and phi == 1 - rho for cov = 0.

Also pins the spec's cov_a sensitivity table, which is what makes this test its
source of record: under slope calibration cov_a stops being decorative on the
simulated path, since it is never sent to qsim and never measured back, so it
reaches the allocation only through phi.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 5: `ForkJoinStation.dT_dS` — the radial closed form

**Files:**
- Modify: `qopt/station.py` (add `dT_dS` to `ForkJoinStation`, after `sojourn_time` ~:470)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: `ForkJoinStation.mu` and `.r` (EFFECTIVE, post-`_anchor`), `Station._check_stable`, `forkjoin_approx.t_ul`'s structure.
- Produces: `ForkJoinStation.dT_dS(S)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zeta.py`:

```python
def test_forkjoin_closed_form_matches_the_central_difference():
    # Across r AND r_star, because r_star rewrites which server binds: r_star < 1 swaps
    # the anchor, and the closed form reads the EFFECTIVE mu and r, not the constructed
    # ones. r_star = 1.0 is included deliberately -- see the next test.
    for r in (1.0, 2.0, 4.0, 20.0):
        for r_star in (0.05, 1.0, 2.0, 3.0, 50.0):
            st = ForkJoinStation(0.45, 1.0, r=r, c1=4.0, c2=1.0, r_star=r_star)
            base = st.gamma / st.mu
            for mult in (1.01, 1.5, 4.0, 100.0):
                S = base * mult
                assert st.dT_dS(S) == pytest.approx(
                    _central_difference(st, S), rel=1e-5
                ), (r, r_star, mult)


def test_forkjoin_closed_form_is_right_where_the_policy_helper_is_wrong():
    # forkjoin_policy._dt_dm1 takes the "m1 is the non-bottleneck" branch at m1 == m2 and
    # drops the alpha*t_bot term entirely. That is harmless inside _min_on_spend_line,
    # whose kink is measure-zero there, but it is wrong for pricing dT/d(spend) -- and
    # r_star = 1 is exactly where tight budgets and every beta1 == beta2 station sit.
    # Measured at 17.3% for spend/floor = 1.05 (findings.md section 5).
    #
    # This test does not reimplement the helper; it pins that the radial derivative
    # agrees with a difference of the SHIPPED sojourn_time at r_star = 1, which is the
    # property the helper lacks.
    st = ForkJoinStation(0.45, 1.0, r=1.0, c1=4.0, c2=1.0, r_star=1.0)
    base = st.gamma / st.mu
    for mult in (1.05, 1.5, 4.0):
        S = base * mult
        assert st.dT_dS(S) == pytest.approx(_central_difference(st, S), rel=1e-5), mult


def test_forkjoin_phi_stays_near_one_but_not_at_one():
    # The fork-join is the mildest deviation of the station types qopt ships -- within
    # about 1.3% of 1 -- which is why a network whose only non-M/M/1 stations are
    # fork-joins gains only 0.0002-0.039%. It is still NOT 1, so it still moves.
    st = ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0)
    base = st.gamma / st.mu
    values = [st.phi(base * m) for m in (1.01, 1.5, 4.0, 100.0)]
    assert all(0.95 < v < 1.05 for v in values), values
    assert any(abs(v - 1.0) > 1e-6 for v in values), values


def test_forkjoin_phi_is_positive_over_a_wide_grid():
    # Spec assumption 2, and eq 21 needs a strictly positive zeta: sqrt(w*zeta/...).
    for r in (1.0, 2.0, 20.0):
        for r_star in (0.05, 1.0, 3.0, 50.0):
            st = ForkJoinStation(0.45, 1.0, r=r, c1=4.0, c2=1.0, r_star=r_star)
            base = st.gamma / st.mu
            for mult in (1.000001, 1.001, 1.1, 2.0, 10.0, 1e4, 1e8):
                assert st.phi(base * mult) > 0.0, (r, r_star, mult)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: the four new tests PASS already, because the base-class difference is what they compare against and it is currently what `dT_dS` returns. **This is the one place in the plan where the test cannot fail first.** So instead:

- [ ] **Step 3: Establish that the tests bite, by writing a deliberately wrong override first**

Add this to `ForkJoinStation` and run the tests:

```python
    def dT_dS(self, S):
        return -1.0      # placeholder, deliberately wrong
```

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: the four fork-join tests FAIL. This proves they constrain the override and not merely the default. Leave the placeholder in place for Step 4 to replace.

- [ ] **Step 4: Implement the radial closed form**

Replace the placeholder with:

```python
    def dT_dS(self, S):
        """Radial derivative of `t_ul` along this station's CURRENT ray.

        Both effective rates scale with S, so this differentiates
        t_ul(gamma, a*S, b*S) in S with a = mu and b = mu*r held fixed. That is the
        derivative slope calibration needs, and it equals the true marginal of the
        coupled problem only ON the optimal ray -- which the Optimizer guarantees by
        calling `retune` LAST in each iteration.

        `t_bot` needs no max() here. `_anchor` pairs `mu` with the slower server and
        keeps `r >= 1`, so m1 <= m2 always and `t_ul`'s max() resolves to 1/x1 at every
        point of the ray. The branch therefore never switches and this is smooth, which
        is also why differencing `sojourn_time` agrees with it.

        NOT `forkjoin_policy._dt_dm1`: that takes the non-bottleneck branch at m1 == m2
        and drops the alpha*t_bot term, which is fine for the measure-zero kink inside
        `_min_on_spend_line` but wrong for pricing dT/d(spend) -- a 17.3% error at
        spend/floor = 1.05, and r_star = 1 is where tight budgets sit.

        alpha is homogeneous of degree -1 in S, hence the -alpha/S term.
        """
        a, b = self.mu, self.mu * self.r      # effective rates: a binds, b >= a
        m1, m2 = a * S, b * S
        self._check_stable(m1)
        x1, x2 = m1 - self.gamma, m2 - self.gamma
        D = x1 + x2
        t_ub = 1.0 / x1 + 1.0 / x2 - 1.0 / D
        t_bot = 1.0 / x1
        alpha = (self.gamma / m1 + self.gamma / m2) / 8.0
        d_ub = -a / x1 ** 2 - b / x2 ** 2 + (a + b) / D ** 2
        d_bot = -a / x1 ** 2
        return (alpha / S) * (t_ub - t_bot) + (1.0 - alpha) * d_ub + alpha * d_bot
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 29 passed.

Run: `python3 -m pytest -q`
Expected: 493 passed, 10 skipped.

- [ ] **Step 6: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 1: drop the dalpha/dS term -- the exact omission _dt_dm1 makes.
s = s.replace("        return (alpha / S) * (t_ub - t_bot) + (1.0 - alpha) * d_ub + alpha * d_bot",
              "        return (1.0 - alpha) * d_ub + alpha * d_bot", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: the fd-agreement tests fail
git checkout -- qopt/station.py

python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 2: use the CONSTRUCTED ratio instead of the effective one, so r_star < 1
# silently prices the wrong server.
s = s.replace("        a, b = self.mu, self.mu * self.r      # effective rates: a binds, b >= a",
              "        a, b = self.mu, self.mu * self.r_base", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: the r_star sweep fails
git checkout -- qopt/station.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: 493 passed, 10 skipped
```

- [ ] **Step 7: Commit**

```bash
git add qopt/station.py tests/test_zeta.py
git commit -m "feat: radial closed-form dT_dS for ForkJoinStation

Differentiates t_ul along the station's current ray, reading the EFFECTIVE mu and
r so an r_star < 1 that swaps which server binds is priced correctly. t_bot needs
no max(): _anchor keeps m1 <= m2 at every point of a ray, so the branch never
switches and the derivative is smooth.

Deliberately not forkjoin_policy._dt_dm1, which drops the alpha*t_bot term at
m1 == m2 -- harmless for the measure-zero kink in _min_on_spend_line, wrong by
17.3% for pricing dT/d(spend) at spend/floor = 1.05, and r_star = 1 is exactly
where tight budgets sit.

Checked against a difference of the shipped sojourn_time across four r values and
five rays, r_star = 1.0 included.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 6: The `zeta_from` slope branch

**Files:**
- Modify: `qopt/station.py` (`Station.zeta_from` ~:145)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: `Station.phi`, `self._zeta_mode`, `ZETA_SLOPE`.
- Produces: `Station.zeta_from(T, S)` returning `φ·T·x` in slope mode and `T·x` otherwise. `Station.zeta(S)` inherits the behaviour unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zeta.py`:

```python
def test_level_mode_is_bit_for_bit_the_shipped_expression():
    # `==`, not approx: the default path must not move by one ulp. Binding x to a local
    # does not change float semantics, but reordering the multiply would.
    for st in _every_station_kind():
        for T in (0.5, 2.5, 137.125):
            for S in (2.0, 7.5):
                assert st.zeta_from(T, S) == T * (S * st.mu - st.gamma), (st.name, T, S)


def test_slope_mode_is_phi_times_the_level_value():
    for st in _every_station_kind(zeta_mode=ZETA_SLOPE):
        for S in (2.0, 7.5):
            T = st.sojourn_time(S)
            level = T * (S * st.mu - st.gamma)
            assert st.zeta_from(T, S) == pytest.approx(st.phi(S) * level, rel=1e-15), st.name


def test_slope_zeta_is_x_squared_times_the_slope_in_x():
    # The identity that makes eq 21 exact: zeta/x has derivative -|dT/dx| in x, so
    # zeta = x**2 * |dT/dx|. Checked against dT_dS with the mu factor undone.
    st = GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0, zeta_mode=ZETA_SLOPE)
    for S in (0.5, 1.0, 4.0):
        x = S * st.mu - st.gamma
        dT_dx = st.dT_dS(S) / st.mu
        assert st.zeta_from(st.sojourn_time(S), S) == pytest.approx(
            x ** 2 * abs(dT_dx), rel=1e-12
        ), S


def test_mm1_slope_and_level_zeta_agree_exactly():
    # phi == 1 identically, so the two calibrations must coincide -- and since phi is
    # computed, not assumed, this also pins that the closed form returns exactly 1.
    level = GG1Station.mm1(0.6, 1.0, c=2.0)
    slope = GG1Station.mm1(0.6, 1.0, c=2.0, zeta_mode=ZETA_SLOPE)
    for S in (0.7, 1.0, 5.0):
        T = level.sojourn_time(S)
        assert slope.zeta_from(T, S) == pytest.approx(level.zeta_from(T, S), rel=1e-12)


def test_zeta_from_is_linear_in_T_in_both_modes():
    # The ONLY property Optimizer._noise_floor relies on: it propagates a CI half-width
    # through this same hook, so zeta_from(h, S) must be the correctly scaled
    # perturbation. If slope mode were not linear in T, the noise floor would need
    # special-casing and this change would not be cheap.
    for mode in ZETA_MODES:
        for st in _every_station_kind(zeta_mode=mode):
            for S in (2.0, 9.0):
                base = st.zeta_from(1.0, S)
                assert st.zeta_from(2.0, S) == pytest.approx(2.0 * base, rel=1e-14)
                assert st.zeta_from(0.25, S) == pytest.approx(0.25 * base, rel=1e-14)


def test_the_station_zeta_method_follows_the_mode():
    st_l = GG1Station.md1(0.6, 1.0, c=1.0)
    st_s = GG1Station.md1(0.6, 1.0, c=1.0, zeta_mode=ZETA_SLOPE)
    assert st_s.zeta(2.0) != st_l.zeta(2.0)
    assert st_s.zeta(2.0) == pytest.approx(st_s.phi(2.0) * st_l.zeta(2.0), rel=1e-12)


def test_a_non_positive_phi_is_refused_and_the_message_names_the_station():
    # A station whose E[T] does not decrease in capacity has no slope to calibrate to.
    # allocate would reject the zeta anyway; this fails earlier with a message that says
    # which station and at what capacity, because the cause is a modelling error.
    class FlatStation(Station):
        def sojourn_time(self, S):
            self._check_stable(S * self.mu)
            return 1.0                      # constant: dT/dS == 0, so phi == 0

        def sim_node(self, S, job_class):
            raise NotImplementedError

        @property
        def alloc_cost(self):
            return 1.0

        @property
        def default_zeta(self):
            return 1.0

    st = FlatStation(gamma=0.5, mu=1.0, name="flat", zeta_mode=ZETA_SLOPE)
    with pytest.raises(ValueError, match="flat"):
        st.zeta_from(1.0, 2.0)
    # Level mode on the same station is untouched: the guard is slope-only.
    ok = FlatStation(gamma=0.5, mu=1.0, name="flat", zeta_mode=ZETA_LEVEL)
    assert ok.zeta_from(1.0, 2.0) == 1.5      # T*x = 1.0 * (2.0*1.0 - 0.5)


def test_a_tiny_phi_still_produces_a_usable_zeta():
    # Review Focus 4. A cov = 0 station has phi = 1 - rho exactly, so at extreme load
    # zeta_slope is ~1e-6 of zeta_level. ZETA_FLOOR is local to noise_floor and clamps
    # nothing in allocate, which needs only finite and > 0 -- so this must pass straight
    # through rather than being clamped or refused.
    from qopt.allocator import allocate

    st = GG1Station(0.6, 1.0, c=1.0, cov_a=0.0, cov_s=0.0, zeta_mode=ZETA_SLOPE)
    S = st.gamma / st.mu * (1.0 + 1e-6)
    z = st.zeta_from(st.sojourn_time(S), S)
    assert 0.0 < z < 1e-9
    assert math.isfinite(z)
    partner = GG1Station.mm1(0.6, 1.0, c=1.0)
    caps = allocate([st, partner], 10.0, [z, partner.zeta(2.0)])
    assert all(math.isfinite(c) and c > 0 for c in caps)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: the slope-mode tests FAIL — `zeta_from` still ignores the mode, so slope and level values are equal.

- [ ] **Step 3: Add the branch**

Replace `Station.zeta_from` with:

```python
    def zeta_from(self, T, S):
        """Invert the functional form for an externally supplied E[T].

        Pure station arithmetic, independent of where E[T] came from — the analytic
        sojourn time or a simulation run. The single point at which either calibration is
        applied, so the branch lives here and nowhere else.

        LEVEL (eq 22, the default) makes the surrogate zeta/x pass through (S, T). SLOPE
        matches its derivative instead, which is the only thing eq 21 reads -- see
        qopt/zeta.py. The level arm is the shipped expression, operation for operation, so
        the default path does not move by one ulp.

        Both arms are LINEAR IN T, which `Optimizer._noise_floor` depends on: it passes a
        CI half-width in the T position and needs the result to be the correspondingly
        scaled perturbation of zeta.

        No clamp on phi. It is legitimately tiny -- exactly `1 - rho` for a cov = 0
        station, measured down to 1e-6 -- and `allocate` requires only that zeta be
        finite and positive.
        """
        x = S * self.mu - self.gamma
        if self._zeta_mode == ZETA_SLOPE:
            phi = self.phi(S)
            if not (math.isfinite(phi) and phi > 0.0):
                raise ValueError(
                    f"station {self.name!r}: slope-calibrated zeta needs a finite, "
                    f"strictly positive phi, got {phi} at S={S}. E[T] must strictly "
                    f"decrease in capacity for there to be a slope to calibrate to."
                )
            return phi * T * x
        return T * x
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 37 passed.

Run: `python3 -m pytest -q`
Expected: 501 passed, 10 skipped. **Every pre-existing test must still pass unchanged** — that is the bit-for-bit claim.

- [ ] **Step 5: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 1: divide by phi instead of multiplying. Still positive, still linear in T,
# and still equal to level for M/M/1 -- only the non-unit-phi tests can see it.
s = s.replace("            return phi * T * x", "            return T * x / phi", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: the phi-times-level and x^2 tests fail
git checkout -- qopt/station.py

python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/station.py")
s = p.read_text()
# Mutation 2: apply phi in BOTH modes, breaking the bit-for-bit default.
s = s.replace("        if self._zeta_mode == ZETA_SLOPE:", "        if True:", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: many pre-existing failures
git checkout -- qopt/station.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: 501 passed, 10 skipped
```

- [ ] **Step 6: Commit**

```bash
git add qopt/station.py tests/test_zeta.py
git commit -m "feat: apply slope calibration in zeta_from

One branch at the single point where eq 22 is inverted. The level arm is the
shipped expression operation for operation, pinned with == rather than approx, so
the default path does not move by one ulp and no existing test changes.

Both arms stay linear in T, which is the only property Optimizer._noise_floor
relies on -- it passes a CI half-width in the T position -- so the noise floor
needs no change at all.

A non-positive phi is refused with a message naming the station and capacity,
since the cause is a modelling error rather than a budget one. phi is NOT
clamped: it is legitimately as small as 1e-6 for a cov = 0 station at high load,
and allocate needs only a finite positive zeta.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 7: Report ζ-as-used, φ, and the modes

**Files:**
- Modify: `qopt/optimizer.py` (`Result` ~:14–47, the final reporting block ~:302–326)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: `Station.zeta_mode`, `Station.phi`, `ZETA_SLOPE`.
- Produces: `Result.zeta_phi: list`, `Result.zeta_mode: list`, `Result.zeta_shape_flags: list` — all defaulted to empty, so the direct `Result(...)` constructions in `tests/test_optimizer_loop.py:97`, `tests/test_example_simulated.py:29` and `tests/test_example_qcsc.py` keep working untouched. Task 9 populates `zeta_shape_flags`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zeta.py`:

```python
from qopt.allocator import min_feasible_budget
from qopt.optimizer import Optimizer, Result


def _mixed_pair(mode_a, mode_b):
    return [
        GG1Station.md1(0.6, 1.5, c=2.0, name="md1", zeta_mode=mode_a),
        GG1Station.mm1(1.2, 3.0, c=0.5, name="mm1", zeta_mode=mode_b),
    ]


def test_result_reports_the_mode_of_each_station():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_LEVEL)
    res = Optimizer(stations, 4.0 * min_feasible_budget(stations)).run()
    assert res.zeta_mode == [ZETA_SLOPE, ZETA_LEVEL]


def test_phi_is_literally_one_for_a_level_station():
    # Not a computed phi: level calibration never consults the derivative, so the
    # default path must neither start paying for one nor acquire a new way to fail.
    # A level station whose phi would RAISE still reports 1.0.
    class BrokenDerivative(GG1Station):
        def dT_dS(self, S):
            raise AssertionError("a level-mode run must never evaluate the derivative")

    stations = [
        BrokenDerivative(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=0.0, name="level"),
        GG1Station.mm1(1.2, 3.0, c=0.5, name="mm1"),
    ]
    res = Optimizer(stations, 4.0 * min_feasible_budget(stations)).run()
    assert res.zeta_phi == [1.0, 1.0]


def test_phi_is_reported_for_a_slope_station_and_recovers_the_eq22_value():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_LEVEL)
    res = Optimizer(stations, 4.0 * min_feasible_budget(stations)).run()
    # An M/D/1 station's phi is strictly below 1.
    assert 0.0 < res.zeta_phi[0] < 1.0
    assert res.zeta_phi[1] == 1.0
    # zeta is the value that actually drove the allocation, so dividing out phi gives
    # back eq 22's level calibration.
    level = res.sojourn_times[0] * (res.capacities[0] * stations[0].mu - stations[0].gamma)
    assert res.zeta[0] / res.zeta_phi[0] == pytest.approx(level, rel=1e-12)
    assert res.zeta[0] == pytest.approx(res.zeta_phi[0] * level, rel=1e-12)


def test_the_new_result_fields_are_all_defaulted():
    # tests/test_optimizer_loop.py and the example tests construct Result directly.
    res = Result(
        capacities=[1.0], sojourn_times=[2.0], zeta=[1.0], objective=2.0,
        iterations=1, converged=True, residual=0.0,
    )
    assert res.zeta_phi == []
    assert res.zeta_mode == []
    assert res.zeta_shape_flags == []


def test_the_reported_zeta_is_the_one_that_drove_the_allocation():
    # Re-running allocate on the reported zeta must reproduce the reported capacities.
    from qopt.allocator import allocate

    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    res = Optimizer(stations, C).run()
    again = allocate(stations, C, res.zeta)
    for a, b in zip(again, res.capacities):
        assert a == pytest.approx(b, rel=1e-8)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: FAIL — `AttributeError: 'Result' object has no attribute 'zeta_mode'`.

- [ ] **Step 3: Add the fields**

In `qopt/optimizer.py`, after the `sim_calls` field in `Result`:

```python
    # ζ calibration diagnostics (qopt/zeta.py). Defaulted, so direct construction is
    # unchanged.
    zeta_phi: list = field(default_factory=list)
    """Per-station phi -- the factor by which `zeta` above exceeds eq 22's level value.

    Literally 1.0 for a level-mode station: that path never evaluates a derivative, so
    it neither pays for one nor gains a new way to fail. For a slope station this is the
    phi that produced the reported `zeta`, so `zeta[i]/zeta_phi[i]` recovers eq 22's
    value up to one rounding.
    """
    zeta_mode: list = field(default_factory=list)
    """Per-station calibration (a qopt.ZETA_* constant), so a mixed network stays legible."""
    zeta_shape_flags: list = field(default_factory=list)
    """Slope-mode stations whose measured E[T] disagrees with their analytic model.

    A MODEL-SPECIFICATION signal, kept out of `degraded` deliberately: `degraded` is the
    simulation-quality audit and `strict=True` raises on any entry, which would abort
    runs whose simulation was fine. See `Optimizer.zeta_shape_tol`.
    """
```

- [ ] **Step 4: Populate them**

In `Optimizer.run`, replace the final ζ block (currently `zeta = [st.zeta_from(T, Si) for ...]`) with:

```python
        sojourn_times = list(evaluation.sojourn_times)
        zeta = [
            st.zeta_from(T, Si) for st, T, Si in zip(stations, sojourn_times, S)
        ]
        # Recomputed HERE rather than captured in the loop: this runs after the last
        # retune, at the same S and the same station state as the zeta_from call above,
        # so it is the phi that produced the reported zeta bit-for-bit. A level station
        # reports the literal 1.0 -- no derivative is evaluated on the default path.
        zeta_phi = [
            st.phi(Si) if st.zeta_mode == ZETA_SLOPE else 1.0
            for st, Si in zip(stations, S)
        ]
        zeta_mode = [st.zeta_mode for st in stations]
```

and add to the `Result(...)` call, after `sim_calls=sim_calls,`:

```python
            zeta_phi=zeta_phi,
            zeta_mode=zeta_mode,
```

Add the import at the top of `qopt/optimizer.py`:

```python
from qopt.zeta import ZETA_SLOPE
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 42 passed.

Run: `python3 -m pytest -q`
Expected: 506 passed, 10 skipped.

- [ ] **Step 6: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/optimizer.py")
s = p.read_text()
# Mutation: compute phi for every station, including level ones.
s = s.replace("            st.phi(Si) if st.zeta_mode == ZETA_SLOPE else 1.0",
              "            st.phi(Si)", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: test_phi_is_literally_one_for_a_level_station fails
git checkout -- qopt/optimizer.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: 506 passed, 10 skipped
```

- [ ] **Step 7: Commit**

```bash
git add qopt/optimizer.py tests/test_zeta.py
git commit -m "feat: report zeta-as-used, phi, and the per-station calibration

Result.zeta continues to hold the zeta that actually drove the allocation, so a
Result explains its own capacities; zeta_phi makes eq 22's level value recoverable
as zeta[i]/zeta_phi[i]. All three new fields are defaulted, so the direct
Result(...) constructions in the optimizer and example tests are untouched.

A level station reports the literal 1.0 rather than a computed phi -- pinned by a
station whose dT_dS raises, which a level-mode run must never call.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 8: End-to-end against the reference optima

**Files:**
- Modify: `qopt/optimizer.py` (the comment at the retune site ~:199–241)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: no new API. This task is the payoff witness and the retune-ordering anchor.

**Reference data:** `docs/slope-calibrated-zeta/probe-output.txt` section 3. Budgets are `mult * sum(st.min_spend)`, the Optimizer runs at its analytic defaults (`tol=1e-9`, `damping=1.0`), fork-joins use `r_star="tuned"`, and **the weights are load-bearing** (Review Focus 1).

- [ ] **Step 1: Write the level-column test first — it validates the port**

Append to `tests/test_zeta.py`:

```python
# Station specs and objectives from docs/slope-calibrated-zeta/probe-output.txt section 3.
# The WEIGHTS are part of the data: eq 21 combines w and zeta under one square root, so a
# dropped weight is indistinguishable from a phi error in the objective alone.

def _net_fj_mm1(mode):
    return [
        ForkJoinStation(0.45, 1.0, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                        name="FJ-A", zeta_mode=mode),
        ForkJoinStation(0.80, 2.0, 2.0, r=2.0, c1=1.0, c2=3.0, r_star="tuned",
                        name="FJ-B", zeta_mode=mode),
        GG1Station(0.60, 1.5, 1.0, c=2.0, cov_a=1.0, cov_s=1.0, name="SS-1",
                   zeta_mode=mode),
        GG1Station(1.20, 3.0, 1.5, c=0.5, cov_a=1.0, cov_s=1.0, name="SS-2",
                   zeta_mode=mode),
    ]


def _net_mixed_cov(mode):
    return [
        ForkJoinStation(0.45, 1.0, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned",
                        name="FJ-A", zeta_mode=mode),
        GG1Station(0.60, 1.5, 1.0, c=2.0, cov_a=1.0, cov_s=0.0, name="M/D/1",
                   zeta_mode=mode),
        GG1Station(1.20, 3.0, 1.5, c=0.5, cov_a=2.0, cov_s=2.0, name="cov2",
                   zeta_mode=mode),
        GG1Station(0.90, 2.0, 1.0, c=1.0, cov_a=1.0, cov_s=1.0, name="M/M/1",
                   zeta_mode=mode),
    ]


_MULTS = (1.01, 1.05, 1.2, 1.5, 2, 5, 20)

_FJ_MM1_LEVEL = (845.653557502, 169.433928883, 42.584355698, 17.147817304,
                 8.624738064, 2.176046247, 0.459984473)
_FJ_MM1_SLOPE = (845.326602738, 169.374048331, 42.572541019, 17.144312750,
                 8.623541689, 2.175978022, 0.459983404)
_MIXED_LEVEL = (672.781044226, 132.897962166, 32.456697879, 12.821589655,
                6.396010744, 1.600929447, 0.336149757)
_MIXED_SLOPE = (672.409070110, 132.736738624, 32.343754976, 12.755896143,
                6.364185455, 1.598169104, 0.336090106)


def _objectives(build, mode):
    out = []
    for mult in _MULTS:
        # A FRESH network per row: a tuned fork-join is mutated by retune, and
        # min_spend must be read before any run has moved the ray.
        stations = build(mode)
        C = mult * min_feasible_budget(stations)
        out.append(Optimizer(stations, C).run().objective)
    return out


@pytest.mark.parametrize("build,expected", [
    (_net_fj_mm1, _FJ_MM1_LEVEL),
    (_net_mixed_cov, _MIXED_LEVEL),
])
def test_level_objectives_reproduce_the_reference_probe_run(build, expected):
    # Review Focus 1. Today's code path already produces these, so this test fails if
    # the station specs were ported wrongly -- a dropped weight, a swapped cov, the
    # wrong r_star -- BEFORE slope mode is exercised at all. Do not loosen the
    # tolerance to make it pass; fix the specs.
    got = _objectives(build, ZETA_LEVEL)
    for mult, g, e in zip(_MULTS, got, expected):
        assert g == pytest.approx(e, rel=1e-9), mult
```

- [ ] **Step 2: Run it to confirm the port is right**

Run: `python3 -m pytest tests/test_zeta.py -k level_objectives -q`
Expected: 2 passed. If it fails, the station specs do not match the probe's — compare against `docs/slope-calibrated-zeta/probe.py` `NET_FJ_MM1` / `NET_MIXED_COV`, and do **not** adjust the tolerance.

- [ ] **Step 3: Write the slope-column test**

Append:

```python
@pytest.mark.parametrize("build,expected", [
    (_net_fj_mm1, _FJ_MM1_SLOPE),
    (_net_mixed_cov, _MIXED_SLOPE),
])
def test_slope_objectives_reproduce_the_reference_probe_run(build, expected):
    got = _objectives(build, ZETA_SLOPE)
    for mult, g, e in zip(_MULTS, got, expected):
        assert g == pytest.approx(e, rel=1e-9), mult


@pytest.mark.parametrize("build,level,slope", [
    (_net_fj_mm1, _FJ_MM1_LEVEL, _FJ_MM1_SLOPE),
    (_net_mixed_cov, _MIXED_LEVEL, _MIXED_SLOPE),
])
def test_slope_never_loses_to_level_on_the_reference_networks(build, level, slope):
    # The objective is a weighted sum of sojourn times, so lower is better. Slope
    # calibration is exact, so it cannot be beaten by the level approximation.
    for mult, l, s in zip(_MULTS, level, slope):
        assert s <= l, mult


def test_the_mixed_network_gain_reaches_the_documented_half_percent():
    # findings.md section 4: up to 0.515%, at C/floor = 1.5 on the mixed-cov network.
    # The gain tracks |phi - 1|, so it is the network with an M/D/1 and a cov = 2
    # station that shows it, not the fork-join pair.
    i = _MULTS.index(1.5)
    gain = (_MIXED_LEVEL[i] - _MIXED_SLOPE[i]) / _MIXED_LEVEL[i]
    assert gain == pytest.approx(0.00515, rel=0.02)
    # And the fork-join-only network gains far less, for the same reason.
    j = _MULTS.index(1.5)
    fj_gain = (_FJ_MM1_LEVEL[j] - _FJ_MM1_SLOPE[j]) / _FJ_MM1_LEVEL[j]
    assert fj_gain < 0.0005
```

- [ ] **Step 4: Run to verify**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 49 passed.

- [ ] **Step 5: Add the retune-ordering anchor**

Append:

```python
def test_slope_calibration_needs_the_retune_to_run_last():
    # The radial derivative equals the true marginal only ON the optimal ray, so
    # zeta_from must see a station whose ray is already optimal for the spend it holds.
    # The Optimizer guarantees that by calling retune LAST in each iteration.
    #
    # This test pins the CONSEQUENCE rather than the ordering: at convergence every
    # tuned fork-join must be sitting on the ray that is locally optimal for its own
    # spend. Reordering the retune leaves the station one iteration stale, and on a
    # network that is still moving that shows up here.
    from qopt.forkjoin_policy import optimal_ray

    stations = _net_mixed_cov(ZETA_SLOPE)
    C = 1.5 * min_feasible_budget(stations)
    res = Optimizer(stations, C).run()
    fj = stations[0]
    spend = res.capacities[0] * fj.alloc_cost
    assert fj.r_star == pytest.approx(
        optimal_ray(fj.gamma, fj.mu_base, fj.r_base, fj.c1, fj.c2, spend), rel=1e-9
    )
```

- [ ] **Step 6: Verify the anchor actually bites, and record the honest answer**

The spec requires reporting whether this test really pins the ordering. Move the retune before the allocation and see:

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/optimizer.py")
s = p.read_text()
# Mutation: retune BEFORE eq 21 instead of after, so zeta_from sees a stale ray.
s = s.replace("            S_new = [st.retune(s) for st, s in zip(stations, S_new)]",
              "            pass  # retune moved", 1)
s = s.replace("            evaluation = self.analyzer.evaluate(stations, S)",
              "            S = [st.retune(s) for st, s in zip(stations, S)]\n"
              "            evaluation = self.analyzer.evaluate(stations, S)", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # RECORD the result
git checkout -- qopt/optimizer.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: back to green
```

Then add the comment at the retune site in `qopt/optimizer.py`, **matching what you just observed**. If the mutation failed the slope tests, append to the existing "Placed before the residual so that..." paragraph:

```python
            # That ordering is tidiness for the LEVEL calibration and correctness for the
            # slope one (qopt/zeta.py): slope-calibrated zeta prices a fork-join by the
            # derivative along its CURRENT ray, and that equals the true marginal only on
            # the optimal ray -- 3.3e-08 agreement on it, against 20.4% and 12.3%
            # disagreement at r* = 1.0 and 4.0. Retuning last is what leaves the next
            # iteration's zeta_from looking at an already-optimal ray. Pinned by
            # test_slope_calibration_needs_the_retune_to_run_last.
```

If the mutation did **not** fail any test, use this instead, and say so in the commit message — a comment that only looks load-bearing is worse than one that admits its limit:

```python
            # That ordering is tidiness for the LEVEL calibration and correctness for the
            # slope one (qopt/zeta.py): slope-calibrated zeta prices a fork-join by the
            # derivative along its CURRENT ray, and that equals the true marginal only on
            # the optimal ray -- 3.3e-08 agreement on it, against 20.4% and 12.3%
            # disagreement at r* = 1.0 and 4.0. Retuning last is what leaves the next
            # iteration's zeta_from looking at an already-optimal ray.
            #
            # DOCUMENTED BUT NOT TEST-PINNED: moving this retune before eq 21 does not
            # move the converged objective far enough for any assertion to catch, because
            # the fixed point closes the ray one iteration later either way. Treat the
            # reason above as the constraint, not the suite.
```

- [ ] **Step 7: Add the Review Focus 5 and 2 tests**

Append:

```python
def test_slope_mode_on_an_all_mm1_network_is_bit_for_bit_level_mode():
    # Review Focus 5, and the cheapest witness that phi is right: phi == 1 identically
    # on M/M/1, so every capacity must match EXACTLY, not approximately. A stray factor
    # anywhere in the slope arm breaks this.
    def net(mode):
        return [
            GG1Station.mm1(0.6, 1.5, 1.0, c=2.0, name="a", zeta_mode=mode),
            GG1Station.mm1(1.2, 3.0, 1.5, c=0.5, name="b", zeta_mode=mode),
            GG1Station.mm1(0.9, 2.0, 0.2, c=1.0, name="c", zeta_mode=mode),
        ]

    level = net(ZETA_LEVEL)
    slope = net(ZETA_SLOPE)
    C = 3.0 * min_feasible_budget(level)
    r_l = Optimizer(level, C).run()
    r_s = Optimizer(slope, C).run()
    assert r_s.capacities == r_l.capacities
    assert r_s.objective == r_l.objective
    assert r_s.iterations == r_l.iterations


def test_a_slope_run_converges_on_a_network_with_a_very_small_phi():
    # Review Focus 4, end to end. A cov = 0 station's phi falls toward 1 - rho, so its
    # converged zeta is orders of magnitude below its level value while its neighbours'
    # are not. default_zeta is unchanged at 1.0, so the run starts far from the answer
    # and must still converge rather than stall at max_iter.
    stations = [
        GG1Station(0.6, 1.0, 1.0, c=1.0, cov_a=0.0, cov_s=0.0, name="dd1",
                   zeta_mode=ZETA_SLOPE),
        GG1Station.mm1(1.2, 3.0, 1.0, c=0.5, name="mm1", zeta_mode=ZETA_SLOPE),
    ]
    res = Optimizer(stations, 1.05 * min_feasible_budget(stations)).run()
    assert res.converged, res.stop_reason
    assert all(z > 0.0 for z in res.zeta)
    assert 0.0 < res.zeta_phi[0] < 0.5
```

- [ ] **Step 8: Run everything**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 52 passed.

Run: `python3 -m pytest -q`
Expected: 516 passed, 10 skipped.

- [ ] **Step 9: Commit**

```bash
git add qopt/optimizer.py tests/test_zeta.py
git commit -m "test: slope calibration reaches the reference optima end to end

Reproduces both networks of docs/slope-calibrated-zeta/probe-output.txt section 3
through the real Optimizer. The LEVEL column is asserted first and separately:
today's code path already produces it, so a mis-ported station spec -- a dropped
weight, a swapped cov, the wrong r_star -- fails before slope mode is exercised
at all. The weights are load-bearing, since eq 21 combines w and zeta under one
square root.

Also pins the two invariants that catch a stray factor anywhere in the slope arm:
an all-M/M/1 network must come out bit-for-bit identical to level mode, since phi
is identically 1 there, and a network containing a cov = 0 station whose phi
falls to a few percent must still converge from the unchanged default_zeta.

Records at the retune site why that call must stay last: slope calibration prices
a fork-join along its current ray, which equals the true marginal only on the
optimal one.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 9: The measured-vs-analytic E[T] cross-check

**Files:**
- Modify: `qopt/optimizer.py` (`Optimizer.__init__`, the loop body after `evaluate`, the `Result(...)` call)
- Test: `tests/test_zeta.py`

**Interfaces:**
- Consumes: `ZETA_SHAPE_TOL`, `ZETA_SLOPE`, `Station.sojourn_time`, `Result.zeta_shape_flags` from Task 7.
- Produces: `Optimizer(..., zeta_shape_tol=ZETA_SHAPE_TOL)`; populated `Result.zeta_shape_flags`; a `RuntimeWarning` per offending station.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zeta.py`:

```python
from qopt.analyzer import Analyzer, Evaluation


class _ScaledAnalyzer(Analyzer):
    """Reports each station's analytic E[T] scaled by a factor: a stand-in for a
    simulator whose measurement disagrees with the station's model."""

    is_stochastic = False

    def __init__(self, factor):
        self.factor = factor

    def evaluate(self, stations, S, *, fresh_seed=False):
        return Evaluation(
            sojourn_times=[st.sojourn_time(Si) * self.factor
                           for st, Si in zip(stations, S)],
        )


def test_the_cross_check_is_silent_when_measurement_matches_the_model():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning becomes a failure
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(1.10)).run()
    assert res.zeta_shape_flags == []           # 10% is inside the 25% default


def test_the_cross_check_fires_above_the_tolerance_and_names_the_station():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_LEVEL)
    C = 4.0 * min_feasible_budget(stations)
    with pytest.warns(RuntimeWarning, match="md1"):
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0)).run()
    assert len(res.zeta_shape_flags) == 1       # only the SLOPE station is checked
    assert "md1" in res.zeta_shape_flags[0]
    assert "cov" in res.zeta_shape_flags[0]     # the message points at the likely cause


def test_the_cross_check_warns_once_per_station_not_once_per_iteration():
    # The condition it detects is a constructor argument: it cannot heal between
    # iterations, so repeating the warning for every one of them is noise.
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0)).run()
    shape = [w for w in caught if "analytic" in str(w.message)]
    assert len(shape) == 2                      # two slope stations, one warning each
    assert len(res.zeta_shape_flags) == 2
    # Guards against the test being vacuous: if the loop ran no more iterations than it
    # emitted warnings, it proves nothing about per-iteration repetition. If this fires,
    # lower `damping` to force more iterations -- do NOT delete the assertion.
    assert res.iterations > len(shape), res.iterations


def test_the_cross_check_is_vacuous_on_the_analytic_path():
    # AnalyticAnalyzer returns st.sojourn_time(Si), and the check recomputes exactly
    # that at the same S, so the ratio is 1.0 and this can never fire.
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 1.01 * min_feasible_budget(stations)    # a tight budget, where E[T] is largest
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Optimizer(stations, C).run()
    assert res.zeta_shape_flags == []


def test_a_shape_flag_does_not_make_a_strict_run_raise():
    # A cov_a mismatch is a MODEL-SPECIFICATION signal, not a simulation-quality one.
    # Routing it through `degraded` would make strict=True abort a run whose simulation
    # was fine, so it gets its own field.
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    with pytest.warns(RuntimeWarning):
        res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0), strict=True).run()
    assert res.zeta_shape_flags
    assert res.degraded == []


def test_the_cross_check_can_be_retuned_or_disabled():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    # A tighter tolerance fires on a disagreement the default tolerates.
    with pytest.warns(RuntimeWarning):
        Optimizer(stations, C, analyzer=_ScaledAnalyzer(1.10), zeta_shape_tol=0.05).run()
    # None disables it entirely.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = Optimizer(_mixed_pair(ZETA_SLOPE, ZETA_SLOPE), C,
                        analyzer=_ScaledAnalyzer(4.0), zeta_shape_tol=None).run()
    assert res.zeta_shape_flags == []


def test_a_level_only_run_never_evaluates_the_cross_check():
    # The default path must not start computing an analytic sojourn time it did not
    # need before.
    class BrokenAnalytic(GG1Station):
        calls = 0

        def sojourn_time(self, S):
            type(self).calls += 1
            return super().sojourn_time(S)

    stations = [BrokenAnalytic(0.6, 1.5, c=2.0, cov_a=1.0, cov_s=0.0, name="lvl")]
    C = 4.0 * min_feasible_budget(stations)
    res = Optimizer(stations, C, analyzer=_ScaledAnalyzer(4.0)).run()
    assert res.zeta_shape_flags == []
    # Only the analyzer's own calls, one per iteration plus the final evaluation.
    assert BrokenAnalytic.calls == res.iterations + 1


def test_an_invalid_shape_tolerance_is_rejected():
    stations = _mixed_pair(ZETA_SLOPE, ZETA_SLOPE)
    C = 4.0 * min_feasible_budget(stations)
    for bad in (-0.1, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="zeta_shape_tol"):
            Optimizer(stations, C, zeta_shape_tol=bad)
```

Add `import warnings` to the test file's imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'zeta_shape_tol'`.

- [ ] **Step 3: Add the constructor parameter**

In `Optimizer.__init__`, add `zeta_shape_tol=ZETA_SHAPE_TOL` to the keyword-only group, then in the body after `self.warm_start = warm_start`:

```python
        # None disables the measured-vs-analytic E[T] cross-check; any other value must
        # be a usable relative tolerance. `inf` is rejected rather than treated as
        # "disabled" so there is exactly one way to turn it off.
        if zeta_shape_tol is not None and not (
            math.isfinite(zeta_shape_tol) and zeta_shape_tol > 0.0
        ):
            raise ValueError(
                f"zeta_shape_tol must be None or a finite number > 0, got "
                f"{zeta_shape_tol}"
            )
        self.zeta_shape_tol = zeta_shape_tol
```

and extend the import:

```python
from qopt.zeta import ZETA_SHAPE_TOL, ZETA_SLOPE
```

- [ ] **Step 4: Add the check to the loop**

In `Optimizer.run`, initialise beside `degraded = []`:

```python
        zeta_shape_flags = []
        shape_checked = set()     # station ids already flagged; warn once each
```

and immediately after `degraded.extend(evaluation.degraded)` inside the loop:

```python
            # Under slope calibration phi comes from the station's ANALYTIC model while
            # E[T] is measured, so a model parameter describing something the station
            # does not actually see -- `cov_a` for an arrival process shaped by internal
            # traffic -- buys a converged, plausible, quietly suboptimal answer with no
            # symptom of its own. Comparing the two E[T] values tests exactly that
            # assumption, costs one analytic evaluation, and AMPLIFIES what it detects:
            # a 24% error in phi shows up as a 268% error in E[T].
            #
            # Here and not in `zeta_from`, because `_noise_floor` calls that hook with a
            # CI HALF-WIDTH in the T position -- a shape check inside it would compare a
            # half-width against a sojourn time and fire on every stochastic iteration.
            #
            # Warned once per station: the cause is a constructor argument and cannot
            # heal between iterations. Vacuous on the analytic path, where `evaluate`
            # returns this same `sojourn_time` at this same S.
            if self.zeta_shape_tol is not None:
                for st, T, Si in zip(stations, evaluation.sojourn_times, S):
                    if st.zeta_mode != ZETA_SLOPE or id(st) in shape_checked:
                        continue
                    T_model = st.sojourn_time(Si)
                    if abs(T / T_model - 1.0) > self.zeta_shape_tol:
                        shape_checked.add(id(st))
                        message = (
                            f"station {st.name!r}: measured E[T]={T:g} disagrees with "
                            f"its analytic model's {T_model:g} by "
                            f"{abs(T / T_model - 1.0) * 100:.1f}%, above "
                            f"zeta_shape_tol={self.zeta_shape_tol:g}. Slope-calibrated "
                            f"zeta takes the SHAPE of E[T] from that model, so check the "
                            f"station's parameters -- most often cov_a, which is never "
                            f"sent to the simulator and must describe the arrival process "
                            f"the station actually sees, internal traffic included. "
                            f"cov_a=1 is the safe assumption when it is unknown."
                        )
                        zeta_shape_flags.append(message)
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
```

Finally pass it through in the `Result(...)` call, after `zeta_mode=zeta_mode,`:

```python
            zeta_shape_flags=zeta_shape_flags,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_zeta.py -q`
Expected: 60 passed.

Run: `python3 -m pytest -q`
Expected: 524 passed, 10 skipped.

- [ ] **Step 6: Verify the tests can fail (mutation check)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/optimizer.py")
s = p.read_text()
# Mutation 1: check every station, not only slope ones.
s = s.replace("                    if st.zeta_mode != ZETA_SLOPE or id(st) in shape_checked:",
              "                    if id(st) in shape_checked:", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: the level-only and one-flag tests fail
git checkout -- qopt/optimizer.py

python3 - <<'PY'
import pathlib
p = pathlib.Path("qopt/optimizer.py")
s = p.read_text()
# Mutation 2: route the flag through `degraded`, which makes strict=True raise.
s = s.replace("                        zeta_shape_flags.append(message)",
              "                        zeta_shape_flags.append(message)\n"
              "                        degraded.append(message)", 1)
p.write_text(s)
PY
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest tests/test_zeta.py -q     # EXPECT: test_a_shape_flag_does_not_make_a_strict_run_raise fails
git checkout -- qopt/optimizer.py
find . -name '__pycache__' -type d -exec rm -rf {} +
python3 -m pytest -q                        # EXPECT: 524 passed, 10 skipped
```

- [ ] **Step 7: Commit**

```bash
git add qopt/optimizer.py tests/test_zeta.py
git commit -m "feat: cross-check measured against analytic E[T] under slope calibration

Slope calibration takes phi from a station's ANALYTIC model while E[T] stays
measured, so cov_a stops being decorative on the simulated path -- it is never
sent to qsim and never measured back, and a value describing an arrival process
the station does not see buys a converged, plausible, quietly suboptimal answer
with no symptom. This compares the two E[T] values, which tests that assumption
directly and amplifies it: a 24% error in phi is a 268% error in E[T].

In the loop and NOT in zeta_from, because _noise_floor calls that hook with a CI
half-width in the T position. Warned once per station, since the cause is a
constructor argument. Vacuous on the analytic path by construction.

Kept out of Result.degraded deliberately: that is the simulation-quality audit
and strict=True raises on any entry, which would abort runs whose simulation was
fine. zeta_shape_tol defaults to 0.25 and takes None to disable -- configurable
because the +/-1.1% calibration evidence contains no G/G/1 with cov != 1, the
station type slope calibration most benefits.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Task 10: Documentation ripple

**Files:**
- Modify: `README.md`, `docs/slope-calibrated-zeta/README.md`, `docs/slope-calibrated-zeta/findings.md` (prose only), `docs/paper-map.md`
- No test file. The selfchecks are the test.

**Interfaces:** none — documentation only.

- [ ] **Step 1: Add the README section**

In `README.md`, after the `## Model` section's discussion of ζ (the paragraph near line 55 describing eq 21/eq 22), add:

```markdown
### ζ calibration: level (default) or slope

`ζ` is the one free parameter that carries a station's queueing behaviour into eq 21.
Eq 22 fixes it by matching the *level* of the sojourn-time curve, `ζ = E[T]·(Sμ − γ)`, and
that is what `qopt` does by default.

Eq 21, though, water-fills on *marginal* returns: it reads the surrogate `T̂ = ζ/x` only
through its derivative and never evaluates it. Matching the slope instead,

    ζ = φ·E[T]·x,    φ = |dE[T]/dS|·x/(μ·E[T])

makes the loop's fixed point the coupled optimum exactly rather than an approximation of
it. Select it per station:

```python
from qopt import GG1Station, ZETA_SLOPE

st = GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0, zeta_mode=ZETA_SLOPE)
```

`φ ≡ 1` for M/M/1, so the two calibrations agree exactly there and an all-M/M/1 network is
bit-for-bit unaffected. The gain tracks `|φ − 1|`: 0.0002–0.039% where only fork-join
stations deviate, up to 0.515% once single-server stations are not M/M/1. `Result.zeta`
reports the ζ that actually drove the allocation, with `Result.zeta_phi` and
`Result.zeta_mode` alongside it, so eq 22's value is recoverable as `zeta[i]/zeta_phi[i]`.

This is a deliberate divergence from eq 22, not an amendment to it — see
`docs/slope-calibrated-zeta/` for the derivation and the measurements.

**One caveat worth reading before switching a simulated run.** φ is computed from the
station's *analytic* model even when `E[T]` is measured, which is what keeps the simulator
in control of the allocation's level. That promotes `cov_a` from nearly decorative to a
live input: it is never sent to the simulator and never measured back, so under slope
calibration it must describe the arrival process the station *actually* sees, internal
traffic included. **When it is unknown, `cov_a = 1` is the safe assumption** — it forfeits
the gain rather than overshooting past it, since understating `cov_a` drives φ toward 1 and
degrades gracefully to level calibration while overstating it can land worse than eq 22.
`Optimizer` cross-checks measured against analytic `E[T]` for slope stations and reports
disagreements in `Result.zeta_shape_flags` (tolerance: `zeta_shape_tol`, default 25%). If
you know the true arrival variability but cannot express it as a `cov_a`, override `phi(S)`
on a subclass.
```

- [ ] **Step 2: Update the analysis directory's status lines**

`docs/slope-calibrated-zeta/README.md` line 3 — replace `**Nothing here is implemented** — ` with:

```markdown
**Implemented** as a per-station opt-in: `zeta_mode=ZETA_SLOPE`, default still level. See
`docs/superpowers/specs/2026-09-22-slope-calibrated-zeta-design.md`. `qopt` on `main` ships the
```

`docs/slope-calibrated-zeta/findings.md` line 3 — replace `**Status: proposal. Not implemented.**` with:

```markdown
**Status: implemented** (2026-09-22) as a per-station opt-in; the default is still eq 22's level
calibration and the paper is unchanged.
```

and in §8, mark step 3:

```markdown
3. ~~If it proceeds: implement behind a per-station opt-in so the default path stays bit-for-bit
   identical, with the two closed forms and a test that pins `φ ≡ 1` for M/M/1.~~ **Done** —
   `qopt/zeta.py`, `Station.dT_dS`/`phi`, `tests/test_zeta.py`.
```

**Edit prose only.** Do not touch any fenced block in `findings.md`.

- [ ] **Step 3: Record the divergence in the paper map**

Find where eq 22 is discussed (`grep -n 'eq 22\|zeta' docs/paper-map.md`) and add there:

```markdown
- **eq 22 (ζ inversion).** qopt implements this as the DEFAULT (`ZETA_LEVEL`) and also
  offers a deliberate divergence, `ZETA_SLOPE`, which calibrates ζ to the slope of E[T]
  rather than its level. The paper is unchanged; see `docs/slope-calibrated-zeta/` and
  `docs/superpowers/specs/2026-09-22-slope-calibrated-zeta-design.md`.
```

- [ ] **Step 4: Run both selfchecks and the suite**

```bash
python3 docs/quotes-selfcheck.py                 # EXPECT: PASSED
PYTHONPATH=. python3 docs/audit-selfcheck.py     # EXPECT: PASSED
python3 -m pytest -q                             # EXPECT: 524 passed, 10 skipped
```

If `quotes-selfcheck.py` fails, a fenced block in `findings.md` was edited — revert that hunk.

- [ ] **Step 5: Verify every README claim against the code**

Each of these must be checkable, not just plausible. Confirm by running:

```bash
python3 - <<'PY'
from qopt import GG1Station, ZETA_SLOPE, Optimizer, min_feasible_budget
st = GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0, zeta_mode=ZETA_SLOPE)
print("constructs:", st.zeta_mode)
pair = [st, GG1Station.mm1(1.2, 3.0, c=0.5, name="m")]
r = Optimizer(pair, 4.0 * min_feasible_budget(pair)).run()
print("zeta_phi:", r.zeta_phi, "modes:", r.zeta_mode, "flags:", r.zeta_shape_flags)
print("recovers eq22:", r.zeta[0] / r.zeta_phi[0])
PY
```

Every attribute the README names must appear in that output. Fix the README, not the output.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/slope-calibrated-zeta/README.md docs/slope-calibrated-zeta/findings.md docs/paper-map.md
git commit -m "docs: document slope-calibrated zeta and retire the proposal status

The README gains the option, the recoverability of eq 22's value from
zeta/zeta_phi, and the cov_a caveat that matters most in practice: phi comes from
the analytic model even when E[T] is measured, so cov_a becomes a live allocation
input on the simulated path and cov_a = 1 is the safe assumption when it is
unknown -- understating it degrades to level calibration, overstating it can land
worse than eq 22.

docs/slope-calibrated-zeta/ no longer says nothing here is implemented, and its
sequencing step 3 is marked done. Prose only: every fenced quote in findings.md
must stay verbatim against probe-output.txt, which quotes-selfcheck.py enforces.

paper-map.md records eq 22 as implemented-by-default with a documented divergence
available, since the paper itself is unchanged.

Co-Authored-By: Claude <209825114+claude[bot]@users.noreply.github.com>"
```

---

## Out of Scope

Recorded so no task quietly picks them up:

- **Redoing the contraction argument** for the new map (`docs/convergence-tuned-r-star` analyses the current one). Measurements show no practical change: iterations 5–18 against 6–18, zero failures across 13 budgets from 1.0001× to 1e4× the floor.
- **The `findings.md` §6 blend knob** `ζ = [1 + f(φ−1)]·T·x`. A research instrument; it stays in the probe.
- **Any edit to `docs/analysis.pdf`** or to the paper.
- **Changing the default.** It stays `ZETA_LEVEL`.
- **Issue #14** (tuned stations stay mutated after a run). Slope calibration adds no new mutation.
- **Consuming a measured interarrival SCV from qsim.** File as an issue after Task 10: *can `qsim-service` report per-station interarrival SCV, and should slope ζ consume it?* `MEASURES` in `qopt/qsim/spec.py` is a closed list of three, and adding to it has known fork-join hazards.
