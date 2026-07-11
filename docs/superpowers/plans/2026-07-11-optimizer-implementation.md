# Queueing Network Capacity Allocation Optimizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python optimizer that allocates resource capacities across a network of single-server (G/G/1) and fork-join queues via the Section-5 fixed-point iteration, minimizing weighted expected sojourn times under a budget constraint.

**Architecture:** A `Station` class hierarchy owns each queue's math (`sojourn_time(S)` and derived `zeta(S)`); a stateless `allocate()` function implements the closed-form capacity allocation (eq 21); an `Optimizer` drives the fixed-point loop (initialize ζ → allocate → recompute ζ → test convergence). Stations are analyzed independently from fixed per-station arrival rates.

**Tech Stack:** Python 3.12 (3.10+ compatible), stdlib only for the core (`math`, `abc`, `dataclasses`); `pytest` for tests. No third-party runtime dependencies.

## Global Constraints

- Python **>= 3.10**. Target/dev interpreter is 3.12.
- Core library uses **stdlib only** — no `numpy` or other third-party runtime deps.
- Tests use **pytest**.
- The fork-join `t_ul` formula is **copied** from `~/Projects/fork-join`
  `forkjoin/analytical.py::mean_response_time` with an attribution comment — **no** runtime
  dependency on that repo.
- Package name: **`qopt`** (flat package at repo root).
- Numeric comparisons in tests use `math.isclose(..., rel_tol=1e-9)` or `pytest.approx`
  unless a looser tolerance is explicitly stated.
- Every station's `mu` is used directly in eqs 21/22. For a fork-join station, `mu` is the
  **slower** server's rate; the faster server's rate is `r·mu`.
- Guards (must hold): budget feasibility `C > Σ_j alloc_cost_j·γ_j/µ_j`; strictly positive
  initial ζ; stability `S·mu > γ` inside `sojourn_time`.

---

## File Structure

```
quantum-optimizer/                  (repo root == cwd)
  pyproject.toml                    # package + pytest config
  qopt/
    __init__.py                     # exports public API
    exceptions.py                   # InfeasibleBudgetError, InstabilityError
    forkjoin_approx.py              # t_ul(lam, mu1, mu2)  (lifted UL formula)
    station.py                      # Station ABC, SingleServerStation, GG1Station, ForkJoinStation
    allocator.py                    # min_feasible_budget(), allocate()  (eq 21)
    optimizer.py                    # Result dataclass, Optimizer (fixed-point loop)
  examples/
    mixed_network.py                # runnable sample: build network, run, print S*, E[T], objective
  tests/
    test_forkjoin_approx.py
    test_station.py
    test_allocator.py
    test_optimizer.py
    test_example.py
```

Interface summary (locked here; tasks must match exactly):

- `t_ul(lam: float, mu1: float, mu2: float) -> float`
- `Station(gamma, mu, weight=1.0, *, name=None)` — abstract; attrs `gamma`, `mu`, `weight`,
  `name`; abstract `sojourn_time(S) -> float`; abstract property `alloc_cost -> float`;
  abstract property `default_zeta -> float`; concrete `zeta(S) -> float`.
- `SingleServerStation(gamma, mu, weight=1.0, *, c, name=None)` — abstract; property
  `alloc_cost == c`; property `default_zeta == 1.0`.
- `GG1Station(gamma, mu, weight=1.0, *, c, cov_a, cov_s, name=None)` — concrete
  `sojourn_time`; classmethods `mm1(gamma, mu, weight=1.0, *, c, name=None)` and
  `md1(gamma, mu, weight=1.0, *, c, name=None)`.
- `ForkJoinStation(gamma, mu, weight=1.0, *, r, c1, c2, name=None)` — concrete
  `sojourn_time`; property `alloc_cost == c1 + c2`; property `default_zeta == 1.5`.
- `min_feasible_budget(stations) -> float`
- `allocate(stations, C: float, zeta_vec: list[float]) -> list[float]`
- `Optimizer(stations, budget, *, tol=1e-9, max_iter=1000, initial_zeta=None)`; `run() -> Result`
- `Result(capacities, sojourn_times, zeta, objective, iterations, converged)` (dataclass)

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `qopt/__init__.py`
- Create: `qopt/exceptions.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable `qopt` package; `qopt.exceptions.InfeasibleBudgetError`,
  `qopt.exceptions.InstabilityError` (both subclass `QOptError(Exception)`).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "qopt"
version = "0.1.0"
description = "Capacity allocation optimizer for a network of queues"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["."]
include = ["qopt*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `qopt/exceptions.py`**

```python
"""Exceptions raised by the qopt optimizer."""


class QOptError(Exception):
    """Base class for all qopt errors."""


class InfeasibleBudgetError(QOptError):
    """Raised when the budget cannot satisfy the network's stability constraints."""


class InstabilityError(QOptError):
    """Raised when a capacity leaves a station unstable (S*mu <= gamma)."""
```

- [ ] **Step 3: Create `qopt/__init__.py`** (public API surface; filled in as modules land)

```python
"""qopt: capacity allocation optimizer for a network of queues."""

from qopt.exceptions import InfeasibleBudgetError, InstabilityError, QOptError

__all__ = ["QOptError", "InfeasibleBudgetError", "InstabilityError"]
```

- [ ] **Step 4: Create `tests/test_smoke.py`**

```python
def test_package_imports():
    import qopt
    from qopt.exceptions import InfeasibleBudgetError, InstabilityError

    assert issubclass(InfeasibleBudgetError, qopt.QOptError)
    assert issubclass(InstabilityError, qopt.QOptError)
```

- [ ] **Step 5: Create venv and install (editable) + run smoke test**

Run:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -q -e ".[dev]"
.venv/bin/python -m pytest tests/test_smoke.py -v
```
Expected: `test_package_imports PASSED`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml qopt/ tests/test_smoke.py
git commit -m "feat: project scaffolding (package, exceptions, smoke test)"
```

---

## Task 2: Fork-join UL approximation

**Files:**
- Create: `qopt/forkjoin_approx.py`
- Test: `tests/test_forkjoin_approx.py`

**Interfaces:**
- Consumes: `qopt.exceptions.InstabilityError`.
- Produces: `t_ul(lam: float, mu1: float, mu2: float) -> float` — UL (upper–lower bound
  interpolation) mean response time for a heterogeneous 2-queue fork-join system.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forkjoin_approx.py
import math

import pytest

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul


def test_homogeneous_matches_nelson_tantawi():
    # For mu1 == mu2, T_UL reduces to (12 - rho) / (8 (mu - lam)).
    lam, mu = 0.5, 1.0
    rho = lam / mu
    expected = (12 - rho) / (8 * (mu - lam))  # = 2.875
    assert t_ul(lam, mu, mu) == pytest.approx(expected, rel=1e-12)


def test_heterogeneous_known_value():
    # Cross-checked against the fork-join repo's mean_response_time (doc table row
    # mu=1.0, mu=2.0, lam=0.6 -> ~2.641).
    assert t_ul(0.6, 2.0, 1.0) == pytest.approx(2.640873, rel=1e-5)


def test_symmetric_in_rates():
    assert t_ul(0.6, 2.0, 1.0) == pytest.approx(t_ul(0.6, 1.0, 2.0), rel=1e-12)


def test_unstable_raises():
    with pytest.raises(InstabilityError):
        t_ul(1.0, 1.0, 2.0)  # lam >= min(mu1, mu2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_forkjoin_approx.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.forkjoin_approx'`.

- [ ] **Step 3: Write the implementation**

```python
# qopt/forkjoin_approx.py
"""UL approximation for the heterogeneous 2-queue fork-join mean response time.

Lifted from the fork-join repo (forkjoin/analytical.py::mean_response_time):
a convex blend of the independent upper bound and the bottleneck lower bound,
  T_UL = (1 - alpha) * T_UB + alpha * T_bot,  alpha = (rho1 + rho2) / 8.
Exact for the homogeneous case (mu1 == mu2). Copied here to avoid a runtime
dependency on that repo.
"""

from qopt.exceptions import InstabilityError


def t_ul(lam, mu1, mu2):
    """Mean response time of a 2-queue fork-join system (UL interpolation).

    Args:
        lam: Poisson arrival rate to the fork-join station.
        mu1, mu2: effective service rates of the two servers.

    Requires stability: lam < min(mu1, mu2).
    """
    if lam >= mu1 or lam >= mu2:
        raise InstabilityError(
            f"fork-join unstable: need lam < min(mu1, mu2), "
            f"got lam={lam}, mu1={mu1}, mu2={mu2}"
        )
    rho1 = lam / mu1
    rho2 = lam / mu2
    alpha = (rho1 + rho2) / 8.0
    t_ub = 1.0 / (mu1 - lam) + 1.0 / (mu2 - lam) - 1.0 / (mu1 + mu2 - 2.0 * lam)
    t_bot = max(1.0 / (mu1 - lam), 1.0 / (mu2 - lam))
    return (1.0 - alpha) * t_ub + alpha * t_bot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_forkjoin_approx.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add qopt/forkjoin_approx.py tests/test_forkjoin_approx.py
git commit -m "feat: fork-join UL response-time approximation (t_ul)"
```

---

## Task 3: Station base + single-server (G/G/1)

**Files:**
- Create: `qopt/station.py`
- Modify: `qopt/__init__.py` (export `GG1Station`)
- Test: `tests/test_station.py`

**Interfaces:**
- Consumes: `qopt.exceptions.InstabilityError`.
- Produces:
  - `Station(gamma, mu, weight=1.0, *, name=None)` — abstract base. Attributes `gamma`,
    `mu`, `weight`, `name`. Abstract `sojourn_time(S) -> float`; abstract properties
    `alloc_cost -> float`, `default_zeta -> float`; concrete
    `zeta(S) = sojourn_time(S) * (S*mu - gamma)`.
  - `SingleServerStation(gamma, mu, weight=1.0, *, c, name=None)` — abstract; property
    `alloc_cost == c`; property `default_zeta == 1.0`.
  - `GG1Station(gamma, mu, weight=1.0, *, c, cov_a, cov_s, name=None)` — concrete
    `sojourn_time`; classmethods `mm1(...)`, `md1(...)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_station.py
import pytest

from qopt.exceptions import InstabilityError
from qopt.station import GG1Station, SingleServerStation, Station


def test_gg1_cannot_be_singleserver_instantiated():
    # SingleServerStation is abstract (no sojourn_time).
    with pytest.raises(TypeError):
        SingleServerStation(gamma=0.5, mu=1.0, c=1.0)  # type: ignore[abstract]


def test_mm1_sojourn_and_zeta():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    # mu_eff = S*mu = 1.0; E[T] = 1/(1 - 0.6) = 2.5; zeta = 2.5 * 0.4 = 1.0
    assert st.sojourn_time(1.0) == pytest.approx(2.5, rel=1e-12)
    assert st.zeta(1.0) == pytest.approx(1.0, rel=1e-12)
    assert st.alloc_cost == pytest.approx(2.0)
    assert st.default_zeta == pytest.approx(1.0)
    assert isinstance(st, Station)


def test_mm1_zeta_is_one_across_capacities():
    st = GG1Station.mm1(gamma=0.4, mu=1.0, c=1.0)
    for S in (0.6, 1.0, 2.5, 10.0):
        assert st.zeta(S) == pytest.approx(1.0, rel=1e-12)


def test_md1_sojourn_and_zeta():
    st = GG1Station.md1(gamma=0.6, mu=1.0, c=1.0)
    # rho = 0.6; E[T] = 1 * (1 + 0.5 * 0.6/0.4) = 1.75; zeta = 1.75 * 0.4 = 0.7 = 1 - rho/2
    assert st.sojourn_time(1.0) == pytest.approx(1.75, rel=1e-12)
    assert st.zeta(1.0) == pytest.approx(0.7, rel=1e-12)


def test_md1_zeta_is_load_dependent():
    st = GG1Station.md1(gamma=0.6, mu=1.0, c=1.0)
    for S in (1.0, 2.0, 4.0):
        rho = 0.6 / (S * 1.0)
        assert st.zeta(S) == pytest.approx(1 - rho / 2, rel=1e-12)


def test_sojourn_time_unstable_raises():
    st = GG1Station.mm1(gamma=1.0, mu=1.0, c=1.0)
    with pytest.raises(InstabilityError):
        st.sojourn_time(1.0)  # S*mu = 1.0 == gamma


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0.0, mu=1.0, c=1.0),
    dict(gamma=0.5, mu=0.0, c=1.0),
    dict(gamma=0.5, mu=1.0, c=0.0),
])
def test_construction_validation(kwargs):
    with pytest.raises(ValueError):
        GG1Station.mm1(**kwargs)


def test_negative_cov_rejected():
    with pytest.raises(ValueError):
        GG1Station(gamma=0.5, mu=1.0, c=1.0, cov_a=-1.0, cov_s=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_station.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.station'`.

- [ ] **Step 3: Write the implementation**

```python
# qopt/station.py
"""Station hierarchy: each station owns its queueing math."""

from abc import ABC, abstractmethod

from qopt.exceptions import InstabilityError


class Station(ABC):
    """A node in the queueing network.

    Fields (used directly by the allocator and eqs 21/22):
        gamma: fixed arrival rate.
        mu: base service rate (for a fork-join station, the slower server's rate).
        weight: sojourn-time weight (omega).
        name: optional label for reporting.
    """

    def __init__(self, gamma, mu, weight=1.0, *, name=None):
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if mu <= 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        self.gamma = gamma
        self.mu = mu
        self.weight = weight
        self.name = name

    @abstractmethod
    def sojourn_time(self, S):
        """Expected sojourn time E[T] under capacity S. Raises InstabilityError if S*mu <= gamma."""

    @property
    @abstractmethod
    def alloc_cost(self):
        """Cost coefficient used in the budget constraint and eq 21."""

    @property
    @abstractmethod
    def default_zeta(self):
        """Strictly-positive starting guess for zeta."""

    def zeta(self, S):
        """Invert the functional form (eq 22): zeta = E[T] * (S*mu - gamma)."""
        return self.sojourn_time(S) * (S * self.mu - self.gamma)

    def _check_stable(self, mu_eff):
        if mu_eff <= self.gamma:
            raise InstabilityError(
                f"station {self.name!r} unstable: S*mu={mu_eff} <= gamma={self.gamma}"
            )


class SingleServerStation(Station):
    """Abstract base for one-server queues. Concrete subclasses supply sojourn_time."""

    def __init__(self, gamma, mu, weight=1.0, *, c, name=None):
        super().__init__(gamma, mu, weight, name=name)
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = c

    @property
    def alloc_cost(self):
        return self.c

    @property
    def default_zeta(self):
        return 1.0


class GG1Station(SingleServerStation):
    """G/G/1 queue via the Kingman / Allen-Cunneen mean-value approximation.

        E[T] = (1/mu_eff) * [1 + ((cov_a^2 + cov_s^2)/2) * rho/(1-rho)]

    with mu_eff = S*mu and rho = gamma/mu_eff. Exact for any M/G/1 (cov_a == 1).
    """

    def __init__(self, gamma, mu, weight=1.0, *, c, cov_a, cov_s, name=None):
        super().__init__(gamma, mu, weight, c=c, name=name)
        if cov_a < 0:
            raise ValueError(f"cov_a must be >= 0, got {cov_a}")
        if cov_s < 0:
            raise ValueError(f"cov_s must be >= 0, got {cov_s}")
        self.cov_a = cov_a
        self.cov_s = cov_s

    def sojourn_time(self, S):
        mu_eff = S * self.mu
        self._check_stable(mu_eff)
        rho = self.gamma / mu_eff
        k = (self.cov_a ** 2 + self.cov_s ** 2) / 2.0
        return (1.0 / mu_eff) * (1.0 + k * rho / (1.0 - rho))

    @classmethod
    def mm1(cls, gamma, mu, weight=1.0, *, c, name=None):
        """M/M/1 preset (cov_a = cov_s = 1); zeta is identically 1."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=1.0, name=name)

    @classmethod
    def md1(cls, gamma, mu, weight=1.0, *, c, name=None):
        """M/D/1 preset (cov_a = 1, cov_s = 0); zeta = 1 - rho/2."""
        return cls(gamma, mu, weight, c=c, cov_a=1.0, cov_s=0.0, name=name)
```

- [ ] **Step 4: Export from `qopt/__init__.py`**

```python
"""qopt: capacity allocation optimizer for a network of queues."""

from qopt.exceptions import InfeasibleBudgetError, InstabilityError, QOptError
from qopt.station import GG1Station, SingleServerStation, Station

__all__ = [
    "QOptError",
    "InfeasibleBudgetError",
    "InstabilityError",
    "Station",
    "SingleServerStation",
    "GG1Station",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_station.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add qopt/station.py qopt/__init__.py tests/test_station.py
git commit -m "feat: Station base + G/G/1 single-server station with mm1/md1 presets"
```

---

## Task 4: Fork-join station

**Files:**
- Modify: `qopt/station.py` (add `ForkJoinStation`)
- Modify: `qopt/__init__.py` (export `ForkJoinStation`)
- Test: `tests/test_station_forkjoin.py`

**Interfaces:**
- Consumes: `Station` (Task 3), `t_ul` (Task 2).
- Produces: `ForkJoinStation(gamma, mu, weight=1.0, *, r, c1, c2, name=None)` — concrete
  `sojourn_time`; property `alloc_cost == c1 + c2`; property `default_zeta == 1.5`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_station_forkjoin.py
import pytest

from qopt.exceptions import InstabilityError
from qopt.forkjoin_approx import t_ul
from qopt.station import ForkJoinStation, Station


def test_alloc_cost_is_sum_of_both_servers():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=3.0)
    assert st.alloc_cost == pytest.approx(4.0)
    assert st.default_zeta == pytest.approx(1.5)
    assert isinstance(st, Station)


def test_sojourn_uses_t_ul_with_both_effective_rates():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    # m1 = S*mu = 1.0 (slower), m2 = S*r*mu = 2.0 (faster)
    assert st.sojourn_time(1.0) == pytest.approx(t_ul(0.6, 1.0, 2.0), rel=1e-12)


def test_homogeneous_forkjoin_matches_nelson_tantawi():
    st = ForkJoinStation(gamma=0.5, mu=1.0, r=1.0, c1=1.0, c2=1.0)
    rho = 0.5
    expected = (12 - rho) / (8 * (1.0 - 0.5))  # 2.875
    assert st.sojourn_time(1.0) == pytest.approx(expected, rel=1e-12)


def test_zeta_uses_slower_server_rate():
    st = ForkJoinStation(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    expected = st.sojourn_time(1.0) * (1.0 * 1.0 - 0.6)
    assert st.zeta(1.0) == pytest.approx(expected, rel=1e-12)


def test_unstable_raises_on_slower_server():
    st = ForkJoinStation(gamma=1.0, mu=1.0, r=2.0, c1=1.0, c2=1.0)
    with pytest.raises(InstabilityError):
        st.sojourn_time(1.0)  # S*mu = 1.0 == gamma (slower server binds)


@pytest.mark.parametrize("kwargs", [
    dict(gamma=0.6, mu=1.0, r=0.9, c1=1.0, c2=1.0),   # r < 1
    dict(gamma=0.6, mu=1.0, r=2.0, c1=0.0, c2=1.0),   # c1 <= 0
    dict(gamma=0.6, mu=1.0, r=2.0, c1=1.0, c2=-1.0),  # c2 <= 0
])
def test_forkjoin_validation(kwargs):
    with pytest.raises(ValueError):
        ForkJoinStation(**kwargs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_station_forkjoin.py -v`
Expected: FAIL — `ImportError: cannot import name 'ForkJoinStation'`.

- [ ] **Step 3: Add `ForkJoinStation` to `qopt/station.py`**

Add this import near the top of `qopt/station.py` (below the existing imports):

```python
from qopt.forkjoin_approx import t_ul
```

Append this class to the end of `qopt/station.py`:

```python
class ForkJoinStation(Station):
    """Fork-join station: two parallel servers sharing one capacity S.

    Both servers receive capacity S, so effective rates are m1 = S*mu (slower) and
    m2 = S*(r*mu) (faster), preserving the ratio r for all S. mu is the slower server's
    rate; the faster server's rate is r*mu (r >= 1). Cost coefficient is c1 + c2.
    """

    def __init__(self, gamma, mu, weight=1.0, *, r, c1, c2, name=None):
        super().__init__(gamma, mu, weight, name=name)
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        if c1 <= 0:
            raise ValueError(f"c1 must be > 0, got {c1}")
        if c2 <= 0:
            raise ValueError(f"c2 must be > 0, got {c2}")
        self.r = r
        self.c1 = c1
        self.c2 = c2

    @property
    def alloc_cost(self):
        return self.c1 + self.c2

    @property
    def default_zeta(self):
        return 1.5

    def sojourn_time(self, S):
        m1 = S * self.mu          # slower server (binds stability)
        m2 = S * self.r * self.mu  # faster server
        self._check_stable(m1)
        return t_ul(self.gamma, m1, m2)
```

- [ ] **Step 4: Export from `qopt/__init__.py`**

Update the import and `__all__` in `qopt/__init__.py`:

```python
from qopt.station import (
    ForkJoinStation,
    GG1Station,
    SingleServerStation,
    Station,
)
```

and add `"ForkJoinStation"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_station_forkjoin.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add qopt/station.py qopt/__init__.py tests/test_station_forkjoin.py
git commit -m "feat: fork-join station (shared capacity, ratio r preserved)"
```

---

## Task 5: Allocator (eq 21)

**Files:**
- Create: `qopt/allocator.py`
- Modify: `qopt/__init__.py` (export `allocate`, `min_feasible_budget`)
- Test: `tests/test_allocator.py`

**Interfaces:**
- Consumes: any `Station` (uses `.gamma`, `.mu`, `.weight`, `.alloc_cost`).
- Produces:
  - `min_feasible_budget(stations) -> float` — `Σ_j alloc_cost_j * gamma_j / mu_j`.
  - `allocate(stations, C, zeta_vec) -> list[float]` — eq 21. Assumes `C` feasible and every
    `zeta > 0` (the Optimizer enforces both); returns capacities aligned to `stations`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_allocator.py
import pytest

from qopt.allocator import allocate, min_feasible_budget
from qopt.station import ForkJoinStation, GG1Station


def test_single_station_spends_whole_budget():
    st = GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0)
    (S,) = allocate([st], C=4.0, zeta_vec=[1.0])
    assert S == pytest.approx(4.0 / 2.0, rel=1e-12)  # S = C / c = 2.0


def test_budget_fully_spent_identity():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),
        GG1Station.md1(gamma=0.3, mu=2.0, c=1.0),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),
    ]
    C = 20.0
    zeta_vec = [1.0, 0.9, 1.5]
    S = allocate(stations, C, zeta_vec)
    spent = sum(st.alloc_cost * Si for st, Si in zip(stations, S))
    assert spent == pytest.approx(C, rel=1e-12)


def test_min_feasible_budget():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),   # 2 * 0.6/1.0 = 1.2
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),  # 2 * 0.5/1.0 = 1.0
    ]
    assert min_feasible_budget(stations) == pytest.approx(1.2 + 1.0, rel=1e-12)


def test_all_stations_stable_under_feasible_budget():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),
        GG1Station.md1(gamma=0.3, mu=2.0, c=1.0),
    ]
    C = 3 * min_feasible_budget(stations)
    S = allocate(stations, C, [st.default_zeta for st in stations])
    for st, Si in zip(stations, S):
        assert Si * st.mu > st.gamma
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_allocator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.allocator'`.

- [ ] **Step 3: Write the implementation**

```python
# qopt/allocator.py
"""Closed-form capacity allocation (paper eq 21)."""

import math


def min_feasible_budget(stations):
    """Minimum budget to keep every station stable: sum_j alloc_cost_j * gamma_j / mu_j.

    A budget strictly greater than this makes eq 21's slack term positive, so every
    allocated capacity satisfies S_i * mu_i > gamma_i.
    """
    return sum(st.alloc_cost * st.gamma / st.mu for st in stations)


def allocate(stations, C, zeta_vec):
    """Optimal capacities for fixed zeta (paper eq 21).

        S_i = gamma_i/mu_i
            + (C - sum_j c_j gamma_j/mu_j) * sqrt(w_i zeta_i/(c_i mu_i)) / sum_j sqrt(w_j zeta_j c_j/mu_j)

    where c_i = station.alloc_cost, w_i = station.weight. Assumes C is feasible and every
    zeta_i > 0 (enforced by the Optimizer). Returns a list aligned to `stations`.
    """
    base = [st.gamma / st.mu for st in stations]
    slack = C - sum(st.alloc_cost * b for st, b in zip(stations, base))
    denom = sum(
        math.sqrt(st.weight * z * st.alloc_cost / st.mu)
        for st, z in zip(stations, zeta_vec)
    )
    capacities = []
    for st, b, z in zip(stations, base, zeta_vec):
        num = math.sqrt(st.weight * z / (st.alloc_cost * st.mu))
        capacities.append(b + slack * num / denom)
    return capacities
```

- [ ] **Step 4: Export from `qopt/__init__.py`**

Add to `qopt/__init__.py`:

```python
from qopt.allocator import allocate, min_feasible_budget
```

and add `"allocate"` and `"min_feasible_budget"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_allocator.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add qopt/allocator.py qopt/__init__.py tests/test_allocator.py
git commit -m "feat: closed-form capacity allocator (eq 21) + feasibility helper"
```

---

## Task 6: Optimizer (fixed-point loop)

**Files:**
- Create: `qopt/optimizer.py`
- Modify: `qopt/__init__.py` (export `Optimizer`, `Result`)
- Test: `tests/test_optimizer.py`

**Interfaces:**
- Consumes: `allocate`, `min_feasible_budget` (Task 5); any `Station`.
- Produces:
  - `Result` dataclass: `capacities: list[float]`, `sojourn_times: list[float]`,
    `zeta: list[float]`, `objective: float`, `iterations: int`, `converged: bool`.
  - `Optimizer(stations, budget, *, tol=1e-9, max_iter=1000, initial_zeta=None)` with
    `run() -> Result`. Raises `InfeasibleBudgetError` if `budget <= min_feasible_budget`;
    raises `ValueError` if any initial zeta <= 0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_optimizer.py
import pytest

from qopt.allocator import min_feasible_budget
from qopt.exceptions import InfeasibleBudgetError
from qopt.optimizer import Optimizer, Result
from qopt.station import ForkJoinStation, GG1Station


def _mm1_network():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="a"),
        GG1Station.mm1(gamma=0.3, mu=2.0, c=1.0, name="b"),
    ]


def test_mm1_network_converges_with_unit_zeta():
    stations = _mm1_network()
    opt = Optimizer(stations, budget=5 * min_feasible_budget(stations))
    res = opt.run()
    assert res.converged
    assert all(z == pytest.approx(1.0, rel=1e-9) for z in res.zeta)
    # budget fully spent
    spent = sum(st.alloc_cost * S for st, S in zip(stations, res.capacities))
    assert spent == pytest.approx(opt.budget, rel=1e-9)


def test_objective_matches_weighted_sojourn_sum():
    stations = _mm1_network()
    opt = Optimizer(stations, budget=5 * min_feasible_budget(stations))
    res = opt.run()
    expected = sum(
        st.weight * st.sojourn_time(S) for st, S in zip(stations, res.capacities)
    )
    assert res.objective == pytest.approx(expected, rel=1e-12)
    assert res.sojourn_times == pytest.approx(
        [st.sojourn_time(S) for st, S in zip(stations, res.capacities)], rel=1e-12
    )


def test_mixed_network_with_md1_and_forkjoin_converges():
    stations = [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0),      # load-dependent zeta
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0),
    ]
    opt = Optimizer(stations, budget=6 * min_feasible_budget(stations))
    res = opt.run()
    assert res.converged
    for st, S in zip(stations, res.capacities):
        assert S * st.mu > st.gamma  # stable


def test_infeasible_budget_raises():
    stations = _mm1_network()
    with pytest.raises(InfeasibleBudgetError):
        Optimizer(stations, budget=min_feasible_budget(stations)).run()


def test_nonpositive_initial_zeta_raises():
    stations = _mm1_network()
    with pytest.raises(ValueError):
        Optimizer(
            stations,
            budget=5 * min_feasible_budget(stations),
            initial_zeta=[1.0, 0.0],
        ).run()


def test_max_iter_guard_returns_not_converged():
    # tol = 0 is never satisfied by "< tol", so the loop runs to max_iter and reports False.
    stations = _mm1_network()
    opt = Optimizer(
        stations, budget=5 * min_feasible_budget(stations), tol=0.0, max_iter=3
    )
    res = opt.run()
    assert res.converged is False
    assert res.iterations == 3
    assert isinstance(res, Result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qopt.optimizer'`.

- [ ] **Step 3: Write the implementation**

```python
# qopt/optimizer.py
"""Fixed-point optimization loop (paper Steps 0-5)."""

from dataclasses import dataclass

from qopt.allocator import allocate, min_feasible_budget
from qopt.exceptions import InfeasibleBudgetError


@dataclass
class Result:
    """Outcome of an optimization run (lists aligned to the station order)."""

    capacities: list
    sojourn_times: list
    zeta: list
    objective: float
    iterations: int
    converged: bool


class Optimizer:
    """Drives the fixed-point iteration for the capacity allocation problem.

    Loop: allocate from an initial zeta guess, then repeatedly recompute zeta from the
    current capacities (eq 22) and re-allocate (eq 21) until ||S_new - S||_inf < tol or
    max_iter is reached.
    """

    def __init__(self, stations, budget, *, tol=1e-9, max_iter=1000, initial_zeta=None):
        self.stations = list(stations)
        self.budget = budget
        self.tol = tol
        self.max_iter = max_iter
        self.initial_zeta = initial_zeta

    def run(self):
        stations = self.stations

        # Guard: budget must exceed the minimum needed for stability (eq 21 slack > 0).
        if self.budget <= min_feasible_budget(stations):
            raise InfeasibleBudgetError(
                f"budget {self.budget} <= minimum feasible "
                f"{min_feasible_budget(stations)}"
            )

        # Guard: strictly-positive initial zeta.
        if self.initial_zeta is None:
            zeta = [st.default_zeta for st in stations]
        else:
            zeta = list(self.initial_zeta)
            if len(zeta) != len(stations):
                raise ValueError("initial_zeta length must match number of stations")
        if any(z <= 0 for z in zeta):
            raise ValueError("initial zeta values must be strictly positive")

        S = allocate(stations, self.budget, zeta)  # S^(1)
        converged = False
        iterations = 0
        for _ in range(self.max_iter):
            iterations += 1
            zeta = [st.zeta(Si) for st, Si in zip(stations, S)]  # eq 22
            S_new = allocate(stations, self.budget, zeta)        # eq 21
            if max(abs(a - b) for a, b in zip(S_new, S)) < self.tol:
                S = S_new
                converged = True
                break
            S = S_new

        sojourn_times = [st.sojourn_time(Si) for st, Si in zip(stations, S)]
        objective = sum(
            st.weight * t for st, t in zip(stations, sojourn_times)
        )
        return Result(
            capacities=S,
            sojourn_times=sojourn_times,
            zeta=zeta,
            objective=objective,
            iterations=iterations,
            converged=converged,
        )
```

- [ ] **Step 4: Export from `qopt/__init__.py`**

Add to `qopt/__init__.py`:

```python
from qopt.optimizer import Optimizer, Result
```

and add `"Optimizer"` and `"Result"` to `__all__`.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests across all files pass.

- [ ] **Step 6: Commit**

```bash
git add qopt/optimizer.py qopt/__init__.py tests/test_optimizer.py
git commit -m "feat: fixed-point Optimizer loop + Result"
```

---

## Task 7: Runnable example

**Files:**
- Create: `examples/mixed_network.py`
- Test: `tests/test_example.py`

**Interfaces:**
- Consumes: public API from `qopt`.
- Produces: `build_network() -> list[Station]` and `main() -> Result`, so the example is
  both runnable (`python examples/mixed_network.py`) and testable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_example.py
from examples.mixed_network import build_network, main


def test_example_runs_and_converges():
    res = main()
    assert res.converged
    stations = build_network()
    assert len(res.capacities) == len(stations)
    for st, S in zip(stations, res.capacities):
        assert S * st.mu > st.gamma
    assert res.objective > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_example.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'examples.mixed_network'`.

- [ ] **Step 3: Create `examples/__init__.py` (empty) and `examples/mixed_network.py`**

```python
# examples/__init__.py
```

```python
# examples/mixed_network.py
"""Sample mixed network: two single-server queues and one fork-join station."""

from qopt import (
    ForkJoinStation,
    GG1Station,
    Optimizer,
    min_feasible_budget,
)


def build_network():
    return [
        GG1Station.mm1(gamma=0.6, mu=1.0, c=2.0, name="ingest (M/M/1)"),
        GG1Station.md1(gamma=0.4, mu=1.0, c=1.0, name="transform (M/D/1)"),
        ForkJoinStation(gamma=0.5, mu=1.0, r=2.0, c1=1.0, c2=1.0, name="fork-join"),
    ]


def main():
    stations = build_network()
    budget = 6 * min_feasible_budget(stations)
    result = Optimizer(stations, budget=budget).run()

    print(f"budget = {budget:.4f}   converged = {result.converged} "
          f"in {result.iterations} iterations")
    print(f"{'station':22s} {'S*':>10s} {'E[T]':>10s} {'zeta':>10s}")
    for st, S, t, z in zip(
        stations, result.capacities, result.sojourn_times, result.zeta
    ):
        print(f"{st.name:22s} {S:10.4f} {t:10.4f} {z:10.4f}")
    print(f"objective (sum w*E[T]) = {result.objective:.6f}")
    return result


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and the script**

Run: `.venv/bin/python -m pytest tests/test_example.py -v`
Expected: PASS.

Run: `.venv/bin/python examples/mixed_network.py`
Expected: a table of `S*`, `E[T]`, `zeta` per station and an objective value; `converged = True`.

- [ ] **Step 5: Run the full suite once more**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add examples/ tests/test_example.py
git commit -m "feat: runnable mixed-network example + test"
```

---

## Self-Review

**1. Spec coverage:**
- §2 formulation / eq 21 → Task 5 (`allocate`); eq 22 inversion → Task 3 (`Station.zeta`);
  Steps 0–5 loop → Task 6 (`Optimizer.run`). ✓
- §3.1 abstract `SingleServerStation` + `GG1Station(cov_a, cov_s)` + `mm1`/`md1` presets,
  ζ formula → Task 3. ✓
- §3.2 fork-join shared capacity, `alloc_cost = c1+c2`, slower-server binding, `t_ul` →
  Tasks 2 & 4. ✓
- §3.3 shared interface (`sojourn_time`, `alloc_cost`, `zeta`, `mu` used directly) → Task 3. ✓
- §4 components (Station/Allocator/Optimizer, Result fields, default ζ) → Tasks 3–6. ✓
- §5 guards: budget feasibility → Task 6 test + impl; positive initial ζ → Task 6; stability
  guard → Tasks 3/4; construction validation → Tasks 3/4; non-convergence → Task 6. ✓
- §7 acceptance criteria 1–9 → mapped across Tasks 2–6 tests (M/M/1 ζ=1, M/D/1 ζ=1−ρ/2,
  full-budget identity, homogeneous FJ = Nelson–Tantawi + t_ul cross-check, mixed-network
  convergence + objective, infeasible budget, non-positive ζ, max_iter guard, instability
  raise). ✓
- §6 layout → File Structure + tasks. ✓

**2. Placeholder scan:** No TBD/TODO; every code and test step contains complete code; every
command has an expected result. ✓

**3. Type consistency:** `alloc_cost`, `default_zeta`, `zeta(S)`, `sojourn_time(S)`,
`allocate(stations, C, zeta_vec)`, `min_feasible_budget(stations)`,
`Optimizer(..., tol, max_iter, initial_zeta)`, and `Result(capacities, sojourn_times, zeta,
objective, iterations, converged)` are used identically across the tasks that define and
consume them. ✓
