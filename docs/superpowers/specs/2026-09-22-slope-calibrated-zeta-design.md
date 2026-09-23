# Design Spec: Slope-calibrated ζ as a per-station option

Date: 2026-09-22
Status: **Design approved, not yet implemented.**
Analysis this rests on: `docs/slope-calibrated-zeta/findings.md` (the claim, the derivation, the
measured payoff) and `docs/slope-calibrated-zeta/probe.py` / `probe-output.txt` (every number it
cites). This spec does not restate that analysis; it specifies the code.

Companion specs:

- `docs/superpowers/specs/2026-07-10-optimizer-design.md` — the analytic optimizer. Its eq 21 is
  unchanged here; its eq 22 gains an alternative.
- `docs/superpowers/specs/2026-07-29-simulation-support-design.md` — §5.1 of that spec specifies
  `zeta_from(T, S)`, the one seam this design changes.

Equation numbers follow `docs/paper-map.md`: `eq 21` (allocation) and `eq 22` (the ζ inversion)
mean `docs/analysis.pdf`.

---

## 1. Goal

Let a user select, **per station**, whether ζ is calibrated to the *level* of that station's
sojourn-time curve (eq 22, `ζ = T·x`, today's behaviour and the default) or to its *slope*
(`ζ = φ·T·x = x²·|dT/dx|`).

The motivation, established in `findings.md` §3: eq 21 consumes the ζ surrogate **only through
its derivative**, so calibrating ζ to the curve's level spends the single free parameter on the
one quantity the allocator never reads. Calibrating the slope instead turns the loop's fixed
point from an approximation of the coupled optimum into the coupled optimum exactly — measured to
machine precision at all 27 rows tested, with iteration counts essentially unchanged
(`findings.md` §1's claim, measured in §4).

`φ ≡ 1` for M/M/1, which is why this was invisible: it is the station type most of the suite uses.
The gain tracks `|φ − 1|`, not station type — `0.0002–0.039%` where only fork-join stations
deviate, up to `0.515%` once single-server stations are not M/M/1, and `1.59%` on the §7 stress
network.

### 1.1 What does not change

- **The default.** `ZETA_LEVEL`. Every existing construction keeps today's behaviour.
- **eq 21 / `allocate` / `min_feasible_budget` / `noise_floor`.** Not touched. ζ is still a
  positive scalar per station; only its calibration changes.
- **The level arm is bit-for-bit today's arithmetic** — same operations in the same order (§5.1).
- **`sim_calls`.** φ is computed from each station's own analytic `sojourn_time`; it issues no
  simulation requests (§4.4).
- **`Optimizer._noise_floor`.** Slope ζ stays linear in `T`, which is the only property that
  method relies on (§5.2).
- **The paper.** qopt documents a deliberate divergence from eq 22; `docs/analysis.pdf` is not
  edited. The paper lives in another repo and is not publishable as it stands.

### 1.2 Stated assumptions

1. `φ` is derived from each station's **analytic** model even when `E[T]` is measured. This is
   deliberate — `findings.md` §6 — and it is the source of the one genuinely new coupling this
   change introduces (§8).
2. `φ > 0` for every station type shipped today. Measured range over a wide grid:
   `[0.000001, 1.817431]` (`findings.md` §7). Enforced rather than assumed (§4.4).
3. The fork-join derivative is the **radial** one, along the station's current ray. It equals the
   true marginal only *on* the optimal ray, which the optimizer's existing `retune` placement
   guarantees (§7).

---

## 2. Architecture

The selection is **per station**, stored on the station, dispatched inside `Station.zeta_from`.

Rejected alternatives, and why:

- **Per-run on `Optimizer`.** Would require the Optimizer to reach into and mutate stations to
  impose a calibration. Station mutation is already the open defect in issue #14; this adds no
  second mutating path.
- **Calibration strategy objects.** A public `ZetaCalibration` hierarchy for a two-member choice
  is machinery this codebase does not otherwise use. The extensibility it buys — `findings.md`
  §6's blend `ζ = [1 + f(φ−1)]·T·x` — is a research instrument that belongs in the probe.

Per-station also matches where the payoff is: `findings.md` §5 shows the fork-join barely moves
(`φ` within 1.3% of 1) while `M/D/1` and high-`cov` `G/G/1` reach `φ = 0.83` and `1.64`. A user
can calibrate the single-server stations by slope and leave fork-joins on level.

Precedent: `r_star` is selected exactly this way — a string constant validated by
`resolve_r_star`, stored on the station, behaviour branching on it.

---

## 3. Public API

### 3.1 New module `qopt/zeta.py`

```python
ZETA_LEVEL = "level"
"""eq 22: zeta = T*x. The default, and qopt's incumbent."""

ZETA_SLOPE = "slope"
"""zeta = phi*T*x = x^2*|dT/dx|. A deliberate divergence from eq 22 (see docs/slope-calibrated-zeta/)."""

ZETA_SHAPE_TOL = 0.25
"""Default relative tolerance for the measured-vs-analytic E[T] cross-check (section 8.4)."""

def resolve_zeta_mode(mode):
    """None -> ZETA_LEVEL; validate against the known set, naming the allowed values."""
```

A separate module rather than more lines in `station.py` (491 lines already), for the same reason
`forkjoin_policy.py` holds the `R_STAR_*` constants: the constants need the *explanation* of what
level and slope calibration mean beside them, and that explanation is not station plumbing.

Exported from `qopt/__init__.py`: `ZETA_LEVEL`, `ZETA_SLOPE`, `ZETA_SHAPE_TOL`,
`resolve_zeta_mode`.

### 3.2 Station constructor

Keyword-only `zeta_mode=ZETA_LEVEL`, passed through `resolve_zeta_mode` in `Station.__init__`,
stored to `_zeta_mode`, exposed as a **read-only** `zeta_mode` property. Validation happens once,
at construction, so `zeta_from` never has to consider an unknown mode — it branches on
`ZETA_SLOPE` and treats everything else as level.

**The keyword is `zeta_mode=`, not `zeta=`.** `Station.zeta(S)` is already a method
(`qopt/station.py:153`), so an attribute named `zeta` would shadow it. Read-only for the reason
`ForkJoinStation.policy` is read-only: nothing should change a calibration mid-run.

Threaded through `SingleServerStation`, `GG1Station`, `ForkJoinStation`, and the `GG1Station.mm1`
/ `GG1Station.md1` presets. `Network` needs no change — it binds γ, it does not construct
stations.

```python
GG1Station(0.6, 1.5, c=2.0, cov_a=2.0, cov_s=2.0, zeta_mode=ZETA_SLOPE)
ForkJoinStation(0.45, 1.0, r=4.0, c1=4.0, c2=1.0, r_star="tuned")   # stays level
```

---

## 4. The derivative and φ hooks

Three new methods on `Station`.

### 4.1 `dT_dS(S)` — base implementation is a central difference

```python
def dT_dS(self, S):
    h = _FD_STEP * (S - self.gamma / self.mu)      # _FD_STEP = 1e-7, module-private
                                                   # to station.py
    return (self.sojourn_time(S + h) - self.sojourn_time(S - h)) / (2.0 * h)
```

Not an abstract method: slope ζ then works for **any** station, including user subclasses, and no
existing subclass breaks. `probe-output.txt` §1 validates this route — the closed forms below agree with
exactly this difference to `6.7e-09` relative across every case tested.

The step is scaled to **spare capacity**, which buys two properties:

- `S − h > γ/μ` always, by construction, so the difference can never step across the stability
  boundary.
- At `S == γ/μ` exactly, `h = 0` and `sojourn_time` raises `InstabilityError` before any division
  by zero.

For a fork-join, differencing `sojourn_time` varies `S` along the **fixed current ray**, so this
default yields the radial derivative §4.3 requires. It cannot accidentally reproduce the
`_dt_dm1` defect.

### 4.2 `GG1Station.dT_dS` — closed form

`E[T] = 1/m + kγ/(m(m−γ))` with `m = Sμ`, `k = (cov_a² + cov_s²)/2`, so

```python
def dT_dS(self, S):
    m = S * self.mu
    self._check_stable(m)
    x = m - self.gamma
    k = (self.cov_a ** 2 + self.cov_s ** 2) / 2.0
    return -self.mu * (1.0 / m ** 2 + k * self.gamma * (2 * m - self.gamma) / (m * x) ** 2)
```

### 4.3 `ForkJoinStation.dT_dS` — the radial derivative, and why not `_dt_dm1`

```python
def dT_dS(self, S):
    a, b = self.mu, self.mu * self.r          # EFFECTIVE rates: a binds, b >= a
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

Two things make this correct and make the existing helper wrong for this job:

1. **`t_bot` needs no `max()`.** `t_ul` computes `t_bot = max(1/(m1−λ), 1/(m2−λ))`, which resolves
   to `1/x1` here because `_anchor` pairs `mu` with the slower server and keeps `r ≥ 1`, so
   `m1 ≤ m2` always. Along a ray both rates scale together, so `min(m1, m2)` never switches and
   the derivative is smooth.
2. **`forkjoin_policy._dt_dm1` must not be used.** It takes the "m1 is the non-bottleneck" branch
   at `m1 == m2` and drops the `α·t_bot` term entirely. That is harmless inside
   `_min_on_spend_line`, whose docstring argues correctly that the kink is measure-zero there —
   but it is wrong for pricing `dT/d(spend)`, and `r* = 1` is exactly where tight budgets and
   every `β₁ = β₂` station sit. `findings.md` §5 measures the error at **17.3% / 12.0% / 4.3%**
   for spend/floor of 1.05 / 1.5 / 4. This is not hypothetical: building the reference optimum
   with `_dt_dm1` made the exact optimum appear 0.1% *worse* than the shipped loop, which is how
   the defect was found.

`alpha` is homogeneous of degree −1 in `S`, giving `dα/dS = −α/S` — the first term.

### 4.4 `phi(S)` and its guard

```python
def phi(self, S):
    """Elasticity of E[T] in spare capacity: -d log T / d log x. 1.0 <=> eq 22 is already right."""
    x = S * self.mu - self.gamma
    return -self.dT_dS(S) * x / (self.mu * self.sojourn_time(S))
```

**`phi` uses the station's own analytic `sojourn_time`, always — including on the simulated
path**, where `zeta_from` receives a *measured* `T`. This is `findings.md` §6's hybrid,
`ζ = φ_analytic · T_measured · x`, and it is load-bearing in both directions:

- A purely slope-calibrated ζ from an analytic derivative cancels `T` out entirely, and **the
  simulator would stop influencing the allocation at all** — the thing eq 22 exists to prevent.
- The obvious alternative, a secant slope from consecutive loop iterates, does not work: the
  iterates converge, so it degenerates to 0/0 over a shrinking interval while differencing two
  CRN-noisy numbers.

Consequence: φ issues no simulation requests, so `sim_calls` is unchanged. It costs two analytic
`sojourn_time` evaluations per slope station per iteration, against one simulation run.

**Guard.** `zeta_from` validates `φ` finite and `> 0` in slope mode and raises `ValueError`
naming the station and `S`. `allocate` already rejects non-positive ζ, so this is defence with a
better message; a violation means a modelling error, not a budget problem. `ValueError` matches
how `allocate` rejects bad ζ.

**No clamp.** φ is legitimately tiny for some stations — a `cov = 0` station has `φ = 1 − ρ`
*exactly*, because `E[T] = 1/m` has no queueing term, so extra capacity buys it little
(measured minimum `0.000001`). ζ_slope `= φ·T·x` stays strictly positive and `allocate`'s
`sqrt` handles it. Pinned by a test at that extreme (§9).

---

## 5. The ζ seam

### 5.1 `Station.zeta_from` — one branch, in the base class

```python
def zeta_from(self, T, S):
    x = S * self.mu - self.gamma
    if self._zeta_mode == ZETA_SLOPE:
        phi = self.phi(S)
        if not (math.isfinite(phi) and phi > 0.0):
            raise ValueError(<names the station, S, and phi>)
        return phi * T * x
    return T * x
```

No station overrides `zeta_from` today, so this is the only place the branch appears. The level
arm is the same subexpression in the same order as today's `T * (S * self.mu - self.gamma)`;
binding it to a local does not change float semantics, so the default path is bit-for-bit
identical. Pinned with `==`, not `approx` (§9).

### 5.2 `Optimizer._noise_floor` — unchanged

It propagates a CI half-width through this same hook (`qopt/optimizer.py:340`). Slope ζ is
**linear in `T`**, so `zeta_from(h, S) = φ·h·x` is the correctly-propagated perturbation with no
special-casing. This is the property that makes the whole design cheap, and it is pinned as a
test in its own right (§9).

### 5.3 What must **not** go in `zeta_from`

The measured-vs-analytic cross-check of §8.4 belongs in the **optimizer loop, not in
`zeta_from`** — because `_noise_floor` calls `zeta_from` with a *CI half-width* in the `T`
position. A shape check inside the hook would compare a half-width against a sojourn time and
fire spuriously on every stochastic iteration. `zeta_from` cannot distinguish its callers, so the
check goes where the caller knows what it is passing.

---

## 6. `Result` reporting

Three new fields, all defaulted so the direct `Result(...)` constructions in
`tests/test_optimizer_loop.py:97`, `tests/test_example_simulated.py:29` and
`tests/test_example_qcsc.py` keep working untouched:

```python
zeta_phi: list = field(default_factory=list)          # per-station phi; 1.0 for level stations
zeta_mode: list = field(default_factory=list)         # per-station calibration
zeta_shape_flags: list = field(default_factory=list)  # section 8.4 cross-check messages
```

`Result.zeta` holds **the ζ that actually drove the allocation**, so the Result explains its own
capacities. `zeta_mode` is per-station because the selection is, so a mixed network stays
legible.

For a level-mode station `zeta_phi` is **literally `1.0`, not a computed φ**. Level mode never
consults the derivative, and the default path should neither start paying for one nor acquire a
new way to fail.

For a slope station φ is recomputed in the final reporting block (`qopt/optimizer.py:303`), which
runs after the loop's last `retune`, at the same `S` and the same station state as the
`zeta_from` call beside it. So the recomputation is bit-identical, and the eq-22 level value is
recoverable as `zeta[i] / zeta_phi[i]` up to a single rounding.

---

## 7. The `retune` ordering is load-bearing

`findings.md` §7 records an interaction in the opposite direction from the expected one. Slope ζ
needs `dT/dS` along the station's **current** ray, and §5 of that document shows the radial
derivative equals the true marginal only *on* the optimal ray — agreement to `3.3e-08` on the
optimal ray, against `20.4%` and `12.3%` disagreement at `r* = 1.0` and `4.0`.

This works out only because `Optimizer` calls `retune` **last** in each iteration
(`qopt/optimizer.py:241`), so the next iteration's `zeta_from` sees a ray already optimal for the
spend the station holds. That placement is currently documented as *tidiness, not correctness*.
Slope ζ makes it correctness.

Two deliverables:

1. A comment at the retune site recording this, so a future reorder meets the reason first.
2. A regression test anchored on the reference objectives in `probe-output.txt` §3.

**Honesty requirement.** Implementation must verify by mutation whether reordering the retune
actually moves those numbers enough to fail that test. If it does not, the spec and the comment
must say the ordering is *documented but not test-pinned*, rather than claim a pin that does not
exist. A comment that only looks load-bearing is worse than one that admits its limit.

---

## 8. `cov_a` on the simulated path — the new coupling

This section exists because of a review point raised during design, and it is the one place where
slope ζ changes something beyond the allocation arithmetic.

### 8.1 What the code does today

`cov_a` appears in **exactly one place** in the package: `GG1Station.sojourn_time`
(`qopt/station.py:244`). It is **not** emitted to the simulator — `sim_node` sends only `cov_s`,
as the service distribution's SCV (`:265`). The arrival process a station sees in simulation is
emergent: `Network`'s source `arrival_scv` (default 1.0), plus routing, superposition and
upstream departure processes. `qopt/traffic.py` propagates **rates only**, never second moments.

Nor does the simulator report it back. `MEASURES` in `qopt/qsim/spec.py` is a deliberately closed
list — `response-time`, `system-response-time`, `throughput`. No interarrival statistics.
(Whether `qsim-service` *could* produce them is unverified; see §8.6. It is not a settled "no".)

So **on the simulated path today, `cov_a` is inert**: `E[T]` is measured, `ζ = T·x`, and the
constructed `cov_a` influences only the analytic warm start. Under slope ζ, φ is analytic, so
`cov_a` enters **every allocation**. No code change is required for correctness — but a
parameter that was nearly decorative on that path becomes load-bearing.

### 8.2 Sensitivity

From the closed form `φ = (x² + kγ(2m−γ))/(m(x + kγ))`, `k = (cov_a² + cov_s²)/2`, at γ = 0.6,
`cov_s` = 1:

| ρ | true `cov_a` | used `cov_a` | φ_true | φ_used | φ error |
|---|---|---|---|---|---|
| 0.30 | 1.0 | 3.0 | 1.0000 | 1.3818 | +38.2% |
| 0.67 | 1.0 | 3.0 | 1.0000 | 1.2403 | +24.0% |
| 0.95 | 1.0 | 3.0 | 1.0000 | 1.0396 | +4.0% |
| 0.67 | 2.0 | 1.0 | 1.1654 | 1.0000 | −14.2% |

These numbers are a property of the closed form in §4.2 and are **pinned by a test** (§9), which
is their source of record — not a scratch file.

Two consequences, the first of which corrects how `findings.md` §6 applies here:

1. **§6's robustness result does not cover this case.** §6 blends between a *correct* φ and 1 and
   concludes even a 100% overcorrection beats doing nothing. A wrong `cov_a` can instead push φ
   away from a *correct* value of φ = 1 — the M/M/1 case, where level calibration was already
   exact. The effective blend factor is then unbounded, so overstating `cov_a` **can land worse
   than the incumbent**, which no row of §6's table does.
2. **The error is one-directional in a useful way.** Understating `cov_a` drives φ toward 1, which
   degrades gracefully to level-mode behaviour: no gain, no loss. Overstating it is the dangerous
   direction. The error is worst at low load and self-limiting at high load (+4.0% at ρ = 0.95).
   [**Correction (2026-09-23, final review I-1):** every row of the table above is computed at
   `cov_s = 1`, and the one-directional claim holds in `k = (cov_a² + cov_s²)/2`, not in `cov_a`
   alone. φ is strictly increasing in `k` and equals 1 exactly at `k = 1`, so understating `cov_a`
   drives φ toward 1 only while `k` stays above 1; once `k < 1` it drives φ *past* 1 in the other
   direction, which is a loss and not a graceful degradation. See §8.3's correction block.]

### 8.3 The safe-default rule, documented

**When the arrival SCV is unknown, `cov_a = 1` is the safe assumption** — it forfeits the gain
rather than overshooting past it. Goes in the `zeta_mode` docstring and the README, together with
the statement that in slope mode `cov_a` must describe the arrival process the station *actually*
sees, internal traffic included.

**Correction (2026-09-23, final review I-1):** the rule above is wrong whenever `cov_s ≠ 1`, and
the shipped wording in `README.md`'s ζ-calibration subsection is the corrected one. φ depends on
the two coefficients of variation only through `k = (cov_a² + cov_s²)/2`, is strictly increasing
in `k`, and equals 1 exactly at `k = 1`. The assumption that reproduces level calibration — and
so forfeits the gain rather than overshooting past it — is therefore `k = 1`, i.e.
`cov_a = √(2 − cov_s²)`; `cov_a = 1` is level-equivalent only at `cov_s = 1`, which is the
`cov_s` §8.2's table is computed at and where this rule was generalised from. For a
deterministic-service station (`cov_s = 0`) assuming `cov_a = 1` gives `k = 0.5` and φ < 1 at
every load (≈ 0.833 at ρ = 0.5) — the far side of 1 from any truth with `k > 1`, and measured
worse than eq 22 rather than degrading to it. When `cov_s > √2` no arrival process reaches
`k = 1` at all and φ > 1 whatever is assumed, so the closest approach to level calibration is
`cov_a = 0`.

### 8.4 The measured-vs-analytic `E[T]` cross-check

**Where.** In the optimizer loop, after `evaluate`, for slope-mode stations only — not in
`zeta_from` (§5.3).

**Cadence.** Checked every iteration, but **warned at most once per station per run**: the
condition it detects is a constructor argument, so it cannot heal between iterations, and a
20-iteration simulated run would otherwise emit 20 copies of the same warning. One flag per
offending station lands in `Result.zeta_shape_flags`.

**What.** Compare measured `E[T]` against the station's analytic `sojourn_time(S)` at the same
`S`. Flag when `|T_measured/T_analytic − 1| > zeta_shape_tol`.

**Why it is the right detector.** A wrong φ produces a converged, plausible, quietly suboptimal
answer — there is no symptom. The comparison is nearly free, since φ already evaluates
`T_analytic`, and it tests exactly the shape assumption φ rests on. It also *amplifies* what it
is detecting: `cov_a = 3` where the truth is 1 at ρ = 0.67 is a **+24% error in φ but a +268%
error in `E[T]`** (3.385 → 12.45), so the diagnostic is far more sensitive than the defect is
large.

**Threshold: `ZETA_SHAPE_TOL = 0.25`**, bracketed by evidence two orders of magnitude apart:

- *Legitimate disagreement*, from the committed 14-station run in
  `docs/qcsc-example/live-run.log`: every analytic-vs-simulated gap is within **±1.1%** (max
  −1.09% at `qpu_psq`), including the two fork-join stations that use the approximate `t_ul`
  (−1.04%, +0.05%).
- *A blunder*: hundreds of percent, as above.

25% sits ~20× above observed legitimate disagreement and ~10× below blunder scale.

**Honest limit on that calibration.** The QCSC network contains M/M/1 and fork-join stations only.
It has no `G/G/1` with `cov ≠ 1` — which is precisely the station type slope ζ most benefits and
where the Allen–Cunneen approximation's own error is largest. So the lower end of the bracket is
under-sampled for the case that matters most. The threshold is therefore **configurable and the
flag non-fatal**: `Optimizer(..., zeta_shape_tol=ZETA_SHAPE_TOL)`, with `None` disabling the
check.

**How it surfaces.** A `RuntimeWarning` plus an entry in the new `Result.zeta_shape_flags`.

**Deliberately not `Result.degraded`.** `degraded` is the simulation-quality audit (spec 6.8,
7.2), and `Optimizer.strict` raises `SimulationQualityError` on any entry. A `cov_a` mismatch is a
**model-specification** signal, not a simulation-quality one; routing it through `degraded` would
make `strict=True` abort runs whose simulation was fine. A separate field keeps `strict`'s meaning
intact while still surfacing the signal.

**Vacuous on the analytic path, by construction.** `AnalyticAnalyzer.evaluate` returns
`st.sojourn_time(Si)`, and the check recomputes `st.sojourn_time(Si)` at the same `S` and station
state, so the ratio is exactly 1.0 and the check can never fire. Cost there is one comparison.

### 8.5 The override hook, documented

`phi(S)` is a plain method. A user with a better estimate — a measured interarrival SCV, an
analytical superposition argument — overrides it on a subclass, with **no new API**. This is the
supported route for a station whose true arrival variability is known but not expressible as a
constructor `cov_a`, and it will be documented as such.

### 8.6 Follow-up, not code

File an issue: can `qsim-service` report per-station interarrival SCV, and if so should slope ζ
consume it to close the loop? Worth asking rather than guessing — `MEASURES` is closed by design,
and adding to it has known fork-join hazards (two of qsim's own defaults come back as
join-station numbers with `success: true` and no warning).

---

## 9. Testing

New `tests/test_zeta.py`. Per this repo's habit, **every new assertion gets mutation-checked** —
revert the behaviour and confirm the test fails, with bytecode invalidation forced so a stale
`.pyc` cannot fake a green run.

| # | Test | Why |
|---|---|---|
| 1 | `φ ≡ 1` for M/M/1 across loads (≤1e-12) | The invariant that explains why this went unnoticed |
| 2 | `φ = 1 − ρ` exactly for `cov_a = cov_s = 0` | Closed-form boundary case; φ's smallest values |
| 3 | Both closed forms vs central differences, over loads / `cov` / `r` / `r_star` incl. `r_star = 1.0` | Correctness, and the drift guard if `t_ul` ever changes. `r_star = 1.0` is where `_dt_dm1` is wrong by 17.3% |
| 4 | Level mode is bit-for-bit `T*(S*μ − γ)`, via `==` | §5.1's bit-for-bit claim |
| 5 | Unspecified `zeta_mode` resolves to `ZETA_LEVEL` | The default |
| 6 | `zeta_from(2T, S) == 2·zeta_from(T, S)` | The linearity `_noise_floor` depends on (§5.2) |
| 7 | Slope mode end-to-end vs the reference objectives in `probe-output.txt` §3 | The payoff, and §7's ordering anchor |
| 8 | Level-vs-slope objective gap on the mixed network, against `findings.md` §4's 0.515% | That the gain is real, not just the fixed point moving |
| 9 | Mixed network: one slope station, one level station; modes reported | Per-station selection works and is legible |
| 10 | A custom `Station` subclass with no closed form runs on the fd default | §4.1's no-abstract-method promise |
| 11 | Bad mode string names the allowed values; the φ guard fires | §3.1, §4.4 validation |
| 12 | A run at the `φ → 1e-6` extreme completes | §4.4's no-clamp decision |
| 13 | `Result.zeta_phi` is `1.0` for level stations without evaluating a derivative | §6's "does not start paying for one" |
| 14 | `zeta[i]/zeta_phi[i]` recovers the eq-22 value | §6's recoverability claim |
| 15 | The §8.2 sensitivity table | Its source of record |
| 16 | Cross-check fires above tolerance, stays silent below, is vacuous on the analytic path, and does **not** enter `degraded` | §8.4, including the `strict` interaction |

---

## 10. Documentation ripple

- **`README.md`** — the new option, the constants, `Result.zeta`'s mode dependence, §8.3's
  `cov_a` rule, and §8.5's override hook.
- **`docs/slope-calibrated-zeta/README.md`** — currently states "**Nothing here is
  implemented**". Becomes false.
- **`docs/slope-calibrated-zeta/findings.md`** — the "Status: proposal. Not implemented." header
  and §8's sequencing step 3.
- **`docs/paper-map.md`** — record the deliberate divergence from eq 22.

**Constraint:** `docs/quotes-selfcheck.py` requires every fenced quote in `findings.md` to match
`probe-output.txt` verbatim. **Prose only — no fence is touched.** Both
`docs/quotes-selfcheck.py` and `docs/audit-selfcheck.py` run as part of the work.

---

## 11. Out of scope

- **Redoing the contraction proof.** `findings.md` §7 notes that
  `docs/convergence-tuned-r-star`'s analysis is for the current map; slope ζ changes the map. That
  is a docs task on local material, and the measurements show no practical change (iterations
  5–18 vs 6–18, zero failures, zero off-optimum rows over 13 budgets from `1.0001×` to `1e4×` the
  floor).
- **The `findings.md` §6 blend knob `f`.** Stays in the probe.
- **Any change to the paper or `docs/analysis.pdf`.**
- **Changing the default.** Stays `ZETA_LEVEL`.
- **Issue #14** (tuned stations stay mutated after a run). Slope ζ adds no new mutation.
- **Consuming a measured interarrival SCV** (§8.6) — an issue, not this implementation.

---

## 12. Size estimate

| File | Change |
|---|---|
| `qopt/zeta.py` | new, ~70 lines |
| `qopt/station.py` | +~90 |
| `qopt/optimizer.py` | +~30 (Result fields, φ reporting, the §8.4 check) |
| `qopt/__init__.py` | +4 exports |
| `tests/test_zeta.py` | new, ~300 |
| `README.md` | +~35 |
| docs updates | §10 |
