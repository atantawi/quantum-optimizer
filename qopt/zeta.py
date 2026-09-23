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
    # Membership against the two string constants is the whole check: a bool, a number,
    # a list and an ordinary object all compare unequal to "level" and "slope", so they
    # are rejected here with no separate type test -- which is what
    # test_a_non_string_mode_is_rejected documents. Two things ARE accepted, both
    # deliberately: a `str` SUBCLASS carrying a valid value, which compares and behaves
    # as the string it is; and any object whose own `__eq__` claims equality with one of
    # the constants, which `in` honours and which is then returned as-is. Neither is
    # reachable by accident, and a type test would not be worth buying them out.
    if mode not in ZETA_MODES:
        raise ValueError(
            f"zeta_mode must be one of {ZETA_MODES!r} (or None for "
            f"{ZETA_LEVEL!r}), got {mode!r}"
        )
    return mode
