# Design Spec: `qsim-service` — moved

Date: 2026-07-25

The JMT-backed queueing-network **simulation service** (`qsim-service`) now lives in its own
repository, and the authoritative design spec has moved there:

- **Repo:** `Projects/quantum/qsim-service` (standalone, GPL v2 — separate from this Apache-2.0 repo)
- **Spec:** `docs/superpowers/specs/2026-07-25-qsim-service-design.md` in that repo

Why separate: the service links JMT (GPL v2-or-later) engine classes in-process, so it is a GPL
derivative. Keeping it in its own repo behind an HTTP/JSON boundary is the licensing firewall —
`qopt` consumes it as a plain HTTP client and stays Apache-2.0-clean.

The **`qopt`-side integration** (a `SimulationAnalyzer` client at the existing
`Station.sojourn_time` seam, plus noise-aware fixed-point convergence) will be its own spec in
*this* repo when we take that work on.
