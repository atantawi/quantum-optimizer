# Design Spec: `qsim-service` — moved

Date: 2026-07-25

The JMT-backed queueing-network **simulation service** (`qsim-service`) now lives in its own
repository, and the authoritative design spec has moved there:

- **Repo:** `Projects/quantum/qsim-service` (standalone, GPL v2 — separate from this Apache-2.0 repo)
- **Spec:** `docs/superpowers/specs/2026-07-25-qsim-service-design.md` in that repo

Why separate: the service links JMT (GPL v2-or-later) engine classes in-process, so it is a GPL
derivative. Keeping it in its own repo behind an HTTP/JSON boundary is the licensing firewall —
`qopt` consumes it as a plain HTTP client and stays Apache-2.0-clean.

The **`qopt`-side integration** now has its own spec in this repo:
[`2026-07-29-simulation-support-design.md`](2026-07-29-simulation-support-design.md).

One correction to what this pointer originally anticipated: that spec does **not** hang a
`SimulationAnalyzer` off the existing `Station.sojourn_time` seam. A simulation answers for every
station in a single run, so per-station evaluation is the wrong shape — a station's method would
silently depend on sibling state, and the one-POST-per-iteration property would be emergent rather
than stated. The spec introduces a network-level `Analyzer` seam instead; see its §2.1 for the
rejected alternatives.
