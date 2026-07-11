# Queueing Network Capacity Allocation Optimizer

A Python optimizer that allocates resource capacities `S` across a **network of queues** to
minimize the sum of weighted expected sojourn times subject to a budget and stability
constraints. It implements the Section-5 fixed-point iteration of the analysis paper
(`docs/analysis.pdf`, "Optimization and Performance Analysis of Resource Allocation in
Quantum-Centric Supercomputing Environments").

## Model

The network is a collection of **stations**. Two types:

- **Single-server queue** — a G/G/1 queue analyzed with the Kingman / Allen–Cunneen
  mean-value approximation, parameterized by the coefficients of variation of interarrival
  (`cov_a`) and service (`cov_s`) times. M/M/1 (`cov_a=1, cov_s=1`) and M/D/1
  (`cov_a=1, cov_s=0`) are presets; the approximation is exact for any M/G/1.
- **Fork-join queue** — two parallel servers (ratio `r ≥ 1`), analyzed with the UL
  (upper–lower bound interpolation) approximation.

Arrival rates `γ` are fixed per-station constants; the optimizer iterates on the capacity
vector `S` until the optimal `S*` is reached.

## Scope & limitations

Each station is analyzed **independently** from its own arrival rate and coefficients of
variation. The optimizer does **not** model how one station's *departure* process shapes the
*arrival* variability of downstream stations — i.e. variability propagation through the
network is not captured. Doing so faithfully requires **simulation** of the whole network
rather than closed-form per-station analysis. The current per-station analysis is therefore
an **approximation**, and full network coupling is a planned area for **future work**
(the same extension seam as a future simulation-based analyzer).

## Status

Design phase. See:

- `docs/superpowers/specs/2026-07-10-optimizer-design.md` — authoritative design spec.
- `docs/optimizer-brainstorm-summary.md` — problem statement and design rationale.
