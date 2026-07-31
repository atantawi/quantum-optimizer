"""Traffic equations: derive per-station arrival rates from a topology (spec 4)."""

from qopt.exceptions import TopologyError


def solve_traffic(nodes, edges, arrival_rate, source, sink, *, tol=1e-12, max_iter=10_000):
    """Solve lambda = lambda_ext + P^T lambda by fixed-point iteration from lambda = 0.

    Converges geometrically for any open chain, including branching and feedback cycles.

    Does not itself validate that edge endpoints resolve to `nodes`, `source`, or
    `sink`; a typo raises a plain KeyError from the dict lookups below rather than a
    TopologyError. Endpoints are assumed pre-validated by the caller. The only
    production caller is Network.__init__, which runs Network._validate() immediately
    before calling this function and performs exactly that check — so it owns the
    obligation. A future direct caller of solve_traffic must do the same.

    Args:
        nodes: station names — the unknowns. `source` and `sink` are not among them.
        edges: (src, dst, probability) triples; endpoints may be `source` or `sink`.
        arrival_rate: exogenous lambda_0 entering at `source`.
        source, sink: endpoint sentinel names.
        tol: stop once max|delta lambda| < tol.
        max_iter: iteration cap. Hitting it means flow is trapped in a closed
            subnetwork, which is a structural error rather than slow convergence. Must
            be >= 1: at 0 the loop body never runs and this function raises
            TopologyError reporting a residual (max|delta lambda|) of 0, which would
            misleadingly read as "converged to zero" rather than "never ran". This is a
            documented precondition, not a guarded one — callers must pass max_iter >= 1.

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
