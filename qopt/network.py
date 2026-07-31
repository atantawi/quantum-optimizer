"""Network: stations, probabilistic routing, and exogenous arrivals (spec 3, 4)."""

import math
from dataclasses import dataclass

from qopt.exceptions import TopologyError
from qopt.station import distribution_dict
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
