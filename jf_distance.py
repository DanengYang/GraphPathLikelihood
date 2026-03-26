"""Graph-distance utilities for layered halo graphs.

We define environment bins using the shortest-path distance on the host-edge
subgraph of a single snapshot (typically z=0). Host edges are treated as
undirected for distance computation.
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np


def _node_is_central(node: dict) -> bool:
    props = node.get("props")
    if isinstance(props, dict) and "is_central" in props:
        return bool(props["is_central"])
    return bool(node.get("is_central", False))


def compute_host_graph_distances(layer: dict) -> np.ndarray:
    """Return host-graph distances to the central(s) for each connected component.

    The layer is expected to contain:
      - layer["nodes"]: list of node dicts
      - layer["id2idx"]: mapping from node id -> index
      - layer["host_edges"]: list of (child_id, host_id) edges (directed),
        treated as undirected here.

    If a component contains nodes flagged as central, we compute distances to
    those central nodes. Otherwise we fall back to using the maximum-degree node
    as the distance source (legacy behavior).
    """
    nodes: List[dict] = layer.get("nodes", []) or []
    n = len(nodes)
    if n == 0:
        return np.zeros(0, dtype=int)

    id2idx = layer.get("id2idx", {}) or {}
    adj: List[List[int]] = [[] for _ in range(n)]
    for child, host in (layer.get("host_edges", []) or []):
        if child in id2idx and host in id2idx:
            i = int(id2idx[child])
            j = int(id2idx[host])
            if 0 <= i < n and 0 <= j < n:
                adj[i].append(j)
                adj[j].append(i)

    is_central = [_node_is_central(node) for node in nodes]
    deg = [len(nei) for nei in adj]

    dist = np.full(n, np.inf, dtype=float)
    seen = [False] * n

    for start in range(n):
        if seen[start]:
            continue
        # Find connected component
        comp: List[int] = []
        q = deque([start])
        seen[start] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for nb in adj[v]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)

        if not comp:
            continue

        sources = [v for v in comp if is_central[v]]
        if not sources:
            max_deg = max((deg[v] for v in comp), default=0)
            sources = [v for v in comp if deg[v] == max_deg] or [comp[0]]

        # Multi-source BFS
        q = deque(sources)
        dmap = {s: 0 for s in sources}
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in dmap:
                    continue
                dmap[nb] = dmap[cur] + 1
                q.append(nb)
        for v in comp:
            dist[v] = dmap.get(v, np.inf)

    dist[np.isinf(dist)] = 0.0
    return dist.astype(int)


__all__ = ["compute_host_graph_distances"]

