"""Operators for temporal transport and host-graph coupling."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from jf_utils import get_prop_any


def periodic_delta(a, b, box_size):
    """Compute the periodic displacement vector between two positions."""
    delta = a - b
    if box_size is None:
        return delta
    return delta - np.round(delta / box_size) * box_size


def build_T(prev_layer, curr_layer):
    """Construct the temporal transport matrix from layer at t-1 to layer at t."""
    ids_prev = prev_layer["id2idx"]
    ids_curr = curr_layer["id2idx"]
    rows, cols, data = [], [], []
    for source, target in curr_layer["time_edges"]:
        if source in ids_prev and target in ids_curr:
            rows.append(ids_curr[target])
            cols.append(ids_prev[source])
            data.append(1.0)
    if not rows:
        shared = set(ids_prev.keys()).intersection(ids_curr.keys())
        for node_id in shared:
            rows.append(ids_curr[node_id])
            cols.append(ids_prev[node_id])
            data.append(1.0)
    n_curr = len(curr_layer["nodes"])
    n_prev = len(prev_layer["nodes"])
    return coo_matrix((data, (rows, cols)), shape=(n_curr, n_prev)).tocsr()


def build_host_L(layer, box_size=None, xi=1.0):
    """Construct the host Laplacian based on spatial proximity."""
    nodes = layer["nodes"]
    n_nodes = len(nodes)
    positions = np.zeros((n_nodes, 3), dtype=float)
    has_positions = True
    for idx, node in enumerate(nodes):
        pos = get_prop_any(node, "pos")
        if pos is None or len(pos) < 3:
            has_positions = False
            break
        positions[idx] = np.array(pos, dtype=float)[:3]

    radii = np.ones(n_nodes, dtype=float)
    has_radii = True
    for idx, node in enumerate(nodes):
        radius = get_prop_any(node, "R200c")
        if radius is None:
            has_radii = False
            break
        try:
            radii[idx] = float(radius)
        except Exception:
            has_radii = False
            break

    rows, cols, data = [], [], []
    if len(layer["host_edges"]) == 0 or not has_positions:
        return csr_matrix((n_nodes, n_nodes), dtype=float)

    if not has_radii:
        sample = np.random.choice(n_nodes, size=min(n_nodes, 256), replace=False)
        dmins = []
        for idx in sample:
            dmin = np.inf
            for jdx in sample:
                if idx == jdx:
                    continue
                dv = periodic_delta(positions[idx], positions[jdx], box_size)
                dist = float(np.linalg.norm(dv))
                if dist < dmin:
                    dmin = dist
            if np.isfinite(dmin):
                dmins.append(dmin)
        scale = np.median(dmins) if dmins else 1.0
        radii[:] = scale

    ids = layer["id2idx"]
    for source, target in layer["host_edges"]:
        if source not in ids or target not in ids:
            continue
        i = ids[source]
        j = ids[target]
        dv = periodic_delta(positions[i], positions[j], box_size)
        d2 = float(np.dot(dv, dv))
        sigma_i = (xi * radii[i]) ** 2 + 1e-30
        sigma_j = (xi * radii[j]) ** 2 + 1e-30
        w_ij = np.exp(-d2 / (2.0 * sigma_i))
        w_ji = np.exp(-d2 / (2.0 * sigma_j))
        weight = 0.5 * (w_ij + w_ji)
        if weight <= 0:
            continue
        rows.append(i)
        cols.append(j)
        data.append(weight)
        rows.append(j)
        cols.append(i)
        data.append(weight)

    if not rows:
        return csr_matrix((n_nodes, n_nodes), dtype=float)

    W = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    degree = np.array(W.sum(axis=1)).ravel()
    return diags(degree) - W


__all__ = ["periodic_delta", "build_T", "build_host_L"]
