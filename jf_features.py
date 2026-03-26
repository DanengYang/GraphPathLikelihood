"""Legacy feature helpers retained for compatibility (e.g. layer redshift)."""

from __future__ import annotations

import numpy as np

from jf_extractors import extract_field_array
from jf_operators import build_T, build_host_L
from jf_time import scale_to_redshift, snap_to_scale_factor


def log_safe(arr, min_val=1e-12):
    return np.log(np.clip(arr, min_val, None))


def log1p_safe(arr):
    return np.log1p(np.clip(arr, -0.999999, None))


def layer_redshift(layer):
    snap = layer.get("snap")
    scale = snap_to_scale_factor(snap)
    if scale is None:
        scale = layer.get("scale_factor")
    if scale is None:
        return None
    z = scale_to_redshift(scale)
    return float(z) if z is not None else None


def compute_host_features(layer):
    """Return host index, in/out degree derived from host edges."""
    n = len(layer["nodes"])
    id2idx = layer["id2idx"]
    host_index = np.full(n, -1, dtype=int)
    indegree = np.zeros(n, dtype=float)
    outdegree = np.zeros(n, dtype=float)
    for src, tgt in layer["host_edges"]:
        if src in id2idx and tgt in id2idx:
            si = id2idx[src]
            ti = id2idx[tgt]
            host_index[si] = ti
            outdegree[si] += 1.0
            indegree[ti] += 1.0
    return host_index, indegree, outdegree


def build_enriched_features(prev_layer, curr_layer, field, extra_fields=None):
    """Construct legacy graph-native features (retained for archival compatibility)."""
    if extra_fields is None:
        extra_fields = []

    n = len(curr_layer["nodes"])
    features = {}

    prev_vals = extract_field_array(prev_layer, field)
    curr_mass = extract_field_array(curr_layer, "M_halo")
    prev_mass = extract_field_array(prev_layer, "M_halo")

    T = build_T(prev_layer, curr_layer)
    transported_prev = T @ np.nan_to_num(prev_vals, nan=0.0)
    transported_mass = T @ np.nan_to_num(prev_mass, nan=0.0)

    features["log1p_prev"] = log1p_safe(transported_prev)
    features["prev_raw"] = np.nan_to_num(transported_prev, nan=0.0)
    features["log_mass"] = log1p_safe(curr_mass)
    features["delta_log_mass"] = log1p_safe(curr_mass) - log1p_safe(transported_mass)

    host_index, indegree, outdegree = compute_host_features(curr_layer)
    features["log1p_host_in_degree"] = log1p_safe(indegree)
    features["log1p_host_out_degree"] = log1p_safe(outdegree)

    host_mass = np.zeros(n, dtype=float)
    host_mass.fill(np.nan)
    for i, h_idx in enumerate(host_index):
        if h_idx >= 0:
            host_mass[i] = curr_mass[h_idx]
    features["log1p_host_mass"] = log1p_safe(np.nan_to_num(host_mass, nan=0.0))
    ratio = np.divide(np.nan_to_num(host_mass, nan=0.0), np.clip(curr_mass, 1e-12, None))
    features["host_mass_ratio"] = np.nan_to_num(ratio, nan=0.0)

    row_sum = np.array(T.sum(axis=1)).ravel()
    features["log1p_transport_degree"] = log1p_safe(row_sum)

    z = layer_redshift(curr_layer)
    if z is not None:
        features["redshift"] = np.full(n, float(z))
        features["log1p_redshift"] = log1p_safe(np.full(n, float(z)))
    else:
        features["redshift"] = np.zeros(n, dtype=float)
        features["log1p_redshift"] = np.zeros(n, dtype=float)

    L_host = build_host_L(curr_layer)
    if L_host.nnz > 0:
        avg_mass = np.zeros(n, dtype=float)
        avg_mass[:] = np.nan
        degree = np.array((L_host != 0).sum(axis=1)).ravel()
        degree_safe = np.where(degree == 0, 1.0, degree)
        adjacency = (L_host != 0).astype(float)
        for i in range(n):
            if degree[i] > 0:
                neighbours = adjacency.getrow(i).indices
                avg_mass[i] = np.mean(curr_mass[neighbours])
        features["avg_host_mass"] = np.nan_to_num(avg_mass, nan=0.0)
        features["log1p_avg_host_mass"] = log1p_safe(np.nan_to_num(avg_mass, nan=0.0))
        features["host_degree"] = degree_safe
    else:
        features["avg_host_mass"] = np.zeros(n, dtype=float)
        features["log1p_avg_host_mass"] = np.zeros(n, dtype=float)
        features["host_degree"] = np.zeros(n, dtype=float)

    for extra in extra_fields:
        prev_extra = extract_field_array(prev_layer, extra)
        transported_extra = T @ np.nan_to_num(prev_extra, nan=0.0)
        features[f"log1p_prev_{extra}"] = log1p_safe(transported_extra)
        current_extra = extract_field_array(curr_layer, extra)
        features[f"log1p_curr_{extra}"] = log1p_safe(current_extra)

    design = np.column_stack([features[k] for k in sorted(features.keys())])
    return design, sorted(features.keys())


__all__ = [
    "build_enriched_features",
]
