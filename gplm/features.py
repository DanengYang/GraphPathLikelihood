"""Feature construction utilities for GPLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from jf_extractors import extract_field_array
from jf_utils import canonical_field_name
from jf_features import compute_host_features, layer_redshift
from jf_operators import build_T


def compute_dt(prev_layer: dict, curr_layer: dict) -> float:
    """Return cosmic-time delta (in Gyr) between two layers."""
    t0 = prev_layer.get("time")
    t1 = curr_layer.get("time")
    if t0 is None or t1 is None:
        raise ValueError("Layer times missing; ensure jf_time.ensure_layer_times_physical was applied.")
    dt = float(t1 - t0)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Non-positive dt encountered: {dt}")
    return dt


LOG_EPS = 1e-6
MSTAR_EPS = 1e-6
LOG_TARGET_FIELDS = {"M_star", "M_gas", "M_wind", "M_bh", "SFR", "SubhaloSFR"}
LOG_TARGET_FIELDS_CANON = {canonical_field_name(name) for name in LOG_TARGET_FIELDS}
MSTAR_CANON = canonical_field_name("M_star")
MGAS_CANON = canonical_field_name("M_gas")

# Thresholds for zero-mass masking in increments
M_GAS_EPS = 1e-6


def safe_log(x: np.ndarray, offset: float, min_offset: float = LOG_EPS) -> np.ndarray:
    """Log transform with adjustable offset."""
    adj = max(offset, min_offset)
    return np.log(np.clip(x, 0.0, None) + adj)


def safe_log10(x: np.ndarray, offset: float, min_offset: float = MSTAR_EPS) -> np.ndarray:
    adj = max(offset, min_offset)
    return np.log10(np.clip(x, 0.0, None) + adj)


@dataclass
class LayerPairFeatures:
    """Container with per-layer quantities used for tensors."""

    node_features: np.ndarray
    node_targets: np.ndarray
    node_mask: np.ndarray
    halo_mass: np.ndarray
    transported_state: Dict[str, np.ndarray]
    curr_state: Dict[str, np.ndarray]
    dt: float
    redshift: float
    host_index: np.ndarray
    env_features: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None
    new_node_mask: np.ndarray


def build_edge_index(edge_list: List[Tuple[int, int]]) -> torch.Tensor:
    if not edge_list:
        return torch.zeros((2, 0), dtype=torch.long)
    arr = np.array(edge_list, dtype=np.int64)
    return torch.from_numpy(arr.T.copy())


def extract_fields(layer: dict, fields: Iterable[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for field in fields:
        arr = extract_field_array(layer, field)
        out[field] = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
    return out


def build_layer_pair_features(
    prev_layer: dict,
    curr_layer: dict,
    fields: Iterable[str],
    extra_node_features: Iterable[str] | None = None,
    use_spatial_features: bool = False,
    include_first_star: bool = True,
    env_feature_names: Sequence[str] | None = None,
    use_env_features: bool = False,
    env_state: Dict[str, Dict[int, float]] | None = None,
    mass_log_eps: float = 0.05,
    include_history_flags: bool = True,
) -> LayerPairFeatures:
    """Assemble numeric features for a pair of layers."""
    field_list = list(dict.fromkeys(fields))
    field_canonical = [canonical_field_name(name) for name in field_list]
    dt = compute_dt(prev_layer, curr_layer)
    transported = {}
    T = build_T(prev_layer, curr_layer)
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    new_node_mask = row_sums == 0.0

    for field in field_list:
        prev_vals = extract_field_array(prev_layer, field)
        transported[field] = T @ np.nan_to_num(prev_vals, nan=0.0)
    curr_state = extract_fields(curr_layer, field_list)

    halo_mass = curr_state.get("M_halo")
    if halo_mass is None:
        halo_mass = extract_field_array(curr_layer, "M_halo")

    target_list = []
    mask_list = []
    feature_columns = []
    env_feature_set = set(env_feature_names or [])
    has_star_prev_feature: np.ndarray | None = None
    has_gas_prev_feature: np.ndarray | None = None
    for field, canon in zip(field_list, field_canonical):
        truth = curr_state[field]
        prev = transported[field]
        if canon in LOG_TARGET_FIELDS_CANON:
            prev_clamped = np.clip(prev, 0.0, None)
            truth_clamped = np.clip(truth, 0.0, None)
            target = safe_log(truth_clamped, mass_log_eps) - safe_log(prev_clamped, mass_log_eps)
            mask = np.isfinite(target)
            if canon == MSTAR_CANON:
                zero_prev = prev_clamped <= MSTAR_EPS
                zero_curr = truth_clamped <= MSTAR_EPS
                has_star_prev_feature = (1.0 - zero_prev.astype(float))
                mask &= ~(zero_prev & zero_curr)
                if not include_first_star:
                    mask &= ~(zero_prev & (~zero_curr))
            elif canon == MGAS_CANON:
                # Mask double-zero gas segments and expose a has_gas_prev feature
                zero_prev_g = prev_clamped <= M_GAS_EPS
                zero_curr_g = truth_clamped <= M_GAS_EPS
                has_gas_prev_feature = (1.0 - zero_prev_g.astype(float))
                mask &= ~(zero_prev_g & zero_curr_g)
        else:
            target = truth - prev
            mask = np.isfinite(target)
        target = np.nan_to_num(target, nan=0.0)
        mask_list.append(mask.astype(np.float32))
        target_list.append(target)
        # Only expose transported (previous) values so inference matches training.
        feature_columns.append(safe_log(prev, mass_log_eps))

    if extra_node_features:
        extras = extract_fields(curr_layer, extra_node_features)
        for name in extra_node_features:
            feature_columns.append(safe_log(extras[name], mass_log_eps))

    targets = np.column_stack(target_list) if target_list else np.zeros((len(curr_layer["nodes"]), 0))
    mask = np.column_stack(mask_list) if mask_list else np.zeros_like(targets)
    if targets.size and new_node_mask.any():
        targets[new_node_mask, :] = 0.0
        mask[new_node_mask, :] = 0.0

    host_index, indegree, outdegree = compute_host_features(curr_layer)
    halo_mass = np.clip(halo_mass, 1e-12, None)
    host_mass = np.full_like(halo_mass, 1e-12)
    for idx, h_idx in enumerate(host_index):
        if h_idx >= 0:
            host_mass[idx] = max(halo_mass[h_idx], 1e-12)
    env_cols = [
        indegree,
        outdegree,
        safe_log(host_mass, mass_log_eps),
        safe_log(halo_mass, mass_log_eps),
        safe_log10(host_mass, mass_log_eps),
        safe_log10(halo_mass, mass_log_eps),
    ]
    feature_columns.extend(env_cols)

    # Additional environment features
    if include_history_flags:
        if MSTAR_CANON in field_canonical and has_star_prev_feature is not None:
            feature_columns.append(has_star_prev_feature.astype(np.float32))
        if MGAS_CANON in field_canonical and has_gas_prev_feature is not None:
            feature_columns.append(has_gas_prev_feature.astype(np.float32))

    if use_env_features and env_feature_set:
        is_satellite = (host_index >= 0).astype(np.float32)
        time_since_infall = np.zeros_like(halo_mass, dtype=float)
        if "time_since_infall" in env_feature_set and env_state is not None:
            first_host_time = env_state.setdefault("first_host_time", {})
            current_time = float(curr_layer.get("time", 0.0))
            for idx, node in enumerate(curr_layer.get("nodes", [])):
                node_id = node.get("id")
                if node_id is None:
                    continue
                if host_index[idx] >= 0:
                    if node_id not in first_host_time:
                        first_host_time[node_id] = current_time
                    time_since_infall[idx] = max(current_time - first_host_time[node_id], 0.0)
        host_mass_log10 = safe_log10(host_mass, mass_log_eps)
        for feat in env_feature_set:
            if feat == "is_satellite":
                feature_columns.append(is_satellite)
            elif feat == "host_mass":
                feature_columns.append(host_mass_log10)
            elif feat == "time_since_infall":
                feature_columns.append(time_since_infall)

    redshift = layer_redshift(curr_layer) or 0.0
    positions = np.nan_to_num(extract_field_array(curr_layer, "halo_pos"), nan=0.0)
    velocities = None
    if use_spatial_features:
        velocities = np.nan_to_num(extract_field_array(curr_layer, "halo_vel"), nan=0.0)
        host_dist = np.zeros(len(curr_layer["nodes"]), dtype=float)
        host_rel_speed = np.zeros(len(curr_layer["nodes"]), dtype=float)
        for idx, h_idx in enumerate(host_index):
            if h_idx >= 0 and h_idx < positions.shape[0]:
                delta_pos = positions[h_idx] - positions[idx]
                host_dist[idx] = float(np.linalg.norm(delta_pos))
                if velocities is not None and velocities.shape[1] >= 3:
                    delta_vel = velocities[h_idx] - velocities[idx]
                    host_rel_speed[idx] = float(np.linalg.norm(delta_vel))
        feature_columns.append(np.log1p(host_dist))
        feature_columns.append(host_rel_speed)
    feature_columns.append(np.full(len(curr_layer["nodes"]), redshift))
    feature_columns.append(np.full(len(curr_layer["nodes"]), dt))

    feature_mat = np.column_stack(feature_columns) if feature_columns else np.zeros((len(curr_layer["nodes"]), 0))

    return LayerPairFeatures(
        node_features=feature_mat.astype(np.float32, copy=False),
        node_targets=targets.astype(np.float32, copy=False),
        node_mask=mask.astype(np.float32, copy=False),
        halo_mass=halo_mass.astype(np.float32, copy=False),
        transported_state=transported,
        curr_state=curr_state,
        dt=dt,
        redshift=float(redshift),
        host_index=host_index.astype(np.int64, copy=False),
        env_features=np.column_stack(env_cols).astype(np.float32, copy=False) if env_cols else np.zeros((len(curr_layer["nodes"]), 0), dtype=np.float32),
        positions=positions.astype(np.float32, copy=False),
        velocities=None if velocities is None else velocities.astype(np.float32, copy=False),
        new_node_mask=new_node_mask.astype(bool, copy=False),
    )
