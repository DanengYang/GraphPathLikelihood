"""Inference utilities for GPLM."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from gplm.data import GraphEFTDataset
from gplm.features import LOG_TARGET_FIELDS_CANON
from gplm.model import GraphEFTModel, GraphEFTConfig
from jf_graph import LayeredGraph
from jf_extractors import extract_field_array
from jf_utils import canonical_field_name


def load_model(path: str | Path, config: GraphEFTConfig, device: torch.device) -> GraphEFTModel:
    model = GraphEFTModel(config)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model



def euler_step(
    transported: Dict[str, np.ndarray],
    field_order: Sequence[str],
    delta: torch.Tensor,
    mass_log_eps: float = 0.05,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    delta_np = delta.detach().cpu().numpy()
    offset_linear = max(mass_log_eps, 1e-12)
    for idx, field in enumerate(field_order):
        base = transported[field]
        canon = canonical_field_name(field)
        if canon in LOG_TARGET_FIELDS_CANON:
            base_clamped = np.clip(base, 0.0, None)
            log_prev = np.log(base_clamped + offset_linear)
            log_pred = log_prev + delta_np[:, idx]
            out[field] = np.clip(np.exp(log_pred) - offset_linear, 0.0, None)
        else:
            out[field] = base + delta_np[:, idx]
    return out


def apply_model_to_graph(
    graph_path: str | Path,
    model: GraphEFTModel,
    fields: Sequence[str],
    device: torch.device,
    use_spatial_features: bool = False,
    include_first_star: bool = True,
    env_feature_names: Sequence[str] | None = None,
    use_env_features: bool = True,
    ablate_host_edges: bool = False,
    sigma_floor: float = 1e-3,
    extra_features: Sequence[str] | None = None,
    mass_log_eps: float = 0.05,
    residual_scales: Dict[str, float] | None = None,
) -> tuple[Dict[str, Sequence[np.ndarray]], Dict[str, Sequence[np.ndarray]]]:
    graph = LayeredGraph(str(graph_path))
    predictions: Dict[str, Sequence[np.ndarray]] = {field: [] for field in fields}
    sigmas: Dict[str, Sequence[np.ndarray]] = {field: [] for field in fields}
    if not graph.layers:
        return predictions, sigmas

    dataset = GraphEFTDataset(
        [graph_path],
        fields,
        extra_features=extra_features,
        use_spatial_features=use_spatial_features,
        include_first_star=include_first_star,
        env_feature_names=env_feature_names,
        use_env_features=use_env_features,
        ablate_host_edges=ablate_host_edges,
        mass_log_eps=mass_log_eps,
    )
    first_layer = graph.layers[0]
    for field in fields:
        initial = extract_field_array(first_layer, field)
        predictions[field].append(np.nan_to_num(initial, nan=0.0))
        sigmas[field].append(np.full_like(initial, np.nan, dtype=float))

    scale_tensor: torch.Tensor | None = None
    log_scale: torch.Tensor | None = None
    scale_vec = None
    if residual_scales:
        lookup = {canonical_field_name(name): float(value) for name, value in residual_scales.items()}
        scale_vec = np.array([lookup.get(canonical_field_name(field), 1.0) for field in fields], dtype=np.float32)
    for idx in range(1, len(graph.layers)):
        batch = dataset[idx - 1]
        data = batch.pyg_data.to(device)
        drift_rate, diff_out = model(
            data.x,
            data.edge_index_temporal,
            data.edge_index_host,
            getattr(data, "edge_attr_temporal", None),
            getattr(data, "edge_attr_host", None),
        )
        if scale_vec is not None:
            if scale_tensor is None or scale_tensor.device != drift_rate.device:
                scale_tensor = torch.from_numpy(scale_vec).to(drift_rate.device).view(1, -1)
                log_scale = torch.log(scale_tensor.clamp(min=1e-8))
            drift_rate = drift_rate * scale_tensor
            if model.config.full_diffusion:
                diff_out = diff_out * scale_tensor.view(1, -1, 1)
            else:
                diff_out = diff_out + 2.0 * log_scale
        pair = dataset.layer_pairs[idx - 1]  # type: ignore[index]
        features = pair["features"]
        transported = features.transported_state  # type: ignore[attr-defined]
        delta_total = drift_rate * float(features.dt)
        updated = euler_step(
            transported,
            fields,
            delta_total,
            mass_log_eps=mass_log_eps,
        )
        sigma_arrays: Dict[str, np.ndarray] = {}
        if not model.config.full_diffusion:
            # Diffusion reported in rate space: D (x^2 / t). Per-step increment variance: D * dt
            sigma2 = torch.exp(diff_out).detach() * float(features.dt)
            sigma2 = torch.clamp(sigma2, min=sigma_floor**2)
            sigma_np = torch.sqrt(sigma2).cpu().numpy()
            for field_idx, field in enumerate(fields):
                sigma_arrays[field] = sigma_np[:, field_idx]
        else:
            chol = diff_out.detach()
            sqrt_dt = math.sqrt(float(features.dt))
            chol_step = chol * sqrt_dt
            variances = torch.sum(chol_step * chol_step, dim=2)
            variances = torch.clamp(variances, min=sigma_floor**2)
            sigma_np = torch.sqrt(variances).cpu().numpy()
            for field_idx, field in enumerate(fields):
                sigma_arrays[field] = sigma_np[:, field_idx]
        new_mask = getattr(features, "new_node_mask", None)
        if new_mask is not None and np.any(new_mask):
            for field in fields:
                truth_curr = features.curr_state.get(field)
                if truth_curr is not None:
                    updated[field][new_mask] = truth_curr[new_mask]
                if field in sigma_arrays:
                    sigma_arrays[field][new_mask] = np.nan
        for field in fields:
            predictions[field].append(updated[field])
            if field in sigma_arrays:
                sigmas[field].append(sigma_arrays[field])
            else:
                sigmas[field].append(np.full_like(updated[field], np.nan, dtype=float))
    return predictions, sigmas
