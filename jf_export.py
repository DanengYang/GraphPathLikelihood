"""Export helpers for graph-EFT predictions."""

from __future__ import annotations

import copy
import json
from typing import Dict, List

import numpy as np

from jf_graph import LayeredGraph

# --- helper: also emit linear rhalf_* when we see logrhalf_* ---
def _maybe_emit_linear_rhalf(container: dict, field: str, value: float) -> None:
    """
    If 'field' is logrhalf_* and value is finite, also write linear rhalf_* into the same dict.
    This lets viz scripts that expect rhalf_* (linear) work even if the model evolves logs.
    """
    if not isinstance(field, str):
        return
    if not field.startswith("logrhalf_"):
        return
    lin_name = field.replace("log", "", 1)  # logrhalf_gas -> rhalf_gas
    if np.isfinite(value):
        try:
            container[lin_name] = float(np.exp(value))
            return
        except Exception:
            pass
    # fall back to explicit None if not finite / conversion failed
    container[lin_name] = None

def export_predictions(
    graph: LayeredGraph,
    fields: List[str],
    predictions: Dict[str, List[np.ndarray]],
    out_path: str,
) -> None:
    """Write painted fields back to a JSON matching the input schema."""
    data = copy.deepcopy(graph.raw)

    if graph.schema == "nodes_links" and isinstance(data, dict) and "nodes" in data:
        id_to_layer = {}
        for layer_idx, layer in enumerate(graph.layers):
            for node_idx, node in enumerate(layer["nodes"]):
                id_to_layer[node["id"]] = (layer_idx, node_idx)

        for node in data["nodes"]:
            node_id = node.get("id")
            if node_id not in id_to_layer:
                continue
            layer_idx, node_idx = id_to_layer[node_id]
            weights = dict(node.get("weights", {}))
            for field in fields:
                arr = predictions.get(field, None)
                if arr is None:
                    continue
                value = arr[layer_idx][node_idx] if node_idx < len(arr[layer_idx]) else np.nan
                if np.isfinite(value):
                    weights[field] = float(value)
                    _maybe_emit_linear_rhalf(weights, field, value)
            # keep SubhaloMassType in sync for key baryonic components
            mass_type = weights.get("SubhaloMassType")
            if isinstance(mass_type, list):
                mt = list(mass_type)
                if "M_gas" in fields:
                    if len(mt) < 1:
                        mt.extend([0.0] * (1 - len(mt)))
                    val = weights.get("M_gas")
                    if val is not None:
                        mt[0] = val
                if "M_star" in fields:
                    if len(mt) < 5:
                        mt.extend([0.0] * (5 - len(mt)))
                    val = weights.get("M_star")
                    if val is not None:
                        mt[4] = val
                if "M_bh" in fields:
                    if len(mt) < 6:
                        mt.extend([0.0] * (6 - len(mt)))
                    val = weights.get("M_bh")
                    if val is not None:
                        mt[5] = val
                weights["SubhaloMassType"] = mt
            node["weights"] = weights
    else:
        if isinstance(data, dict) and isinstance(data.get("layers"), list):
            target_layers = data["layers"]
        elif isinstance(data, list):
            target_layers = data
        else:
            target_layers = None

        if target_layers is not None:
            for layer_idx, layer in enumerate(graph.layers):
                if layer_idx >= len(target_layers):
                    break
                nodes = target_layers[layer_idx].get("nodes", target_layers[layer_idx].get("vertices", []))
                for node_idx, node in enumerate(nodes):
                    props = dict(node.get("props", {}))
                    for field in fields:
                        arr = predictions.get(field, None)
                        if arr is None or layer_idx >= len(arr):
                            continue
                        values = arr[layer_idx]
                        if node_idx >= len(values):
                            continue
                        value = values[node_idx]
                        props[field] = float(value) if np.isfinite(value) else None
                        _maybe_emit_linear_rhalf(props, field, value)
                    # maintain SubhaloMassType if present in props
                    mass_type = props.get("SubhaloMassType")
                    if isinstance(mass_type, list):
                        mt = list(mass_type)
                        if "M_gas" in fields:
                            if len(mt) < 1:
                                mt.extend([0.0] * (1 - len(mt)))
                            val = props.get("M_gas")
                            if val is not None:
                                mt[0] = val
                        if "M_star" in fields:
                            if len(mt) < 5:
                                mt.extend([0.0] * (5 - len(mt)))
                            val = props.get("M_star")
                            if val is not None:
                                mt[4] = val
                        if "M_bh" in fields:
                            if len(mt) < 6:
                                mt.extend([0.0] * (6 - len(mt)))
                            val = props.get("M_bh")
                            if val is not None:
                                mt[5] = val
                        props["SubhaloMassType"] = mt
                    node["props"] = props
        else:
            clean = {"meta": graph.meta, "layers": []}
            for layer_idx, layer in enumerate(graph.layers):
                nodes_out = []
                for node_idx, node in enumerate(layer["nodes"]):
                    props = {}
                    for field in fields:
                        arr = predictions.get(field, None)
                        if arr is None:
                            continue
                        value = arr[layer_idx][node_idx]
                        props[field] = float(value) if np.isfinite(value) else None
                        _maybe_emit_linear_rhalf(props, field, value)
                    nodes_out.append({"id": node["id"], "props": props})
                clean["layers"].append(
                    {"snap": layer.get("snap"), "time": layer.get("time"), "nodes": nodes_out, "edges": []}
                )
            data = clean

    with open(out_path, "w") as handle:
        json.dump(data, handle, indent=2)


def export_sigmas(
    graph: LayeredGraph,
    fields: List[str],
    sigmas: Dict[str, List[np.ndarray]],
    out_path: str,
) -> None:
    """Export per-field uncertainties mirroring the graph structure."""
    data = copy.deepcopy(graph.raw)

    if graph.schema == "nodes_links" and isinstance(data, dict) and "nodes" in data:
        id_to_layer = {}
        for layer_idx, layer in enumerate(graph.layers):
            for node_idx, node in enumerate(layer["nodes"]):
                id_to_layer[node["id"]] = (layer_idx, node_idx)
        for node in data["nodes"]:
            node_id = node.get("id")
            if node_id not in id_to_layer:
                continue
            layer_idx, node_idx = id_to_layer[node_id]
            weights = dict(node.get("weights", {}))
            for field in fields:
                arr = sigmas.get(field, None)
                if arr is None:
                    continue
                value = arr[layer_idx][node_idx] if node_idx < len(arr[layer_idx]) else np.nan
                if np.isfinite(value):
                    weights[f"{field}_sigma"] = float(value)
            node["weights"] = weights
    else:
        if isinstance(data, dict) and isinstance(data.get("layers"), list):
            target_layers = data["layers"]
        elif isinstance(data, list):
            target_layers = data
        else:
            target_layers = None
        if target_layers is not None:
            for layer_idx, layer in enumerate(graph.layers):
                if layer_idx >= len(target_layers):
                    break
                nodes = target_layers[layer_idx].get("nodes", target_layers[layer_idx].get("vertices", []))
                for node_idx, node in enumerate(nodes):
                    props = dict(node.get("props", {}))
                    for field in fields:
                        arr = sigmas.get(field, None)
                        if arr is None or layer_idx >= len(arr):
                            continue
                        values = arr[layer_idx]
                        if node_idx >= len(values):
                            continue
                        value = values[node_idx]
                        if np.isfinite(value):
                            props[f"{field}_sigma"] = float(value)
                    node["props"] = props
        else:
            clean = {"meta": graph.meta, "layers": []}
            for layer_idx, layer in enumerate(graph.layers):
                nodes_out = []
                for node_idx, node in enumerate(layer["nodes"]):
                    props = {}
                    for field in fields:
                        arr = sigmas.get(field, None)
                        if arr is None:
                            continue
                        value = arr[layer_idx][node_idx]
                        if np.isfinite(value):
                            props[f"{field}_sigma"] = float(value)
                    nodes_out.append({"id": node["id"], "props": props})
                clean["layers"].append(
                    {"snap": layer.get("snap"), "time": layer.get("time"), "nodes": nodes_out, "edges": []}
                )
            data = clean

    with open(out_path, "w") as handle:
        json.dump(data, handle, indent=2)


__all__ = ["export_predictions", "export_sigmas"]
