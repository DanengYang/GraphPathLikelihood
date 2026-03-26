"""Deterministic transport-only baseline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from jf_extractors import extract_field_array
from jf_graph import LayeredGraph
from jf_operators import build_T
from jf_time import ensure_layer_times_physical


@dataclass
class TransportOnlyConfig:
    """Placeholder config to keep CLI/API explicit."""

    mode: str = "transport_only"


def generate_transport_predictions(
    graph: LayeredGraph,
    fields: Sequence[str],
    config: TransportOnlyConfig | None = None,
) -> Dict[str, List[np.ndarray]]:
    """Generate per-layer transport-only predictions.

    The update is deterministic:
    - existing nodes: temporal transport from previous-layer predictions
    - newly attached nodes: seeded with the current-layer truth values
    """

    if config is not None and config.mode != "transport_only":
        raise ValueError("Only transport_only mode is supported in the github package.")

    ensure_layer_times_physical(graph.layers)
    result: Dict[str, List[np.ndarray]] = {field: [] for field in fields}
    if not graph.layers:
        return result

    first = graph.layers[0]
    for field in fields:
        init = np.nan_to_num(extract_field_array(first, field), nan=0.0).astype(np.float64, copy=False)
        result[field].append(np.clip(init, 0.0, None))

    for idx in range(1, len(graph.layers)):
        prev = graph.layers[idx - 1]
        curr = graph.layers[idx]
        T = build_T(prev, curr)
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        new_mask = row_sums == 0.0
        for field in fields:
            prev_pred = result[field][-1]
            curr_truth = np.nan_to_num(extract_field_array(curr, field), nan=0.0).astype(np.float64, copy=False)
            curr_pred = np.asarray(T @ prev_pred, dtype=np.float64)
            if np.any(new_mask):
                curr_pred = curr_pred.copy()
                curr_pred[new_mask] = curr_truth[new_mask]
            result[field].append(np.clip(curr_pred, 0.0, None))
    return result
