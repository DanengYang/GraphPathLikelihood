"""Export helpers for GPLM outputs."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from jf_constants import BACKGROUND_FIELDS
from jf_export import export_predictions, export_sigmas
from jf_extractors import extract_field_array
from jf_graph import LayeredGraph


def export_all(
    graph: LayeredGraph,
    fields: Sequence[str],
    preds: Dict[str, List[np.ndarray]],
    sigmas: Dict[str, List[np.ndarray]] | None,
    out_pred: str,
    out_sigma: str | None = None,
) -> None:
    """Write predictions (and optional sigmas) to disk, copying background fields."""
    pred_fields = list(fields)
    for field in fields:
        if field in BACKGROUND_FIELDS:
            preds[field] = [extract_field_array(layer, field) for layer in graph.layers]
            if sigmas is not None:
                sigmas[field] = [np.full_like(arr, np.nan) for arr in preds[field]]
    export_predictions(graph, pred_fields, preds, out_pred)
    if sigmas is not None and out_sigma is not None:
        export_sigmas(graph, pred_fields, sigmas, out_sigma)


__all__ = ["export_all"]
