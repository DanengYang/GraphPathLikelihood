"""Layered graph loading and normalization utilities."""

import json

from jf_constants import HOST_EDGE_KEYS, TEMPORAL_EDGE_KEYS
from jf_time import ensure_layer_times_physical


def _ensure_id2idx(layer):
    if "id2idx" in layer and isinstance(layer["id2idx"], dict):
        return
    node_ids = [node.get("id") for node in layer.get("nodes", [])]
    layer["id2idx"] = {nid: idx for idx, nid in enumerate(node_ids)}


def _normalize_time_edges(layers):
    """Normalize raw time-edge storage to earlier->later edges on the later layer.

    This supports both legacy graphs that store time edges as later->earlier and
    newer graphs that store them as earlier->later.
    """
    for layer in layers:
        _ensure_id2idx(layer)
        layer.setdefault("time_edges", [])

    node_to_snap = {}
    for layer in layers:
        snap = layer.get("snap")
        if snap is None:
            continue
        for node in layer.get("nodes", []):
            node_id = node.get("id")
            if node_id is not None:
                node_to_snap[node_id] = int(snap)

    normalized = {int(layer.get("snap")): [] for layer in layers if layer.get("snap") is not None}
    for layer in layers:
        for edge in layer.get("time_edges", []) or []:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            a, b = edge
            snap_a = node_to_snap.get(a)
            snap_b = node_to_snap.get(b)
            if snap_a is None or snap_b is None or snap_a == snap_b:
                continue
            if snap_a < snap_b:
                normalized.setdefault(snap_b, []).append((a, b))
            else:
                normalized.setdefault(snap_a, []).append((b, a))

    for layer in layers:
        snap = layer.get("snap")
        if snap is None:
            layer["time_edges"] = []
            continue
        layer["time_edges"] = normalized.get(int(snap), [])


def guess_schema_and_layers(data):
    """Normalize supported graph schemas into the layered representation."""
    if isinstance(data, dict) and isinstance(data.get("layers"), list):
        return data.get("meta", {}), data["layers"], "layers"

    if isinstance(data, dict) and isinstance(data.get("nodes"), list) and isinstance(data.get("links"), list):
        nodes = data["nodes"]
        links = data["links"]
        snaps = sorted({int(node.get("layer")) for node in nodes})
        layers = []
        for snap in snaps:
            nodes_here = [node for node in nodes if int(node.get("layer")) == snap]
            host_edges, time_edges = [], []
            for edge in links:
                kind = str(edge.get("kind", "")).lower()
                if kind in HOST_EDGE_KEYS and int(edge.get("layer_from")) == snap and int(edge.get("layer_to")) == snap:
                    host_edges.append((edge.get("source"), edge.get("target")))
            for edge in links:
                kind = str(edge.get("kind", "")).lower()
                if kind in TEMPORAL_EDGE_KEYS:
                    layer_from = int(edge.get("layer_from"))
                    layer_to = int(edge.get("layer_to"))
                    if layer_from == snap and layer_to < layer_from:
                        time_edges.append((edge.get("target"), edge.get("source")))
                    if layer_to == snap and layer_from < layer_to:
                        time_edges.append((edge.get("source"), edge.get("target")))
            node_ids = [node.get("id") for node in nodes_here]
            id2idx = {nid: idx for idx, nid in enumerate(node_ids)}
            normalized_nodes = []
            for node in nodes_here:
                props = dict(node.get("weights", {}))
                if "pos_ckpch" in node:
                    props["pos_ckpch"] = node["pos_ckpch"]
                if "is_central" in node:
                    props["is_central"] = node["is_central"]
                if "in_radius" in node:
                    props["in_radius"] = node["in_radius"]
                normalized_nodes.append({"id": node.get("id"), "props": props, "weights_ref": True})
            layers.append(
                {
                    "snap": snap,
                    "time": None,
                    "nodes": normalized_nodes,
                    "id2idx": id2idx,
                    "host_edges": host_edges,
                    "time_edges": time_edges,
                }
            )

        meta = data.get("meta", {})
        box = meta.get("boxsize_ckpch") or meta.get("box_size") or meta.get("BoxSize")
        if box is not None:
            try:
                meta["box_size"] = float(box)
            except Exception:
                pass
        return meta, layers, "nodes_links"

    return {}, [], "unknown"


class LayeredGraph:
    """Load and expose layered graph data from JSON."""

    def __init__(self, path):
        with open(path, "r") as handle:
            data = json.load(handle)
        meta, layers, schema = guess_schema_and_layers(data)
        self.meta = meta if isinstance(meta, dict) else {}
        self.box_size = float(self.meta.get("box_size")) if "box_size" in self.meta else None
        layers.sort(key=lambda layer: int(layer.get("snap") if layer.get("snap") is not None else 0))
        _normalize_time_edges(layers)
        self.layers = layers
        try:
            ensure_layer_times_physical(self.layers, overwrite=False)
        except Exception:
            pass
        self.raw = data
        self.schema = schema


__all__ = ["guess_schema_and_layers", "LayeredGraph"]
