"""Dataset and batching utilities for GPLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from gplm.features import LayerPairFeatures, build_edge_index, build_layer_pair_features, safe_log
from jf_graph import LayeredGraph
from jf_utils import canonical_field_name
from jf_time import ensure_layer_times_physical


@dataclass
class GraphEFTBatch:
    """Mini-batch container wrapping a stacked PyG Data object."""

    pyg_data: Data
    field_names: Sequence[str]
    dt: torch.Tensor
    redshift: torch.Tensor
    mask: torch.Tensor
    halo_mass: torch.Tensor
    snap: torch.Tensor


class GraphEFTDataset(Dataset):
    """Turn layered halo graphs into temporal edge datasets for GPLM."""

    def __init__(
        self,
        graph_paths: Sequence[str | Path],
        fields: Sequence[str],
        extra_features: Sequence[str] | None = None,
        use_spatial_features: bool = False,
        include_first_star: bool = True,
        env_feature_names: Sequence[str] | None = None,
        use_env_features: bool = False,
        ablate_host_edges: bool = False,
        supplementary_fields: Sequence[str] | None = None,
        mass_log_eps: float = 0.05,
        include_history_flags: bool = True,
    ):
        self.graph_paths = [Path(p) for p in graph_paths]
        self.fields = list(fields)
        self.field_canonical = [canonical_field_name(name) for name in self.fields]
        self.extra_features = list(extra_features or [])
        self.use_spatial_features = use_spatial_features
        self.include_first_star = include_first_star
        self.env_feature_names = list(env_feature_names or [])
        self.use_env_features = use_env_features
        self.ablate_host_edges = ablate_host_edges
        self.mass_log_eps = mass_log_eps
        self.supplementary_fields = list(supplementary_fields or [])
        self.include_history_flags = include_history_flags
        supplementary_canon = {canonical_field_name(name) for name in self.supplementary_fields}
        unknown = supplementary_canon - set(self.field_canonical)
        if unknown:
            raise ValueError(f"Supplementary fields {sorted(unknown)} are not present in the target field list {self.fields}.")
        self.supplementary_indices = [idx for idx, canon in enumerate(self.field_canonical) if canon in supplementary_canon]
        self.layer_pairs: List[Dict[str, object]] = []
        self.snap_values: List[int] = []

        for path in self.graph_paths:
            layered = LayeredGraph(str(path))
            ensure_layer_times_physical(layered.layers)
            env_state: Dict[str, Dict[int, float]] = {"first_host_time": {}}
            for idx in range(1, len(layered.layers)):
                prev_layer = layered.layers[idx - 1]
                curr_layer = layered.layers[idx]
                pair = build_layer_pair_features(
                    prev_layer,
                    curr_layer,
                    self.fields,
                    extra_node_features=self.extra_features,
                    use_spatial_features=self.use_spatial_features,
                    include_first_star=self.include_first_star,
                    env_feature_names=self.env_feature_names,
                    use_env_features=self.use_env_features,
                    env_state=env_state,
                    mass_log_eps=self.mass_log_eps,
                    include_history_flags=self.include_history_flags,
                )
                self._apply_supplementary_mask(pair)

                # The model operates on the current-layer node state only. Temporal
                # transport from the previous layer is already encoded in the node
                # features via build_T(...), so temporal edges should live in the
                # current-layer index space rather than mixing prev/curr indices.
                temporal_edges: List[tuple[int, int]] = [
                    (idx_curr, idx_curr) for idx_curr, is_new in enumerate(pair.new_node_mask) if not is_new
                ]
                temporal_attr = (
                    np.full((len(temporal_edges), 1), pair.dt, dtype=np.float32) if temporal_edges else np.zeros((0, 1), dtype=np.float32)
                )

                host_edges: List[tuple[int, int]] = []
                host_attr_rows: List[List[float]] = []
                for src, tgt in curr_layer["host_edges"]:
                    if src in curr_layer["id2idx"] and tgt in curr_layer["id2idx"]:
                        child = curr_layer["id2idx"][src]
                        host = curr_layer["id2idx"][tgt]
                        host_edges.append((child, host))
                        host_mass = pair.curr_state.get("M_halo", np.zeros(len(curr_layer["nodes"]), dtype=float))
                        host_attr_rows.append(
                            [
                                safe_log(np.array([host_mass[host]]), self.mass_log_eps)[0],
                                safe_log(np.array([host_mass[child]]), self.mass_log_eps)[0],
                                pair.dt,
                            ]
                        )
                host_attr = np.array(host_attr_rows, dtype=np.float32) if host_attr_rows else np.zeros((0, 3), dtype=np.float32)
                if self.ablate_host_edges:
                    host_edges = []
                    host_attr = np.zeros((0, 3), dtype=np.float32)

                snap_curr = curr_layer.get("snap")
                if snap_curr is not None:
                    self.snap_values.append(int(snap_curr))

                self.layer_pairs.append(
                    {
                        "path": path,
                        "snap_prev": prev_layer.get("snap"),
                        "snap_curr": snap_curr,
                        "features": pair,
                        "temporal_edges": temporal_edges,
                        "host_edges": host_edges,
                        "temporal_attr": temporal_attr,
                        "host_attr": host_attr,
                    }
                )

    def _apply_supplementary_mask(self, features: LayerPairFeatures) -> None:
        if not self.supplementary_indices:
            return
        if features.node_mask.size == 0:
            return
        features.node_mask[:, self.supplementary_indices] = 0.0

    def __len__(self) -> int:
        return len(self.layer_pairs)

    def __getitem__(self, index: int) -> GraphEFTBatch:
        payload = self.layer_pairs[index]
        features: LayerPairFeatures = payload["features"]  # type: ignore[assignment]
        x = torch.from_numpy(features.node_features).float()
        y = torch.from_numpy(features.node_targets).float()
        mask = torch.from_numpy(features.node_mask).float()
        dt = torch.full((x.size(0),), float(features.dt), dtype=torch.float32)
        redshift = torch.full((x.size(0),), float(features.redshift), dtype=torch.float32)

        temporal_index = build_edge_index(payload["temporal_edges"]).long()
        host_index = build_edge_index(payload["host_edges"]).long()
        temporal_attr = torch.from_numpy(payload["temporal_attr"]).float()
        host_attr = torch.from_numpy(payload["host_attr"]).float()
        halo_mass = torch.from_numpy(features.halo_mass).float()
        snap_val = payload.get("snap_curr")
        snap_tensor = torch.full((x.size(0),), float(snap_val) if snap_val is not None else -1.0, dtype=torch.float32)

        data = Data(
            x=x,
            y=y,
            mask=mask,
            dt=dt,
            redshift=redshift,
            edge_index_temporal=temporal_index,
            edge_index_host=host_index,
            edge_attr_temporal=temporal_attr,
            edge_attr_host=host_attr,
            snap=snap_tensor,
        )
        return GraphEFTBatch(
            pyg_data=data,
            field_names=self.fields,
            dt=dt,
            redshift=redshift,
            mask=mask,
            halo_mass=halo_mass,
            snap=snap_tensor,
        )


def collate_fn(batch: List[GraphEFTBatch]) -> GraphEFTBatch:
    """Collate list of GraphEFTBatch into one PyG data batch."""
    data_list = [item.pyg_data for item in batch]
    stacked = Batch.from_data_list(data_list)
    return GraphEFTBatch(
        pyg_data=stacked,
        field_names=batch[0].field_names,
        dt=torch.cat([item.dt for item in batch], dim=0),
        redshift=torch.cat([item.redshift for item in batch], dim=0),
        mask=torch.cat([item.mask for item in batch], dim=0),
        halo_mass=torch.cat([item.halo_mass for item in batch], dim=0),
        snap=torch.cat([item.snap for item in batch], dim=0),
    )
