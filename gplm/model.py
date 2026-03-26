"""Graph Path Likelihood Model (GPLM) definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


def make_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "silu",
    dropout: float | None = None,
) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(nn.SiLU() if activation == "silu" else nn.ReLU())
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.LayerNorm(h))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


@dataclass
class GraphEFTConfig:
    in_dim: int
    hidden_dim: int = 128
    message_layers: int = 3
    heads: int = 4
    edge_dim: int = 4
    drift_output_dim: int = 3
    diffusion_output_dim: int = 3
    dropout: float = 0.0
    layer_norm: bool = True
    residual: bool = True
    full_diffusion: bool = False
    field_names: Optional[Sequence[str]] = None
    use_host_conv: bool = True

    mlp_hidden_dim: int | None = None
    mlp_layers: int | None = None
    mlp_dropout: float | None = None


class GraphEFTModel(nn.Module):
    """Produces drift and diffusion predictions for GPLM."""

    def __init__(self, config: GraphEFTConfig):
        super().__init__()
        self.config = config

        self.temporal_edge_encoder = nn.Linear(1, config.edge_dim)
        self.use_host_conv = config.use_host_conv
        self.host_edge_encoder = nn.Linear(3, config.edge_dim) if self.use_host_conv else None

        self.temporal_convs = nn.ModuleList()
        self.host_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = config.in_dim
        for _ in range(config.message_layers):
            conv_temp = TransformerConv(
                in_channels=in_dim,
                out_channels=config.hidden_dim // config.heads,
                heads=config.heads,
                edge_dim=config.edge_dim,
                dropout=config.dropout,
                beta=False,
                root_weight=True,
            )
            if self.use_host_conv:
                conv_host = TransformerConv(
                    in_channels=in_dim,
                    out_channels=config.hidden_dim // config.heads,
                    heads=config.heads,
                    edge_dim=config.edge_dim,
                    dropout=config.dropout,
                    beta=False,
                    root_weight=True,
                )
            else:
                conv_host = None
            self.temporal_convs.append(conv_temp)
            self.host_convs.append(conv_host)
            self.norms.append(nn.LayerNorm(config.hidden_dim))
            in_dim = config.hidden_dim
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        drift_hidden = config.mlp_hidden_dim or config.hidden_dim
        drift_layers = config.mlp_layers or 1
        drift_dims = [drift_hidden] * drift_layers
        self.drift_head = make_mlp(config.hidden_dim, drift_dims, config.drift_output_dim, dropout=config.mlp_dropout)

        diff_hidden = config.mlp_hidden_dim or config.hidden_dim
        diff_layers = config.mlp_layers or 1
        diff_dims = [diff_hidden] * diff_layers
        if config.full_diffusion:
            if not config.field_names:
                raise ValueError("full_diffusion requires field_names in config")
            n_fields = len(config.field_names)
            diff_out_dim = n_fields * (n_fields + 1) // 2
        else:
            diff_out_dim = config.diffusion_output_dim
            n_fields = diff_out_dim
        self.diff_head = make_mlp(config.hidden_dim, diff_dims, diff_out_dim, dropout=config.mlp_dropout)
        self.full_diffusion = config.full_diffusion
        self.n_fields = n_fields

    def forward(
        self,
        x: torch.Tensor,
        edge_index_temporal: torch.Tensor,
        edge_index_host: torch.Tensor,
        edge_attr_temporal: Optional[torch.Tensor] = None,
        edge_attr_host: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        if edge_attr_temporal is None:
            temporal_attr = torch.ones((edge_index_temporal.size(1), 1), device=x.device)
        else:
            temporal_attr = edge_attr_temporal
        temporal_emb = self.temporal_edge_encoder(temporal_attr)
        host_emb = None
        if self.use_host_conv and self.host_edge_encoder is not None and edge_index_host.size(1) > 0:
            if edge_attr_host is None:
                host_attr = torch.zeros((edge_index_host.size(1), 3), device=x.device)
            else:
                host_attr = edge_attr_host
            host_emb = self.host_edge_encoder(host_attr)

        for conv_temp, conv_host, norm in zip(self.temporal_convs, self.host_convs, self.norms):
            temp_out = conv_temp(h, edge_index_temporal, temporal_emb)
            if self.use_host_conv and conv_host is not None and host_emb is not None and edge_index_host.size(1) > 0:
                host_out = conv_host(h, edge_index_host, host_emb)
            else:
                host_out = torch.zeros_like(temp_out)
            combined = temp_out + host_out
            combined = norm(combined)
            combined = torch.nn.functional.silu(combined)
            if self.config.residual and combined.shape == h.shape:
                h = h + combined
            else:
                h = combined

        h = self.final_norm(h)
        drift = self.drift_head(h)
        diff_out = self.diff_head(h)
        if not self.full_diffusion:
            return drift, diff_out

        chol = self._build_cholesky(diff_out)
        return drift, chol

    def _build_cholesky(self, flat_factors: torch.Tensor) -> torch.Tensor:
        batch = flat_factors.size(0)
        n = self.n_fields
        tril = flat_factors.new_zeros((batch, n, n))
        idx = 0
        for row in range(n):
            for col in range(row + 1):
                value = flat_factors[:, idx]
                if row == col:
                    value = torch.exp(value)
                tril[:, row, col] = value
                idx += 1
        return tril
