"""Training loop for GPLM."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gplm.data import GraphEFTDataset, GraphEFTBatch, collate_fn
from gplm.loss import LossConfig, onsager_machlup_loss
from gplm.model import GraphEFTConfig, GraphEFTModel
from gplm.features import LOG_TARGET_FIELDS_CANON
from jf_utils import canonical_field_name


@dataclass
class TrainerConfig:
    fields: Sequence[str]
    extra_features: Sequence[str]
    class_weights: Dict[str, float] | None = None
    mass_reference: float = 1.0
    mass_exponent: float = 0.0
    mass_log_eps: float = 0.05
    use_spatial_features: bool = False
    include_first_star_transitions: bool = True
    use_env_features: bool = True
    env_features: Sequence[str] = field(default_factory=lambda: ("is_satellite", "host_mass"))
    ablate_host_edges: bool = False
    supplementary_fields: Sequence[str] = field(default_factory=lambda: ("SubhaloSFR",))
    residual_scales: Dict[str, float] = field(default_factory=dict)
    residual_reg_weight: float = 0.0
    residual_reg_power: float = 1.0
    residual_reg_z_ref: float = 2.0
    layer_weights: Dict[int, float] = field(default_factory=dict)
    snap_weight_power: float = 0.0
    two_stage: bool = False
    train_stage: str = "full"
    stageA_epochs: int = 0
    stageB_epochs: int = 0
    stageB_lr_multiplier: float = 0.2
    freeze_backbone_in_stageB_epochs: int = 0
    sigma_floor: float = 1e-3
    stageA_sigma: float = 1e-3
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    grad_clip: float = 1.0
    device: str = "auto"
    num_workers: int = 0
    loss: LossConfig = field(default_factory=LossConfig)
    model: GraphEFTConfig = field(default_factory=lambda: GraphEFTConfig(in_dim=6))
    scheduler: str = "none"


class GraphEFTTrainer:
    def __init__(self, config: TrainerConfig, train_paths: Sequence[str | Path], val_paths: Sequence[str | Path] | None = None):
        self.config = config
        self.field_canonical = [canonical_field_name(f) for f in config.fields]
        self.supplementary_canonical = {canonical_field_name(name) for name in config.supplementary_fields}
        unknown_supp = self.supplementary_canonical - set(self.field_canonical)
        if unknown_supp:
            raise ValueError(f"Supplementary fields {sorted(unknown_supp)} are not part of the training field list {config.fields}.")
        self.train_field_indices = [idx for idx, canon in enumerate(self.field_canonical) if canon not in self.supplementary_canonical]
        if not self.train_field_indices:
            raise ValueError("No trainable fields remain after excluding supplementary fields. Check --supplementary-fields.")
        self._active_index_tensor: torch.Tensor | None = None
        scale_lookup = {canonical_field_name(name): float(value) for name, value in config.residual_scales.items()}
        config.residual_scales = scale_lookup
        self.residual_scales = [scale_lookup.get(canon, 1.0) for canon in self.field_canonical]
        self._residual_scale_tensor: torch.Tensor | None = None
        self.layer_weights = {int(k): float(v) for k, v in config.layer_weights.items()}
        self.train_dataset = GraphEFTDataset(
            train_paths,
            config.fields,
            extra_features=config.extra_features,
            use_spatial_features=config.use_spatial_features,
            include_first_star=config.include_first_star_transitions,
            env_feature_names=config.env_features,
            use_env_features=config.use_env_features,
            ablate_host_edges=config.ablate_host_edges,
            supplementary_fields=config.supplementary_fields,
            mass_log_eps=config.mass_log_eps,
        )
        self.snap_min = min(self.train_dataset.snap_values) if self.train_dataset.snap_values else 0
        self.snap_max = max(self.train_dataset.snap_values) if self.train_dataset.snap_values else 1
        self.val_dataset = (
            GraphEFTDataset(
                val_paths or [],
                config.fields,
                extra_features=config.extra_features,
                use_spatial_features=config.use_spatial_features,
                include_first_star=config.include_first_star_transitions,
                env_feature_names=config.env_features,
                use_env_features=config.use_env_features,
                ablate_host_edges=config.ablate_host_edges,
                supplementary_fields=config.supplementary_fields,
                mass_log_eps=config.mass_log_eps,
            )
            if val_paths
            else None
        )

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        in_dim = self.train_dataset[0].pyg_data.x.size(-1)
        model_cfg = config.model
        model_cfg.in_dim = in_dim
        model_cfg.field_names = list(config.fields)
        model_cfg.drift_output_dim = len(config.fields)
        model_cfg.diffusion_output_dim = len(config.fields)
        config.loss.full_diffusion = model_cfg.full_diffusion
        config.loss.sigma_floor = config.sigma_floor
        self.model = GraphEFTModel(model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = self._build_scheduler()
        if self.config.two_stage:
            self.scheduler = None
        self.field_requires_dt = [canon in LOG_TARGET_FIELDS_CANON for canon in self.field_canonical]
        self.requires_dt_any = any(self.field_requires_dt)
        self.base_lr = self.config.lr
        self.fixed_sigma_value: float | None = None
        self.current_stage = self.config.train_stage
        self.backbone_freeze_epochs_left = 0

    def _make_loader(self, dataset: GraphEFTDataset, shuffle: bool) -> DataLoader:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def fit(self) -> dict[str, float]:
        if self.config.two_stage:
            metrics: dict[str, float] = {}
            if self.config.stageA_epochs > 0:
                metrics.update(
                    self._train_phase(
                        self.config.stageA_epochs,
                        stage_name="drift_only",
                        lr_multiplier=1.0,
                        fixed_sigma=self.config.stageA_sigma,
                    )
                )
            if self.config.stageB_epochs > 0:
                metrics.update(
                    self._train_phase(
                        self.config.stageB_epochs,
                        stage_name="full",
                        lr_multiplier=self.config.stageB_lr_multiplier,
                        freeze_backbone_epochs=self.config.freeze_backbone_in_stageB_epochs,
                    )
                )
            return metrics
        fixed = self.config.stageA_sigma if self.config.train_stage == "drift_only" else None
        return self._train_phase(self.config.max_epochs, stage_name=self.config.train_stage, fixed_sigma=fixed)

    def _train_phase(
        self,
        num_epochs: int,
        stage_name: str,
        lr_multiplier: float = 1.0,
        fixed_sigma: float | None = None,
        freeze_backbone_epochs: int = 0,
    ) -> dict[str, float]:
        if num_epochs <= 0:
            return {}
        self._configure_stage(stage_name, fixed_sigma, lr_multiplier, freeze_backbone_epochs)
        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        val_loader = self._make_loader(self.val_dataset, shuffle=False) if self.val_dataset else None
        best_val = math.inf
        metrics: dict[str, float] = {}
        last_val_loss: float | None = None
        for _ in range(num_epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            if val_loader:
                val_loss = self._run_epoch(val_loader, train=False)
                last_val_loss = val_loss
                if val_loss < best_val:
                    best_val = val_loss
                    metrics[f"{stage_name}_best_val"] = val_loss
                if self.scheduler:
                    self._step_scheduler(val_loss)
            else:
                if self.scheduler:
                    self._step_scheduler(train_loss)
            if self.backbone_freeze_epochs_left > 0:
                self.backbone_freeze_epochs_left -= 1
                if self.backbone_freeze_epochs_left == 0:
                    self._apply_backbone_requires_grad(True)
        metrics[f"{stage_name}_last_train"] = float(train_loss)
        if last_val_loss is not None:
            metrics[f"{stage_name}_last_val"] = float(last_val_loss)
        return metrics

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        total_loss = 0.0
        steps = 0
        self.model.train(train)
        for batch in loader:
            loss = self._step(batch, train=train)
            total_loss += loss
            steps += 1
        return total_loss / max(steps, 1)

    def _step(self, batch: GraphEFTBatch, train: bool) -> float:
        data = batch.pyg_data.to(self.device)
        x = data.x
        y = data.y
        mask = data.mask
        drift, diffusion_out = self.model(
            x,
            data.edge_index_temporal,
            data.edge_index_host,
            getattr(data, "edge_attr_temporal", None),
            getattr(data, "edge_attr_host", None),
        )
        residual_scale = self._get_residual_scale_tensor(drift.device)
        drift = drift * residual_scale
        targets = y
        targets = targets * residual_scale
        # Apply residual scaling to diffusion outputs so mean and covariance remain consistent.
        if self.model.config.full_diffusion:
            diffusion_out = diffusion_out * residual_scale.view(1, -1, 1)
        else:
            log_scale = 2.0 * torch.log(residual_scale.clamp(min=self.config.loss.scale_epsilon))
            diffusion_out = diffusion_out + log_scale
        # Use dt scaling to implement increment model with rate-parameterized drift and diffusion.
        # Build a per-node, per-field scaling matrix where each column equals dt.
        dt = batch.dt.to(self.device).view(-1, 1)
        scaling = dt.repeat(1, drift.size(1))
        weights = torch.ones_like(mask)
        if self.config.mass_exponent != 0.0 or (self.config.class_weights and "M_halo" in self.config.class_weights):
            halo_mass = batch.halo_mass.to(self.device)
            mass_scale = (halo_mass / max(self.config.mass_reference, 1e-6)).clamp(min=1e-3).unsqueeze(1)
            weights = weights * mass_scale.pow(self.config.mass_exponent)
        if self.config.class_weights:
            for idx, field in enumerate(self.config.fields):
                weights[:, idx] *= self.config.class_weights.get(field, 1.0)
        snap = batch.snap.to(self.device)
        if self.layer_weights:
            snap_factor = torch.ones_like(snap)
            for snap_id, factor in self.layer_weights.items():
                snap_factor = torch.where(snap == float(snap_id), snap_factor * factor, snap_factor)
            weights = weights * snap_factor.view(-1, 1)
        elif self.config.snap_weight_power > 0.0 and self.snap_max > self.snap_min:
            span = max(float(self.snap_max - self.snap_min), 1.0)
            norm = torch.clamp((snap - float(self.snap_min)) / span, min=0.0)
            layer_factor = torch.pow(norm + 1e-3, self.config.snap_weight_power)
            weights = weights * layer_factor.view(-1, 1)
        drift_train, diff_train, targets_train, mask_train, weights_train, scaling_train = self._select_train_fields(
            drift,
            diffusion_out,
            targets,
            mask,
            weights,
            scaling,
        )
        loss, _ = onsager_machlup_loss(
            drift_train,
            diff_train,
            targets_train,
            mask_train,
            self.config.loss,
            weights_train,
            scaling=scaling_train,
            fixed_sigma=self.fixed_sigma_value,
        )
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return float(loss.item())

    def _build_scheduler(self):
        name = self.config.scheduler.lower()
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.max_epochs)
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=10, factor=0.5)
        raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _step_scheduler(self, metric: float) -> None:
        if not self.scheduler:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _configure_stage(
        self,
        stage_name: str,
        fixed_sigma: float | None,
        lr_multiplier: float,
        freeze_backbone_epochs: int,
    ) -> None:
        self.current_stage = stage_name
        self.fixed_sigma_value = fixed_sigma
        self.backbone_freeze_epochs_left = max(0, freeze_backbone_epochs)
        new_lr = self.base_lr * lr_multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        self._set_diffusion_trainable(stage_name != "drift_only")
        if self.backbone_freeze_epochs_left > 0:
            self._apply_backbone_requires_grad(False)
        else:
            self._apply_backbone_requires_grad(True)
        self.optimizer.zero_grad(set_to_none=True)

    def _get_active_index_tensor(self, device: torch.device) -> torch.Tensor:
        if self._active_index_tensor is None or self._active_index_tensor.device != device:
            self._active_index_tensor = torch.tensor(self.train_field_indices, dtype=torch.long, device=device)
        return self._active_index_tensor

    def _get_residual_scale_tensor(self, device: torch.device) -> torch.Tensor:
        if self._residual_scale_tensor is None or self._residual_scale_tensor.device != device:
            scale = torch.tensor(self.residual_scales, dtype=torch.float32, device=device).view(1, -1)
            self._residual_scale_tensor = scale
        return self._residual_scale_tensor

    def _select_train_fields(
        self,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None,
        scaling: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if len(self.train_field_indices) == drift.size(1):
            return drift, diffusion, targets, mask, weights, scaling
        idx = self._get_active_index_tensor(drift.device)
        drift_sel = drift.index_select(1, idx)
        targets_sel = targets.index_select(1, idx)
        mask_sel = mask.index_select(1, idx)
        weights_sel = weights.index_select(1, idx) if weights is not None else None
        scaling_sel = scaling.index_select(1, idx) if scaling is not None else None
        if self.model.config.full_diffusion:
            diff_sel = diffusion.index_select(1, idx).index_select(2, idx)
        else:
            diff_sel = diffusion.index_select(1, idx)
        return drift_sel, diff_sel, targets_sel, mask_sel, weights_sel, scaling_sel

    def _set_diffusion_trainable(self, enabled: bool) -> None:
        for param in self.model.diff_head.parameters():
            param.requires_grad = enabled

    def _apply_backbone_requires_grad(self, enabled: bool) -> None:
        modules = [
            self.model.temporal_edge_encoder,
            self.model.host_edge_encoder,
            *self.model.temporal_convs,
            *[m for m in self.model.host_convs if m is not None],
            *self.model.norms,
            self.model.final_norm,
        ]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = enabled
