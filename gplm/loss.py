"""Loss functions for GPLM."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossConfig:
    divergence_weight: float = 0.0
    clamp_log_diffusion: float | None = 4.0
    diff_epsilon: float = 1e-4
    full_diffusion: bool = False
    diagonal_jitter: float = 1e-6
    sigma_floor: float = 1e-3
    scale_epsilon: float = 1e-8


def _diagonal_loss(
    drift: torch.Tensor,
    log_diffusion: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
    weights: torch.Tensor | None,
    scaling: torch.Tensor | None,
    fixed_sigma: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if scaling is None:
        scaling = torch.ones_like(drift)
    scaling = scaling.clamp(min=config.scale_epsilon)
    if fixed_sigma is not None:
        sigma2_step = torch.full_like(drift, fixed_sigma**2)
    else:
        if config.clamp_log_diffusion is not None:
            log_diffusion = log_diffusion.clamp(min=-config.clamp_log_diffusion, max=config.clamp_log_diffusion)
        sigma2_rate = torch.exp(log_diffusion)
        sigma2_step = sigma2_rate * scaling
        if config.sigma_floor is not None and config.sigma_floor > 0.0:
            sigma2_step = torch.clamp(sigma2_step, min=config.sigma_floor**2)
    sigma2_step = sigma2_step + config.diff_epsilon
    residual = targets - drift * scaling
    weighted = residual.pow(2) / sigma2_step + torch.log(sigma2_step)
    if weights is not None:
        weighted = weighted * weights
    masked = weighted * mask
    denom = (mask * (weights if weights is not None else 1.0)).sum().clamp(min=1.0)
    data_loss = 0.5 * masked.sum() / denom
    metrics = {
        "data_loss": data_loss.detach(),
        "residual_norm": (residual * mask).sum().sqrt(),
        "sigma_mean": sigma2_step.mean().detach(),
    }
    return data_loss, metrics


def _full_loss(
    drift: torch.Tensor,
    chol_factors: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
    weights: torch.Tensor | None,
    scaling: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = drift.shape[0]
    data_loss = drift.new_zeros(batch)
    total_weight = drift.new_zeros(batch)
    if scaling is None:
        scaling = torch.ones_like(drift)
    scaling = scaling.clamp(min=config.scale_epsilon)

    residual_full = targets - drift * scaling

    for idx in range(batch):
        obs = mask[idx] > 0.5
        if not torch.any(obs):
            continue
        obs_idx = torch.nonzero(obs, as_tuple=False).squeeze(1)
        scale_vec = scaling[idx, obs_idx]
        res = residual_full[idx, obs_idx]
        w = weights[idx, obs_idx] if weights is not None else None
        L = chol_factors[idx]
        sub_L = L.index_select(0, obs_idx).index_select(1, obs_idx).clone()
        scale_sqrt = torch.sqrt(scale_vec)
        sub_L = sub_L * scale_sqrt.unsqueeze(1)
        sub_L.diagonal().add_(config.diagonal_jitter)
        if config.sigma_floor is not None and config.sigma_floor > 0.0:
            diag = sub_L.diagonal()
            floor = config.sigma_floor
            sign = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
            diag = torch.where(diag.abs() < floor, sign * floor, diag)
            sub_L = sub_L.clone()
            sub_L.diagonal().copy_(diag)
        y = torch.linalg.solve_triangular(sub_L, res.unsqueeze(1), upper=False).squeeze(1)
        mahalanobis = (y ** 2).sum()
        diag_abs = sub_L.diagonal().abs().clamp_min(config.diff_epsilon)
        logdet = 2.0 * torch.log(diag_abs).sum()
        value = 0.5 * (mahalanobis + logdet)
        if w is not None:
            weight_sum = w.sum().clamp(min=config.diff_epsilon)
            value = value * (weight_sum / max(w.numel(), 1))
            total_weight[idx] += weight_sum
        else:
            total_weight[idx] += obs.float().sum()
        data_loss[idx] = value

    denom = total_weight.sum().clamp(min=1.0)
    loss = data_loss.sum() / denom
    metrics = {
        "data_loss": loss.detach(),
        "residual_norm": torch.linalg.norm(residual_full * mask, ord=2),
        "chol_diag_mean": chol_factors.diagonal(dim1=-2, dim2=-1).mean().detach(),
    }
    return loss, metrics


def onsager_machlup_loss(
    drift: torch.Tensor,
    diffusion_output: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    config: LossConfig,
    weights: torch.Tensor | None = None,
    scaling: torch.Tensor | None = None,
    fixed_sigma: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Discrete GPLM action loss supporting diagonal or full diffusion."""
    if config.full_diffusion:
        loss, metrics = _full_loss(drift, diffusion_output, targets, mask, config, weights, scaling)
    else:
        loss, metrics = _diagonal_loss(drift, diffusion_output, targets, mask, config, weights, scaling, fixed_sigma)
    metrics["total"] = loss.detach()
    return loss, metrics
