#!/usr/bin/env python3
"""CLI for training Graph Path Likelihood Model (GPLM)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gplm.trainer import GraphEFTTrainer, TrainerConfig
from jf_constants import BACKGROUND_FIELDS
from jf_utils import canonical_field_name


def parse_weights(entries: list[str] | None) -> dict[str, float] | None:
    if not entries:
        return None
    weights: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid --weight-field '{entry}' (expected name=value).")
        name, value = entry.split("=", 1)
        try:
            weights[name.strip()] = float(value)
        except ValueError as exc:
            raise SystemExit(f"Invalid weight value in '{entry}'") from exc
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Graph Path Likelihood Model (GPLM).")
    parser.add_argument("--train-list", required=True, help="Text file with training graph paths.")
    parser.add_argument("--val-list", default=None, help="Optional validation graph paths file.")
    parser.add_argument("--fields", required=True, help="Comma-separated list of fields to model.")
    parser.add_argument("--extra-features", default="", help="Optional extra features for node encoders.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save", required=True, help="Output checkpoint file.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dimension for the GNN.")
    parser.add_argument("--message-layers", type=int, default=None, help="Override number of message passing layers.")
    parser.add_argument("--dropout", type=float, default=None, help="Edge attention dropout for message passing.")
    parser.add_argument("--mlp-hidden", type=int, default=None, help="Hidden size for per-field heads.")
    parser.add_argument("--mlp-layers", type=int, default=None, help="Number of layers in per-field heads.")
    parser.add_argument("--mlp-dropout", type=float, default=None, help="Dropout inside per-field heads.")
    parser.add_argument("--mass-ref", type=float, default=1.0, help="Reference halo mass for loss weighting.")
    parser.add_argument("--mass-exp", type=float, default=0.0, help="Exponent for halo-mass loss weighting.")
    parser.add_argument("--mass-log-eps", type=float, default=0.05, help="Offset used in log(M+eps) transforms (internal mass units).")
    parser.add_argument("--weight-field", action="append", default=None, help="Field-specific loss weight: name=scale.")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none", help="Learning-rate scheduler.")
    parser.add_argument("--spatial-features", choices=["on", "off"], default="off", help="Include host distance/velocity features.")
    parser.add_argument("--full-diffusion", action="store_true", help="Predict full covariance via Cholesky factors instead of per-field variances.")
    parser.add_argument(
        "--include-first-star-transitions",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to keep transitions from zero to non-zero stellar mass in the loss.",
    )
    parser.add_argument(
        "--env-features",
        default="is_satellite,host_mass",
        help="Comma-separated environment features to append (e.g. is_satellite,host_mass,time_since_infall)",
    )
    parser.add_argument("--use-env-features", type=int, choices=[0, 1], default=1, help="Toggle environment conditioning inputs.")
    parser.add_argument("--ablate-host-edges", type=int, choices=[0, 1], default=0, help="Drop host edges during message passing for ablation studies.")
    parser.add_argument("--use-host-conv", choices=["on", "off"], default="on", help="Enable (on) or disable (off) the host-edge Transformer branch.")
    parser.add_argument("--two-stage", type=int, choices=[0, 1], default=0, help="Run drift-only then full training stages sequentially.")
    parser.add_argument("--train-stage", choices=["full", "drift_only"], default="full", help="Training stage when not using two-stage mode.")
    parser.add_argument("--stageA-epochs", type=int, default=50, help="Epochs for drift-only stage when two-stage is enabled.")
    parser.add_argument("--stageB-epochs", type=int, default=50, help="Epochs for full stage when two-stage is enabled.")
    parser.add_argument("--stageB-lr-multiplier", type=float, default=0.2, help="Learning-rate multiplier applied at the start of stage B.")
    parser.add_argument("--freeze-backbone-in-stageB-epochs", type=int, default=0, help="Number of initial stage-B epochs to freeze the message-passing backbone.")
    parser.add_argument("--sigma-floor", type=float, default=1e-3, help="Variance floor (in dex) applied after exponentiating diffusion outputs.")
    parser.add_argument("--stageA-sigma", type=float, default=1e-3, help="Fixed sigma (dex) used during drift-only stage.")
    parser.add_argument("--residual-reg-weight", type=float, default=0.0, help="Weight for residual drift regularization (0 disables).")
    parser.add_argument("--residual-reg-zref", type=float, default=2.0, help="Redshift reference scaling residual regularization.")
    parser.add_argument("--residual-reg-power", type=float, default=1.0, help="Power applied to (z/z_ref) for the residual regularizer.")
    parser.add_argument("--layer-weight-file", default=None, help="Optional file with per-snapshot loss weights (snap weight per line).")
    parser.add_argument("--snap-weight-power", type=float, default=0.0, help="If >0, weight each layer by (a/a0)^power or (snap) proxy.")
    parser.add_argument(
        "--supplementary-fields",
        default="SubhaloSFR",
        help="Comma-separated supplementary fields that are carried in features but excluded from loss (e.g. SubhaloSFR).",
    )
    parser.add_argument(
        "--residual-scale",
        action="append",
        default=None,
        metavar="FIELD=VALUE",
        help="Scale residual corrections for specific fields (e.g. M_star=0.01).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used for model initialization and data shuffling.")
    return parser.parse_args()


def load_paths(path: str | None) -> list[str]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    raw_fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    fields = [f for f in raw_fields if f not in BACKGROUND_FIELDS]
    removed = sorted(set(raw_fields) - set(fields))
    if removed:
        print(f"[GPLM] Skipping background fields (copied from input): {', '.join(removed)}")
    extras = [f.strip() for f in args.extra_features.split(",") if f.strip()]
    env_feats = [f.strip() for f in args.env_features.split(",") if f.strip()]
    supplementary = [f.strip() for f in args.supplementary_fields.split(",") if f.strip()]
    def _parse_list(entries: list[str] | None):
        if entries is None:
            return None
        values: list[str] = []
        for entry in entries:
            for item in entry.split(","):
                values.append(item.strip())
        return values
    residual_scale = parse_weights(args.residual_scale)
    if residual_scale:
        residual_scale = {canonical_field_name(k): float(v) for k, v in residual_scale.items()}
    train_paths = load_paths(args.train_list)
    val_paths = load_paths(args.val_list)
    if not train_paths:
        raise SystemExit("No training graphs provided.")

    layer_weights: dict[int, float] = {}
    if args.layer_weight_file:
        with open(args.layer_weight_file, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) != 2:
                    raise SystemExit(f"Invalid layer weight line '{line.strip()}'. Expected: <snap> <weight>")
                snap, weight = parts
                layer_weights[int(snap)] = float(weight)
    if args.snap_weight_power != 0.0 and not layer_weights:
        # Use scale factor (if available) or normalized snapshot index
        base_snap = None
        if args.train_list:
            # assume the first graph provides earliest snap
            base_snap = None
        def snap_to_weight(snap_value: int) -> float:
            return float(max(snap_value, 1) ** args.snap_weight_power)
        # Will compute on the fly inside trainer; storing exponent

    config = TrainerConfig(
        fields=fields,
        extra_features=extras,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        mass_reference=args.mass_ref,
        mass_exponent=args.mass_exp,
        mass_log_eps=args.mass_log_eps,
        class_weights=parse_weights(args.weight_field),
        scheduler=args.scheduler,
        use_spatial_features=(args.spatial_features == "on"),
        include_first_star_transitions=bool(args.include_first_star_transitions),
        env_features=env_feats,
        use_env_features=bool(args.use_env_features),
        ablate_host_edges=bool(args.ablate_host_edges),
        supplementary_fields=supplementary,
        residual_scales=residual_scale or {},
        two_stage=bool(args.two_stage),
        train_stage=args.train_stage,
        stageA_epochs=args.stageA_epochs,
        stageB_epochs=args.stageB_epochs,
        stageB_lr_multiplier=args.stageB_lr_multiplier,
        freeze_backbone_in_stageB_epochs=args.freeze_backbone_in_stageB_epochs,
        sigma_floor=args.sigma_floor,
        stageA_sigma=args.stageA_sigma,
        residual_reg_weight=args.residual_reg_weight,
        residual_reg_power=args.residual_reg_power,
        residual_reg_z_ref=args.residual_reg_zref,
        layer_weights=layer_weights,
        snap_weight_power=args.snap_weight_power,
    )
    config.model.full_diffusion = args.full_diffusion
    config.model.use_host_conv = args.use_host_conv == "on"
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.message_layers is not None:
        config.model.message_layers = args.message_layers
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.mlp_hidden is not None:
        config.model.mlp_hidden_dim = args.mlp_hidden
    if args.mlp_layers is not None:
        config.model.mlp_layers = args.mlp_layers
    if args.mlp_dropout is not None:
        config.model.mlp_dropout = args.mlp_dropout

    trainer = GraphEFTTrainer(config, train_paths, val_paths)
    metrics = trainer.fit()

    trainer_payload = {
        "include_first_star_transitions": config.include_first_star_transitions,
        "env_features": list(config.env_features),
        "use_env_features": config.use_env_features,
        "ablate_host_edges": config.ablate_host_edges,
        "two_stage": config.two_stage,
        "train_stage": config.train_stage,
        "stageA_epochs": config.stageA_epochs,
        "stageB_epochs": config.stageB_epochs,
        "stageB_lr_multiplier": config.stageB_lr_multiplier,
        "freeze_backbone_in_stageB_epochs": config.freeze_backbone_in_stageB_epochs,
        "sigma_floor": config.sigma_floor,
        "stageA_sigma": config.stageA_sigma,
        "supplementary_fields": list(config.supplementary_fields),
        "mass_log_eps": config.mass_log_eps,
        "residual_scales": dict(config.residual_scales),
        "residual_reg_weight": config.residual_reg_weight,
        "residual_reg_power": config.residual_reg_power,
        "residual_reg_z_ref": config.residual_reg_z_ref,
        "layer_weights": layer_weights,
        "snap_weight_power": config.snap_weight_power,
        "seed": args.seed,
    }
    payload = {
        "model": trainer.model.state_dict(),
        "config": {
            "fields": fields,
            "extra_features": extras,
            "model": vars(config.model),
            "trainer": trainer_payload,
        },
        "metrics": metrics,
    }
    torch.save(payload, args.save)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
