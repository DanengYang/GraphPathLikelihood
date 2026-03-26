#!/usr/bin/env python3
"""Apply a trained Graph Path Likelihood Model (GPLM) to halo graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gplm.export import export_all
from gplm.inference import apply_model_to_graph
from gplm.model import GraphEFTConfig, GraphEFTModel
from jf_graph import LayeredGraph
from jf_constants import BACKGROUND_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Graph Path Likelihood Model (GPLM) to graphs.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint from train_gplm.py")
    parser.add_argument("--fields", required=True, help="Comma-separated fields to paint.")
    parser.add_argument("--target-list", required=True, help="Text file with target graphs.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--spatial-features", choices=["on", "off"], default="off", help="Use host distance/velocity features if available.")
    parser.add_argument("--ablate-host-edges", type=int, choices=[0, 1], default=0, help="Drop host edges during evaluation for ablation.")
    parser.add_argument("--sigma-dir", default=None, help="Optional directory to write per-field sigma JSONs.")
    parser.add_argument("--mass-log-eps", type=float, default=None, help="Override the log(M+eps) offset stored in the checkpoint (default: checkpoint value).")
    return parser.parse_args()


def load_paths(path: str | None) -> list[str]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    raw_fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    targets = load_paths(args.target_list)
    if not targets:
        raise SystemExit("No target graphs provided.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    chk_config = checkpoint.get("config", {})
    model_cfg = GraphEFTConfig(**chk_config["model"])
    extra_features = chk_config.get("extra_features", [])
    model_fields = list(model_cfg.field_names or [])
    if not model_fields:
        model_fields = raw_fields
        model_cfg.field_names = model_fields
    model = GraphEFTModel(model_cfg)
    model.load_state_dict(checkpoint["model"])
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    model.to(device_t)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_out_dir = Path(args.sigma_dir) if args.sigma_dir else None
    if sigma_out_dir:
        sigma_out_dir.mkdir(parents=True, exist_ok=True)
    def output_name(path: str) -> Path:
        p = Path(path)
        parent = p.parent.name
        base = parent if parent else p.stem
        return out_dir / f"{base}_gplm.json"

    background_requested = [f for f in raw_fields if f in BACKGROUND_FIELDS]
    pred_fields = [f for f in raw_fields if f in model_fields]
    missing = sorted(set(raw_fields) - set(pred_fields) - set(background_requested))
    if missing:
        print(f"[GPLM] Skipping unsupported fields (not in checkpoint): {', '.join(missing)}")
    if not pred_fields:
        pred_fields = model_fields
    export_fields = pred_fields + [f for f in background_requested if f not in pred_fields]

    trainer_cfg = chk_config.get("trainer", {})
    extra_features = chk_config.get("extra_features", [])
    include_first_star = trainer_cfg.get("include_first_star_transitions", True)
    env_features = trainer_cfg.get("env_features", [])
    use_env_features = trainer_cfg.get("use_env_features", True)
    ablate_host = bool(args.ablate_host_edges)
    mass_log_eps = args.mass_log_eps if args.mass_log_eps is not None else trainer_cfg.get("mass_log_eps", 0.05)
    residual_scales = trainer_cfg.get("residual_scales", {})

    for target in targets:
        pred_path = output_name(target)
        preds, sigmas = apply_model_to_graph(
            target,
            model,
            pred_fields,
            device_t,
            use_spatial_features=args.spatial_features == "on",
            include_first_star=include_first_star,
            env_feature_names=env_features,
            use_env_features=use_env_features,
            ablate_host_edges=ablate_host,
            sigma_floor=trainer_cfg.get("sigma_floor", 1e-3),
            extra_features=extra_features,
            mass_log_eps=mass_log_eps,
            residual_scales=residual_scales,
        )
        graph = LayeredGraph(target)
        sigma_path = None
        if sigma_out_dir:
            sigma_path = sigma_out_dir / (pred_path.stem + "_sigma.json")
        export_all(graph, export_fields, preds, sigmas, str(pred_path), str(sigma_path) if sigma_path else None)
        print(f"Painted {target}")


if __name__ == "__main__":
    main()
