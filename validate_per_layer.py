#!/usr/bin/env python3
"""Layer-wise validation for parametric EFT predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from jf_graph import LayeredGraph
from jf_extractors import extract_field_array


def load_graph(path: Path) -> LayeredGraph:
    return LayeredGraph(str(path))


def align_layers(orig: LayeredGraph, pred: LayeredGraph) -> List[Dict[str, np.ndarray]]:
    truth_by_snap = {}
    pred_by_snap = {}
    for layer in orig.layers:
        snap = int(layer.get("snap", len(truth_by_snap)))
        truth_by_snap[snap] = layer
    for layer in pred.layers:
        snap = int(layer.get("snap", len(pred_by_snap)))
        pred_by_snap[snap] = layer

    common_snaps = sorted(set(truth_by_snap.keys()).intersection(pred_by_snap.keys()))
    if not common_snaps:
        raise ValueError("No overlapping snapshots between truth and prediction graphs.")

    per_layer = []
    for snap in common_snaps:
        layer_truth = truth_by_snap[snap]
        layer_pred = pred_by_snap[snap]
        ids_truth = set(layer_truth["id2idx"].keys())
        ids_pred = set(layer_pred["id2idx"].keys())
        common_ids = sorted(ids_truth.intersection(ids_pred))
        if not common_ids:
            continue
        per_layer.append(
            {
                "snap": snap,
                "truth_layer": layer_truth,
                "pred_layer": layer_pred,
                "ids": common_ids,
            }
        )
    return per_layer


def compute_metrics_per_layer(
    orig: LayeredGraph,
    pred: LayeredGraph,
    fields: List[str],
    zero_tol: float = 1e-8,
    log_transform: bool = False,
    log_eps: float = 0.05,
) -> List[Dict[str, float]]:
    alignment = align_layers(orig, pred)
    results: List[Dict[str, float]] = []
    for snap_idx, layers in enumerate(alignment):
        truth_layer = layers["truth_layer"]
        pred_layer = layers["pred_layer"]
        ids = layers["ids"]
        if not ids:
            continue
        truth_idx = [truth_layer["id2idx"][node_id] for node_id in ids]
        pred_idx = [pred_layer["id2idx"][node_id] for node_id in ids]
        row = {"snap": layers["snap"]}
        for field in fields:
            truth_full = extract_field_array(truth_layer, field)
            pred_full = extract_field_array(pred_layer, field)
            truth_raw = truth_full[truth_idx]
            pred_raw = pred_full[pred_idx]
            truth = truth_raw
            pred_vals = pred_raw
            if log_transform:
                truth = np.log10(np.clip(truth_raw, 0.0, None) + log_eps)
                pred_vals = np.log10(np.clip(pred_raw, 0.0, None) + log_eps)
            mask = np.isfinite(truth) & np.isfinite(pred_vals)
            if not np.any(mask):
                row[f"{field}_rmse"] = np.nan
                row[f"{field}_mae"] = np.nan
                row[f"{field}_zero_truth"] = int(np.sum(truth_raw <= zero_tol))
                row[f"{field}_zero_pred"] = int(np.sum(pred_raw <= zero_tol))
                continue
            diff = pred_vals[mask] - truth[mask]
            row[f"{field}_rmse"] = float(np.sqrt(np.mean(diff * diff)))
            row[f"{field}_mae"] = float(np.mean(np.abs(diff)))
            row[f"{field}_zero_truth"] = int(np.sum(truth_raw <= zero_tol))
            row[f"{field}_zero_pred"] = int(np.sum(pred_raw <= zero_tol))
        results.append(row)
    return results


def write_metrics(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("snap")
        if rows:
            keys = [k for k in rows[0].keys() if k != "snap"]
            for key in keys:
                handle.write(f",{key}")
        handle.write("\n")
        for row in rows:
            keys = [k for k in row.keys() if k != "snap"]
            handle.write(str(row["snap"]))
            for key in keys:
                handle.write(f",{row[key]}")
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-layer metrics for parametric EFT predictions.")
    parser.add_argument("--truth-json", required=True, help="Reference layered graph JSON.")
    parser.add_argument("--pred-json", required=True, help="Painted layered graph JSON.")
    parser.add_argument(
        "--fields",
        default="M_star,M_gas,M_wind,M_bh,Z_gas,Z_star,rhalf_gas,rhalf_star,rhalf_dm",
        help="Comma-separated list of fields to evaluate.",
    )
    parser.add_argument("--out-csv", required=True, help="Where to write the per-layer metrics CSV.")
    parser.add_argument("--zero-tol", type=float, default=1e-8, help="Threshold treated as zero occupancy.")
    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Optional directory to save RMSE/MAE plots per field.",
    )
    parser.add_argument("--log-metrics", type=int, choices=[0, 1], default=0, help="If 1, compute metrics in log10(M+eps).")
    parser.add_argument("--mass-log-eps", type=float, default=0.05, help="Offset used in log10(M+eps) when --log-metrics=1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    truth = load_graph(Path(args.truth_json))
    pred = load_graph(Path(args.pred_json))
    rows = compute_metrics_per_layer(
        truth,
        pred,
        fields,
        zero_tol=args.zero_tol,
        log_transform=bool(args.log_metrics),
        log_eps=args.mass_log_eps,
    )
    write_metrics(Path(args.out_csv), rows)
    if args.fig_dir:
        import matplotlib.pyplot as plt

        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        snaps = [row["snap"] for row in rows]
        for field in fields:
            rmse_key = f"{field}_rmse"
            mae_key = f"{field}_mae"
            if rmse_key not in rows[0]:
                continue
            rmse = [row.get(rmse_key, np.nan) for row in rows]
            mae = [row.get(mae_key, np.nan) for row in rows]
            plt.figure()
            plt.plot(snaps, rmse, marker="o", label="RMSE")
            plt.plot(snaps, mae, marker="s", label="MAE")
            plt.xlabel("Snapshot")
            plt.ylabel("Error")
            plt.title(f"Per-layer errors for {field}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{field}_metrics.png", dpi=150)
            plt.close()
    print(f"Wrote per-layer metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
