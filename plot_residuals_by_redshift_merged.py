#!/usr/bin/env python3
"""Merged residual diagnostics by redshift for Transport-only and GPLM.

For each retained layer, the script computes the log-ratio residual
    log10(M_pred / M_truth)
and summarizes it with the median and 16--84 percentile band. The two modeled
fields are overlaid with different colors, and the x-axis is labeled by the
corresponding redshift instead of snapshot number.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jf_extractors import extract_field_array
from jf_graph import LayeredGraph
from jf_time import scale_to_redshift, snap_to_scale_factor
from validate_per_layer import align_layers


def load_pairs(truth_list: Path, pred_dir: Path, pred_suffix: str) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    with truth_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            truth_path = line.strip()
            if not truth_path:
                continue
            truth = Path(truth_path).resolve()
            parent = truth.parent.name
            pred_name = f"{parent}{pred_suffix}" if parent else truth.stem + pred_suffix
            pred = pred_dir / pred_name
            if pred.exists():
                pairs.append((truth, pred.resolve()))
    if not pairs:
        raise SystemExit("No graph pairs found.")
    return pairs


def compute_seed_masks(graph: LayeredGraph) -> Dict[int, np.ndarray]:
    masks: Dict[int, np.ndarray] = {}
    for idx, layer in enumerate(graph.layers):
        n = len(layer.get("nodes", []))
        seeded = np.ones(n, dtype=bool)
        if idx > 0:
            prev_layer = graph.layers[idx - 1]
            ids_curr = layer.get("id2idx", {})
            ids_prev = prev_layer.get("id2idx", {})
            for src, tgt in layer.get("time_edges", []):
                if src in ids_prev and tgt in ids_curr:
                    seeded[ids_curr[tgt]] = False
        snap = layer.get("snap")
        key = int(snap) if snap is not None else idx
        masks[key] = seeded
    return masks


def collect_residuals_by_snapshot(
    field: str,
    truth_list: Path,
    pred_dir: Path,
    pred_suffix: str,
    *,
    exclude_seeded: bool = True,
) -> Dict[int, np.ndarray]:
    by_snap: Dict[int, List[np.ndarray]] = {}
    for tpath, ppath in load_pairs(truth_list, pred_dir, pred_suffix):
        tg = LayeredGraph(str(tpath))
        pg = LayeredGraph(str(ppath))
        seed_masks = compute_seed_masks(tg) if exclude_seeded else {}
        for bundle in align_layers(tg, pg):
            lt = bundle["truth_layer"]
            lp = bundle["pred_layer"]
            ids = bundle["ids"]
            if not ids:
                continue
            snap = lt.get("snap")
            snap_key = int(snap) if snap is not None else None
            if snap_key is None:
                continue
            t_idx = np.array([lt["id2idx"][nid] for nid in ids], dtype=int)
            p_idx = np.array([lp["id2idx"][nid] for nid in ids], dtype=int)
            if exclude_seeded:
                seed_layer = seed_masks.get(snap_key)
                if seed_layer is not None:
                    keep = ~seed_layer[t_idx]
                    if not np.any(keep):
                        continue
                    t_idx = t_idx[keep]
                    p_idx = p_idx[keep]
            truth = np.asarray(extract_field_array(lt, field)[t_idx], dtype=float)
            pred = np.asarray(extract_field_array(lp, field)[p_idx], dtype=float)
            mask = np.isfinite(truth) & np.isfinite(pred)
            mask &= (truth > 0.0) & (pred > 0.0)
            if not np.any(mask):
                continue
            resid = np.log10(pred[mask] / truth[mask])
            by_snap.setdefault(snap_key, []).append(resid)
    out: Dict[int, np.ndarray] = {}
    for snap, chunks in by_snap.items():
        out[snap] = np.concatenate(chunks) if chunks else np.array([])
    return out


def snap_to_redshift_label(snap: int) -> str:
    a = snap_to_scale_factor(snap)
    z = scale_to_redshift(a)
    if z is None:
        return str(snap)
    if abs(z - round(z)) < 0.05:
        return f"{int(round(z))}"
    return f"{z:.1f}"


def summary_arrays(by_snap: Dict[int, np.ndarray], snaps: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.full(len(snaps), np.nan, dtype=float)
    lo = np.full(len(snaps), np.nan, dtype=float)
    hi = np.full(len(snaps), np.nan, dtype=float)
    for i, snap in enumerate(snaps):
        arr = by_snap.get(snap, np.array([]))
        if arr.size == 0:
            continue
        med[i] = float(np.median(arr))
        lo[i] = float(np.percentile(arr, 16))
        hi[i] = float(np.percentile(arr, 84))
    return med, lo, hi


def plot_panel(ax, snaps: List[int], summaries: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], title: str) -> None:
    x = np.arange(len(snaps), dtype=float)
    styles = {
        #"M_star": {"color": "#c73e1d", "label": r"$M_\star$"},
        #"M_gas": {"color": "#1f78b4", "label": r"$M_{\rm gas}$"},
        "M_star": {"color": "red", "label": r"$M_\star$"},
        "M_gas": {"color": "blue", "label": r"$M_{\rm gas}$"},
    }
    for field in ("M_star", "M_gas"):
        med, lo, hi = summaries[field]
        style = styles[field]
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.20, linewidth=0)
        ax.plot(x, med, color=style["color"], lw=2.0, marker="o", ms=4, label=style["label"])
    ax.axhline(0.0, color="k", ls="--", lw=1.0)
    ax.set_title(title, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([snap_to_redshift_label(s) for s in snaps], rotation=45, fontsize=12)
    ax.set_xlabel("redshift", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(frameon=False, loc="best", fontsize=13)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth-list", default="test_graphs.txt")
    ap.add_argument("--transport-dir", default="painted_transport")
    ap.add_argument("--pred-dir", default="painted_gplm")
    ap.add_argument("--transport-suffix", default="_transport.json")
    ap.add_argument("--pred-suffix", default="_gplm.json")
    ap.add_argument("--out", default="draft/residuals_by_redshift_merged.png")
    ap.add_argument("--exclude-seeded", type=int, default=1)
    args = ap.parse_args()

    transport = {
        field: collect_residuals_by_snapshot(
            field,
            Path(args.truth_list),
            Path(args.transport_dir),
            args.transport_suffix,
            exclude_seeded=bool(args.exclude_seeded),
        )
        for field in ("M_star", "M_gas")
    }
    gplm = {
        field: collect_residuals_by_snapshot(
            field,
            Path(args.truth_list),
            Path(args.pred_dir),
            args.pred_suffix,
            exclude_seeded=bool(args.exclude_seeded),
        )
        for field in ("M_star", "M_gas")
    }

    snaps = sorted(set(transport["M_star"]) | set(transport["M_gas"]) | set(gplm["M_star"]) | set(gplm["M_gas"]))
    if not snaps:
        raise SystemExit("No residual data found.")

    transport_summary = {field: summary_arrays(transport[field], snaps) for field in ("M_star", "M_gas")}
    gplm_summary = {field: summary_arrays(gplm[field], snaps) for field in ("M_star", "M_gas")}

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    plot_panel(axes[0], snaps, transport_summary, "Transport-only")
    plot_panel(axes[1], snaps, gplm_summary, "GPLM")
    axes[0].set_ylabel(r"$\log_{10}(M_{\rm pred}/M_{\rm truth})$", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"[resid-merged] wrote {args.out}")


if __name__ == "__main__":
    main()
