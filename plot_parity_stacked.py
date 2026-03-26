#!/usr/bin/env python3
"""Stacked parity plots across all snapshots for Transport-only vs GPLM.

By default, newly entering nodes are excluded so the comparison focuses on
nodes whose values are predicted from earlier-layer transport/inference rather
than truth-seeded at entry. Applies a mass window (in code units) and converts
to Msun for display.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from jf_graph import LayeredGraph
from jf_extractors import extract_field_array
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


def stack_field(
    field: str,
    truth_list: Path,
    dir_pred: Path,
    pred_suffix: str,
    *,
    exclude_seeded: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    all_t: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    for tpath, ppath in load_pairs(truth_list, dir_pred, pred_suffix):
        tg = LayeredGraph(str(tpath))
        pg = LayeredGraph(str(ppath))
        seed_masks = compute_seed_masks(tg) if exclude_seeded else {}
        for bundle in align_layers(tg, pg):
            lt = bundle["truth_layer"]
            lp = bundle["pred_layer"]
            ids = bundle["ids"]
            if not ids:
                continue
            t_idx = np.array([lt["id2idx"][nid] for nid in ids], dtype=int)
            p_idx = np.array([lp["id2idx"][nid] for nid in ids], dtype=int)
            if exclude_seeded:
                snap = lt.get("snap")
                snap_key = int(snap) if snap is not None else None
                seed_layer = seed_masks.get(snap_key)
                if seed_layer is not None:
                    keep = ~seed_layer[t_idx]
                    if not np.any(keep):
                        continue
                    t_idx = t_idx[keep]
                    p_idx = p_idx[keep]
            t_arr = extract_field_array(lt, field)[t_idx]
            p_arr = extract_field_array(lp, field)[p_idx]
            all_t.append(t_arr)
            all_p.append(p_arr)
    if not all_t:
        return np.array([]), np.array([])
    return np.concatenate(all_t), np.concatenate(all_p)


def plot_parity(
    ax,
    truth_lin: np.ndarray,
    pred_lin: np.ndarray,
    title: str,
    rng_lin: Tuple[float, float],
    plot_eps: float,
    vmax: float | None = None,
    *,
    title_fontsize: float = 20.0,
    label_fontsize: float = 15.0,
    tick_fontsize: float = 13.0,
) -> Dict[str, float]:
    lo, hi = rng_lin
    mask = np.isfinite(truth_lin) & np.isfinite(pred_lin)
    plot_eps = max(plot_eps, 0.0)
    mask &= (truth_lin >= lo) & (truth_lin <= hi) & (pred_lin >= lo) & (pred_lin <= hi)
    if plot_eps > 0.0:
        mask &= (truth_lin > plot_eps) & (pred_lin > plot_eps)
    if not np.any(mask):
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return {"N": 0, "RMSE": np.nan}
    t = truth_lin[mask]
    p = pred_lin[mask]
    # log10 Msun
    lt = np.log10(t + plot_eps)
    lp = np.log10(p + plot_eps)
    rmse = float(np.sqrt(np.mean((lp - lt) ** 2)))
    # 2D histogram (60x60); zero-count bins shown at colormap minimum
    bins = 60
    xedges = np.linspace(np.log10(lo), np.log10(hi), bins + 1)
    yedges = np.linspace(np.log10(lo), np.log10(hi), bins + 1)
    H, xedges, yedges = np.histogram2d(lt, lp, bins=[xedges, yedges])
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        cmap="cividis",
        vmin=0,
        vmax=vmax if vmax is not None else None,
    )
    lims = (np.log10(lo), np.log10(hi))
    ax.plot(lims, lims, "--", color="k", lw=1.0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"truth: $\log_{10}(M/(\mathrm{M}_\odot/h))$", fontsize=label_fontsize)
    ax.set_ylabel(r"pred: $\log_{10}(M/(\mathrm{M}_\odot/h))$", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=10.0)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    return {"N": int(mask.sum()), "RMSE": rmse, "mappable": im}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth-list", default="test_graphs.txt")
    ap.add_argument("--transport-dir", default="painted_transport")
    ap.add_argument("--pred-dir", default="painted_gplm")
    ap.add_argument("--transport-suffix", default="_transport.json")
    ap.add_argument("--pred-suffix", default="_gplm.json")
    # Backward-compatibility aliases.
    ap.add_argument("--sam-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gneft-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fields", default="M_star,M_gas")
    ap.add_argument("--out-prefix", default="draft/parity_stacked")
    ap.add_argument("--min-code", type=float, default=1e-5)
    ap.add_argument("--max-code", type=float, default=10 ** 2.1)
    ap.add_argument("--plot-eps", type=float, default=0.0, help="Floor added before log10 (in Msun). If <=0, uses min-code.")
    ap.add_argument("--exclude-seeded", type=int, default=1, help="Exclude newly entering truth-seeded nodes (default: 1).")
    args = ap.parse_args()

    if args.plot_eps <= 0.0:
        args.plot_eps = args.min_code * 1e10
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    transport_dir = Path(args.sam_dir) if args.sam_dir else Path(args.transport_dir)
    pred_dir = Path(args.gneft_dir) if args.gneft_dir else Path(args.pred_dir)
    field_labels = {"M_star": r"$M_{\\star}$", "M_gas": r"$M_{\\rm gas}$"}
    title_labels = {"M_star": r"M$_{\star}$", "M_gas": r"M$_{\rm gas}$"}
    # code units to Msun
    lo = args.min_code * 1e10
    hi = args.max_code * 1e10
    for f in fields:
        t_sam, p_sam = stack_field(
            f, Path(args.truth_list), transport_dir, args.transport_suffix, exclude_seeded=bool(args.exclude_seeded)
        )
        t_g, p_g = stack_field(
            f, Path(args.truth_list), pred_dir, args.pred_suffix, exclude_seeded=bool(args.exclude_seeded)
        )
        # convert from code units to Msun
        t_sam *= 1e10
        p_sam *= 1e10
        t_g *= 1e10
        p_g *= 1e10

        fig = plt.figure(figsize=(11.2, 5.4))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.055], wspace=0.18)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1], sharex=ax_left, sharey=ax_left)
        cax = fig.add_subplot(gs[0, 2])
        vmax = 500.0 if f == "M_star" else (200.0 if f == "M_gas" else None)
        fl = field_labels.get(f, f)
        title_part = title_labels.get(f, f)
        title_sam = f"Transport-only ({title_part})"
        title_g = f"GPLM ({title_part})"
        stats_sam = plot_parity(
            ax_left, t_sam, p_sam, title_sam, (lo, hi), args.plot_eps, vmax=vmax,
            title_fontsize=22.0, label_fontsize=16.0, tick_fontsize=13.0,
        )
        stats_g = plot_parity(
            ax_right, t_g, p_g, title_g, (lo, hi), args.plot_eps, vmax=vmax,
            title_fontsize=22.0, label_fontsize=16.0, tick_fontsize=13.0,
        )
        cb = fig.colorbar(stats_g["mappable"], cax=cax)
        cb.set_label("counts", fontsize=14.0)
        cb.ax.tick_params(labelsize=12.0)
        # Stats label: use TeX-safe line breaks when text.usetex is enabled.
        line_break = r"\\" if plt.rcParams.get("text.usetex", False) else "\n"
        stats_text_sam = f"N={stats_sam['N']}{line_break}RMSE={stats_sam['RMSE']:.3f}"
        stats_text_g = f"N={stats_g['N']}{line_break}RMSE={stats_g['RMSE']:.3f}"
        ax_left.text(
            0.02,
            0.98,
            stats_text_sam,
            transform=ax_left.transAxes,
            fontsize=14.0,
            ha="left",
            va="top",
            multialignment="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax_right.text(
            0.02,
            0.98,
            stats_text_g,
            transform=ax_right.transAxes,
            fontsize=14.0,
            ha="left",
            va="top",
            multialignment="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax_right.set_ylabel("")
        out = f"{args.out_prefix}_{f}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[parity] wrote {out}")


if __name__ == "__main__":
    main()
