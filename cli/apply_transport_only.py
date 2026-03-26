#!/usr/bin/env python3
"""Apply deterministic transport-only baseline to layered graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gplm.export import export_all
from gplm.transport_baseline import TransportOnlyConfig, generate_transport_predictions
from jf_constants import BACKGROUND_FIELDS
from jf_graph import LayeredGraph


def load_paths(list_file: Path | None, extra: List[str]) -> List[Path]:
    paths: List[Path] = []
    if list_file:
        with list_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    paths.append(Path(line).resolve())
    for path in extra:
        if path:
            paths.append(Path(path).resolve())
    if not paths:
        raise SystemExit("No target graphs provided; supply --target-list or positional paths.")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", default="M_star,M_gas", help="Comma-separated list of fields to export.")
    parser.add_argument("--target-list", default=None, help="Text file containing layered graph paths (one per line).")
    parser.add_argument("paths", nargs="*", help="Additional layered graph paths.")
    parser.add_argument("--out-dir", required=True, help="Directory to store painted transport-only graphs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not fields:
        raise SystemExit("No fields specified via --fields.")
    target_paths = load_paths(Path(args.target_list).resolve() if args.target_list else None, args.paths)

    config = TransportOnlyConfig(mode="transport_only")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def output_name(graph_path: Path) -> Path:
        parent = graph_path.parent.name
        base = parent if parent else graph_path.stem
        return out_dir / f"{base}_transport.json"

    background_requested = [f for f in fields if f in BACKGROUND_FIELDS]
    pred_fields = [f for f in fields if f not in BACKGROUND_FIELDS]
    export_fields = pred_fields + [f for f in background_requested if f not in pred_fields]

    for target in target_paths:
        if not target.exists():
            raise FileNotFoundError(f"Missing layered graph: {target}")
        graph = LayeredGraph(str(target))
        preds = generate_transport_predictions(graph, pred_fields, config)
        pred_path = output_name(target)
        export_all(graph, export_fields, preds, None, str(pred_path))
        print(f"[apply_transport_only] Painted {target} -> {pred_path}")


if __name__ == "__main__":
    main()
