#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${1:-dataGraphs}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ensure-graph-jsons] Missing directory: ${DATA_DIR}" >&2
  exit 1
fi

json_count=$(find "${DATA_DIR}" -type f -name '*.json' | wc -l | tr -d ' ')
gz_count=$(find "${DATA_DIR}" -type f -name '*.json.gz' | wc -l | tr -d ' ')

if [[ "${json_count}" != "0" ]]; then
  echo "[ensure-graph-jsons] Found ${json_count} JSON graph file(s); no decompression needed."
  exit 0
fi

if [[ "${gz_count}" == "0" ]]; then
  echo "[ensure-graph-jsons] No graph JSON files or JSON archives found under ${DATA_DIR}." >&2
  exit 1
fi

echo "[ensure-graph-jsons] Restoring ${gz_count} compressed graph file(s)."
bash "${ROOT_DIR}/decompress_data_graphs.sh" "${DATA_DIR}"
