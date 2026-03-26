#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

FIELDS=${FIELDS:-"M_star,M_gas,SubhaloSFR"}
TARGET_LIST=${TARGET_LIST:-test_graphs.txt}
TRANSPORT_OUT_DIR=${TRANSPORT_OUT_DIR:-painted_transport}

mkdir -p "${TRANSPORT_OUT_DIR}"

python cli/apply_transport_only.py \
  --fields "${FIELDS}" \
  --target-list "${TARGET_LIST}" \
  --out-dir "${TRANSPORT_OUT_DIR}"

echo "[run_transport_only] Completed deterministic transport-only painting."
