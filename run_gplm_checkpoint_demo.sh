#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CHECKPOINT=${CHECKPOINT:-"checkpoints/gplm_host.pt"}
TEST_LIST=${TEST_LIST:-"test_graphs.txt"}
FIELDS=${FIELDS:-"M_star,M_gas,SubhaloSFR"}
TRANSPORT_OUT_DIR=${TRANSPORT_OUT_DIR:-"painted_transport"}
OUT_DIR=${OUT_DIR:-"painted_gplm"}
PARITY_PREFIX=${PARITY_PREFIX:-"figs/parity_stacked"}
RESIDUAL_OUT=${RESIDUAL_OUT:-"figs/residuals_by_redshift_merged.png"}
DEVICE=${DEVICE:-auto}
MASS_LOG_EPS=${MASS_LOG_EPS:-0.1}

bash ensure_graph_jsons.sh

mkdir -p "${TRANSPORT_OUT_DIR}" "${OUT_DIR}" "$(dirname "${PARITY_PREFIX}")" "$(dirname "${RESIDUAL_OUT}")"

FIELDS="${FIELDS}" TARGET_LIST="${TEST_LIST}" TRANSPORT_OUT_DIR="${TRANSPORT_OUT_DIR}" bash run_transport_only.sh

python cli/apply_gplm.py \
  --checkpoint "${CHECKPOINT}" \
  --fields "${FIELDS}" \
  --target-list "${TEST_LIST}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --spatial-features on \
  --sigma-dir "${OUT_DIR}/sigma" \
  --mass-log-eps "${MASS_LOG_EPS}" \
  --ablate-host-edges 0

python plot_parity_stacked.py \
  --truth-list "${TEST_LIST}" \
  --transport-dir "${TRANSPORT_OUT_DIR}" \
  --pred-dir "${OUT_DIR}" \
  --transport-suffix "_transport.json" \
  --pred-suffix "_gplm.json" \
  --fields "M_star,M_gas" \
  --out-prefix "${PARITY_PREFIX}" \
  --plot-eps 0.0

python plot_residuals_by_redshift_merged.py \
  --truth-list "${TEST_LIST}" \
  --transport-dir "${TRANSPORT_OUT_DIR}" \
  --pred-dir "${OUT_DIR}" \
  --transport-suffix "_transport.json" \
  --pred-suffix "_gplm.json" \
  --out "${RESIDUAL_OUT}"

echo "[run_gplm_checkpoint_demo] Completed checkpoint-based painting and validation figures."
