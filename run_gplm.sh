#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

bash ensure_graph_jsons.sh

TRAIN_LIST=${TRAIN_LIST:-train_paths.txt}
TEST_LIST=${TEST_LIST:-test_graphs.txt}
FIELDS=${FIELDS:-"M_star,M_gas,SubhaloSFR"}
CHECKPOINT=${CHECKPOINT:-"checkpoints/gplm_host.pt"}
OUT_DIR=${OUT_DIR:-"painted_gplm"}
TRANSPORT_OUT_DIR=${TRANSPORT_OUT_DIR:-"painted_transport"}
EXTRA_FEATURES=${EXTRA_FEATURES:-M_halo}
STAR_WEIGHT=${STAR_WEIGHT:-1.9}
MASS_EXPONENT=${MASS_EXPONENT:--1.2}
FULL_DIFFUSION=${FULL_DIFFUSION:-on}
SUPP_FIELDS=${SUPP_FIELDS:-SubhaloSFR}
EVAL_FIELDS=${EVAL_FIELDS:-M_star,M_gas}
RESIDUAL_SCALES=${RESIDUAL_SCALES:-M_star=0.01}
RESIDUAL_REG_WEIGHT=${RESIDUAL_REG_WEIGHT:-0.0}
RESIDUAL_REG_ZREF=${RESIDUAL_REG_ZREF:-2.0}
RESIDUAL_REG_POWER=${RESIDUAL_REG_POWER:-2.0}
SNAP_WEIGHT_POWER=${SNAP_WEIGHT_POWER:-1.2}
MASS_LOG_EPS=${MASS_LOG_EPS:-0.1}
PARITY_PREFIX=${PARITY_PREFIX:-figs/parity_stacked}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-1234}

DIFF_ARGS=()
if [[ "${FULL_DIFFUSION}" == "on" ]]; then
  DIFF_ARGS+=(--full-diffusion)
fi

mkdir -p "$(dirname "${CHECKPOINT}")" "${OUT_DIR}" "${TRANSPORT_OUT_DIR}" "$(dirname "${PARITY_PREFIX}")"

FIELDS="${FIELDS}" TARGET_LIST="${TEST_LIST}" TRANSPORT_OUT_DIR="${TRANSPORT_OUT_DIR}" bash run_transport_only.sh

train_cmd=(
  python cli/train_gplm.py
  --train-list "${TRAIN_LIST}"
  --fields "${FIELDS}"
  --extra-features "${EXTRA_FEATURES}"
  --hidden-dim 256
  --message-layers 8
  --mlp-hidden 256
  --mlp-layers 3
  --mlp-dropout 0.1
  --dropout 0.05
  --mass-exp "${MASS_EXPONENT}"
  --scheduler cosine
  --batch-size 8
  --device "${DEVICE}"
  --seed "${SEED}"
  --env-features is_satellite,host_mass,time_since_infall
  --use-env-features 1
  --two-stage 1
  --stageA-epochs 50
  --stageB-epochs 200
  --stageB-lr-multiplier 0.2
  --freeze-backbone-in-stageB-epochs 5
  --sigma-floor 1e-3
  --stageA-sigma 1e-3
  --residual-reg-weight "${RESIDUAL_REG_WEIGHT}"
  --residual-reg-zref "${RESIDUAL_REG_ZREF}"
  --residual-reg-power "${RESIDUAL_REG_POWER}"
  --snap-weight-power "${SNAP_WEIGHT_POWER}"
  --supplementary-fields "${SUPP_FIELDS}"
  --use-host-conv on
  --mass-log-eps "${MASS_LOG_EPS}"
  --weight-field "M_star=${STAR_WEIGHT}"
  --save "${CHECKPOINT}"
  --spatial-features on
)
train_cmd+=("${DIFF_ARGS[@]}")
if [[ -n "${RESIDUAL_SCALES}" ]]; then
  OLDIFS="${IFS}"
  IFS=',' read -ra _RS <<< "${RESIDUAL_SCALES}"
  IFS="${OLDIFS}"
  for entry in "${_RS[@]}"; do
    trimmed="$(echo "${entry}" | xargs)"
    if [[ -n "${trimmed}" ]]; then
      train_cmd+=(--residual-scale "${trimmed}")
    fi
  done
fi
"${train_cmd[@]}"

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

TEST_LIST_ENV="${TEST_LIST}" OUT_DIR_ENV="${OUT_DIR}" GPLM_LIST_ENV="${GPLM_LIST:-test_gplm_graphs.txt}" python - <<'PY'
import os
from pathlib import Path

truth_list = Path(os.environ["TEST_LIST_ENV"])
out_dir = Path(os.environ["OUT_DIR_ENV"])
out_list = Path(os.environ["GPLM_LIST_ENV"])
paths = []
with truth_list.open("r", encoding="utf-8") as handle:
    for line in handle:
        raw = line.strip()
        if not raw:
            continue
        truth = Path(raw)
        base = truth.parent.name or truth.stem
        pred = out_dir / f"{base}_gplm.json"
        if pred.exists():
            paths.append(str(pred.resolve()))
out_list.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
print(f"[run_gplm] wrote {out_list} with {len(paths)} paths")
PY

python plot_parity_stacked.py \
  --truth-list "${TEST_LIST}" \
  --transport-dir "${TRANSPORT_OUT_DIR}" \
  --gplm-dir "${OUT_DIR}" \
  --transport-suffix "_transport.json" \
  --pred-suffix "_gplm.json" \
  --fields "${EVAL_FIELDS}" \
  --out-prefix "${PARITY_PREFIX}" \
  --plot-eps 0.0

echo "[run_gplm] Completed host-conditioned GPLM training and evaluation."
