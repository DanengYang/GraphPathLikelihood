#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${1:-dataGraphs}"
KEEP_ORIGINALS="${KEEP_ORIGINALS:-0}"
LIMIT_BYTES=$((25 * 1024 * 1024))

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[compress-dataGraphs] Missing directory: ${DATA_DIR}" >&2
  exit 1
fi

count=0
failed=0

while IFS= read -r -d '' json_path; do
  gz_path="${json_path}.gz"
  tmp_path="${gz_path}.tmp"

  if [[ -f "${gz_path}" ]]; then
    if [[ "${KEEP_ORIGINALS}" != "1" && -f "${json_path}" ]]; then
      rm -f "${json_path}"
      count=$((count + 1))
      echo "[compress-dataGraphs] Removed restored JSON; archive already exists: ${json_path}"
    else
      echo "[compress-dataGraphs] Skip existing archive: ${gz_path}"
    fi
    continue
  fi

  gzip -n -9 -c "${json_path}" > "${tmp_path}"
  size_bytes=$(stat -c '%s' "${tmp_path}")

  if (( size_bytes >= LIMIT_BYTES )); then
    echo "[compress-dataGraphs] ERROR: ${json_path} compresses to $((size_bytes / 1024 / 1024)) MB, not below 25 MB" >&2
    rm -f "${tmp_path}"
    failed=1
    continue
  fi

  mv "${tmp_path}" "${gz_path}"
  if [[ "${KEEP_ORIGINALS}" != "1" ]]; then
    rm -f "${json_path}"
  fi

  count=$((count + 1))
  echo "[compress-dataGraphs] ${json_path} -> ${gz_path} ($(python - <<'PY' "${size_bytes}"
import sys
print(f"{int(sys.argv[1]) / 1024 / 1024:.2f} MB")
PY
))"
done < <(find "${DATA_DIR}" -type f -name '*.json' -print0 | sort -z)

if (( failed != 0 )); then
  echo "[compress-dataGraphs] At least one file exceeded the 25 MB compressed limit." >&2
  exit 1
fi

echo "[compress-dataGraphs] Compressed ${count} file(s)."
