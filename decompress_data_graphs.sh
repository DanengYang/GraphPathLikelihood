#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${1:-dataGraphs}"
DELETE_ARCHIVES="${DELETE_ARCHIVES:-0}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[decompress-dataGraphs] Missing directory: ${DATA_DIR}" >&2
  exit 1
fi

count=0

while IFS= read -r -d '' gz_path; do
  json_path="${gz_path%.gz}"

  if [[ -f "${json_path}" ]]; then
    echo "[decompress-dataGraphs] Skip existing file: ${json_path}"
    continue
  fi

  if [[ "${DELETE_ARCHIVES}" == "1" ]]; then
    gzip -d "${gz_path}"
  else
    gzip -dk "${gz_path}"
  fi

  count=$((count + 1))
  echo "[decompress-dataGraphs] Restored ${json_path}"
done < <(find "${DATA_DIR}" -type f -name '*.json.gz' -print0 | sort -z)

echo "[decompress-dataGraphs] Restored ${count} file(s)."
