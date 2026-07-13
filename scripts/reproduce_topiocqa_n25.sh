#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${1:-${MAPMATCHED_TOPIOCQA_PATH:-}}"
PROFILE="topiocqa-n25-minilm-knn-full"
REPORT_DIR="${MAPMATCHED_REPORT_DIR:-$ROOT_DIR/reports/$PROFILE}"

if [[ -z "$DATA_PATH" ]]; then
  echo "usage: $0 /path/to/topiocqa_valid.jsonl" >&2
  echo "or set MAPMATCHED_TOPIOCQA_PATH" >&2
  exit 2
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "TopiOCQA data file not found: $DATA_PATH" >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

python3 -m mapmatched.eval \
  --profile "$PROFILE" \
  --benchmark topiocqa \
  --tier micro \
  --data-path "$DATA_PATH" \
  --conversation-limit 25 \
  --embedder sentence-transformers \
  --st-model sentence-transformers/all-MiniLM-L6-v2 \
  --graph-source knn \
  --knn-neighbors 10 \
  --ranking-mode full \
  --candidate-limit 100 \
  --recall-k 100 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42 \
  --output "$REPORT_DIR/report.json" \
  --markdown-output "$REPORT_DIR/report.md"

echo "JSON report: $REPORT_DIR/report.json"
echo "Markdown report: $REPORT_DIR/report.md"
