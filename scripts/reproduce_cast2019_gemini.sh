#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="cast2019-gemini35flash-minilm-knn-full"
REPORT_DIR="${MAPMATCHED_REPORT_DIR:-$ROOT_DIR/reports/$PROFILE}"
GEMINI_MODEL="${MAPMATCHED_GEMINI_MODEL:-gemini-3.5-flash}"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY must be set" >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

python3 -m mapmatched.eval \
  --profile "$PROFILE" \
  --benchmark cast2019 \
  --tier micro \
  --conversation-limit 50 \
  --embedder sentence-transformers \
  --st-model sentence-transformers/all-MiniLM-L6-v2 \
  --graph-source knn \
  --knn-neighbors 10 \
  --ranking-mode full \
  --candidate-limit 100 \
  --recall-k 100 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42 \
  --include-gemini-rewrite \
  --gemini-model "$GEMINI_MODEL" \
  --output "$REPORT_DIR/report.json" \
  --markdown-output "$REPORT_DIR/report.md"

echo "JSON report: $REPORT_DIR/report.json"
echo "Markdown report: $REPORT_DIR/report.md"
