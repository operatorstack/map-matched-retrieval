# Evaluation harness

Map-matched retrieval makes a narrow claim: it should lift **underspecified
follow-up turns** without materially harming **sharp standalone turns**. M2 adds
an optional evaluation harness behind `pip install map-matched-retrieval[eval]`.

## Install

```console
pip install map-matched-retrieval[eval,graph]
```

The harness uses NumPy for kNN graph construction and optional Hugging Face /
ir-datasets loaders for benchmark metadata. It does **not** download embedding
models.

## Tiers

| Tier | Purpose | Network |
| --- | --- | --- |
| **A — synthetic** | CI fixtures; proves metrics, slices, baselines | No |
| **B — micro** | Real TopiOCQA / CAsT structure with gold/judged passages only | Yes |
| **C — full** | Full Wikipedia / CAR+MARCO corpora (documented offline workflow) | Yes + heavy infra |

## Quick start (Tier A)

```console
python examples/05_eval_demo.py
```

Or via the module CLI:

```console
python -m mapmatched.eval --benchmark synthetic --output eval-report.json
```

## Claim and slices

Every turn is classified using trace **emission entropy**:

- **Standalone slice** — turn index 0, or entropy below the median threshold
- **Follow-up slice** — later turns with entropy at or above the threshold

Results are reported separately for each slice. The pass/fail gate compares the
best map-matched configuration against the β=0 pointwise baseline:

- Follow-up lift must exceed `follow_up_min_delta` (default 0.0)
- Standalone nDCG@3 must stay within `standalone_tolerance` (default 0.02)

## Methods and baselines

| Method | Description |
| --- | --- |
| `pointwise` | `transition_weight=0` — independent per-turn top-1 |
| `mapmatched` | Full trajectory decoder with configurable β |
| `history_concat` | Dense retrieval over concatenated query history |
| `maximal_marginal_relevance` | Per-turn MMR re-ranking (not map-matched retrieval) |
| `resolved_oracle` | CAsT resolved utterances (upper bound) |

## Benchmarks

### TopiOCQA (micro)

```console
# download a split first (HF datasets dropped the custom dataset script):
#   https://huggingface.co/datasets/McGill-NLP/TopiOCQA -> data/topiocqa_valid.jsonl
python -m mapmatched.eval --benchmark topiocqa \
    --data-path topiocqa_valid.jsonl --conversation-limit 25 \
    --embedder sentence-transformers --graph-source section --ranking-mode full
```

Reads the released JSON/JSONL directly (`--data-path` or `MAPMATCHED_TOPIOCQA_PATH`)
and builds a micro-corpus from gold passages and additional answers. The section
graph keys on the Wikipedia article title. Note TopiOCQA is topic-switch heavy,
so it stresses the standalone (H0) side as much as the follow-up (H1) side.

### TREC CAsT 2019 (micro)

```console
python -m mapmatched.eval --benchmark cast2019 --embedder sentence-transformers
```

Uses ir-datasets id `trec-cast/v1/2019/judged`. Real passage text comes from the
collection `docs_store()` (MS MARCO + TREC CAR), which ir-datasets downloads on
first use — **multi-GB**, so a full run is heavy and best done on a workstation.
Without the collection the loader degrades to using doc ids as passage text
(metrics not meaningful). CAsT's drill-down follow-ups are the fairer test for
the follow-up-lift claim than TopiOCQA's topic switches.

## Graph source and ranking mode

- `--graph-source knn` (default) builds the embedding-kNN fallback graph;
  `--graph-source section` builds a structured graph from `Passage.group_key`
  (same key = adjacent; different key = clamped `maximum_distance`).
- `--ranking-mode full` (default) re-ranks the candidate window by trajectory
  score; `--ranking-mode rank1` reproduces the legacy decoded-chunk-first order.
- `--candidate-limit` (default 100) sizes the re-rankable window; the gold
  passage must be within it to be re-ranked (otherwise recall bounds the score).

## Bootstrap confidence intervals

Use `--bootstrap-samples` to resample conversations and compute 95% percentile
CIs for nDCG@3 on each slice. Disabled by default (`0`) for fast smoke runs;
use `1000` for publishable numbers. `--bootstrap-seed` (default 42) keeps runs
reproducible.

```console
python -m mapmatched.eval --benchmark synthetic \
    --bootstrap-samples 200 --output eval-report.json
```

The markdown table adds an `nDCG@3 95% CI` column; JSON reports include
`ndcg_at_3_ci` as `[lower, upper]` on each slice.

## Reproducing headline numbers

The README headline table uses **Tier B dev-slice** results with a
caller-supplied embedder. The built-in `DeterministicHashEmbedder` is for tests
and smoke runs only.

For publishable numbers:

1. Choose an embedding model and implement `QueryEmbedder` / `PassageEmbedder`.
2. Build a micro-corpus or full corpus index.
3. Run the ablation grid and record JSON + markdown output.
4. Paste the markdown table into README with the embedder and tier noted.

## Output

- `eval-report.json` — full per-turn and per-slice metrics
- Markdown table — paste-ready README fragment with claim check summary
