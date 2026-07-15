# Evaluation harness

Map-matched retrieval makes a narrow claim: it should lift **underspecified
follow-up turns** without materially harming **sharp standalone turns**. The
evaluation harness measures that claim without making conversational RAG the
library's API boundary.

## Install

```console
python -m pip install -e ".[eval,graph,st]"
```

The harness uses NumPy for kNN graph construction and optional Hugging Face /
ir-datasets loaders for benchmark metadata. It does **not** download embedding
models. Install `.[eval,graph,st,gemini]` only when running the Gemini rewrite
baseline.

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
| `gemini_rewrite` | Gemini rewrites each turn into a standalone query before dense retrieval |
| `maximal_marginal_relevance` | Per-turn MMR re-ranking (not map-matched retrieval) |
| `resolved_oracle` | CAsT resolved utterances (upper bound) |

## Benchmarks

### TopiOCQA (micro)

Download `data/topiocqa_valid.jsonl` from the
[TopiOCQA dataset repository](https://huggingface.co/datasets/McGill-NLP/TopiOCQA),
then run the pinned profile:

```console
./scripts/reproduce_topiocqa_n25.sh data/topiocqa_valid.jsonl
```

Reads the released JSON/JSONL directly (`--data-path` or `MAPMATCHED_TOPIOCQA_PATH`)
and builds a micro-corpus from gold passages and additional answers. The pinned
profile selects the first 25 conversations in file order, records the file
SHA-256 and selected IDs, uses `sentence-transformers/all-MiniLM-L6-v2`, a
10-neighbor kNN graph, full ranking, a 100-candidate window, and 1,000 bootstrap
draws with seed 42.

TopiOCQA is topic-switch heavy, so it stresses the standalone side as much as the
follow-up side. It is licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); the
dataset is not redistributed by this repository. These runs are micro-corpus
experiments, not full-Wikipedia retrieval.

The pinned run at git revision `60a9d694b807c3eb49da2a00743c6f1a05d52e6d`
produced byte-identical reports twice:

| Slice | Method | nDCG@3 | Delta vs pointwise | Paired delta 95% CI |
| --- | --- | ---: | ---: | ---: |
| Follow-up | Map-matched β=0.5 | 0.195 | +0.045 | [+0.018, +0.077] |
| Follow-up | Map-matched β=1.0 | 0.234 | +0.084 | [+0.046, +0.128] |
| Follow-up | MMR | 0.151 | +0.001 | [+0.000, +0.003] |
| Standalone | Map-matched β=1.0 | 0.373 | +0.031 | [+0.009, +0.055] |

The complete aggregate table and provenance are committed in
[`results/topiocqa_n25_minilm_knn.md`](../results/topiocqa_n25_minilm_knn.md).
Positive intervals support the claim on this fixed 25-conversation micro-corpus;
they do not establish full-corpus or cross-benchmark generalization.

### TREC CAsT 2019 (micro)

```console
export GEMINI_API_KEY="..."
python -m pip install -e ".[eval,graph,st,gemini]"
./scripts/reproduce_cast2019_gemini.sh
```

Uses ir-datasets id `trec-cast/v1/2019/judged`. Real passage text comes from the
collection `docs_store()` (MS MARCO + TREC CAR), which ir-datasets downloads on
first use — **multi-GB**, so a full run is heavy and best done on a workstation.
Without the collection the loader degrades to using doc ids as passage text
(metrics not meaningful). CAsT's drill-down follow-ups are the fairer test for
the follow-up-lift claim than TopiOCQA's topic switches.

The script adds `gemini_rewrite` to the normal ablation grid. It sends each raw
utterance and its prior user utterances to `gemini-3.5-flash` with minimal
thinking, retrieves with the returned standalone query, and compares it with both
pointwise retrieval and CAsT's manual `resolved_oracle`. `GEMINI_API_KEY` is read
from the environment and is never written to reports. Reports record the Gemini
model and prompt version. The baseline is opt-in because it makes one paid,
networked model request per selected turn; `--conversation-limit` bounds those
requests. Hosted-model output is not immutable across model revisions.
Successful rewrites are checkpointed in `rewrites.json`, so rerunning the profile
resumes after transient API failures instead of repeating completed requests.

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
CIs for nDCG@3 on each slice. Method comparisons resample the same conversations
for treatment and pointwise retrieval, producing a paired CI on the nDCG@3
delta. Disabled by default (`0`) for fast smoke runs; use `1000` for reported
numbers. `--bootstrap-seed` (default 42) keeps runs reproducible.

```console
python -m mapmatched.eval --benchmark synthetic \
    --bootstrap-samples 200 --output eval-report.json
```

The markdown table includes absolute and paired-delta 95% CIs. JSON reports
include method-level `ndcg_at_3_ci` values and explicit `comparisons` with
`ndcg_at_3_delta_ci`. The claim gate remains based on configured point-estimate
thresholds; a paired interval excluding zero is the uncertainty check.

## Reproducing headline numbers

The README headline table uses **Tier B dev-slice** results with a
caller-supplied embedder. The built-in `DeterministicHashEmbedder` is for tests
and smoke runs only.

The TopiOCQA script writes full JSON and markdown reports under
`reports/topiocqa-n25-minilm-knn-full/`. Generated reports are ignored because
they contain machine-run detail; committed headline values must include the
profile, dataset SHA-256, model, graph settings, conversation count, and paired
interval.

## Output

- `eval-report.json` — full per-turn and per-slice metrics
- Markdown table — paste-ready README fragment with claim check summary
