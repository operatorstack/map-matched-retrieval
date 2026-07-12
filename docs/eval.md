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
python -m mapmatched.eval --benchmark topiocqa --conversation-limit 50
```

Loads the Hugging Face validation split and builds a micro-corpus from gold
passages and additional answers.

### TREC CAsT 2019 (micro)

```console
python -m mapmatched.eval --benchmark cast2019 --include-resolved-oracle
```

Loads train topics and qrels via ir-datasets. Passage text defaults to doc IDs
in micro mode; supply a real corpus mapping for full runs.

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
