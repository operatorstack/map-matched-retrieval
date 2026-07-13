# Changelog

All notable changes to this project are documented here.

## Unreleased

- Add an opt-in Gemini conversational query rewrite baseline for TREC CAsT,
  configured by `GEMINI_API_KEY`, with paired pointwise comparisons and
  model/prompt provenance in evaluation reports.
- Replace unreachable GitHub Actions commit references with the current
  `checkout@v7` and `setup-python@v6` major tags, restore Python 3.11 NumPy
  compatibility, and keep the base-package job dependency-free.
- Add paired conversation-level bootstrap intervals for method nDCG@3 deltas,
  preserve conversation IDs in reports, and fix the aggregate `all`-slice
  bootstrap interval.
- Add a pinned TopiOCQA n=25 MiniLM/kNN reproduction script with dataset
  checksum, selected conversation IDs, model, graph, package, and git provenance.
- Cache repeated query embeddings in the eval provider and reuse one bounded
  shortest-path search across all targets for a graph source.
- Build embedding kNN graphs with blockwise NumPy top-k selection instead of
  Python-sorting every corpus pair, keeping large judged corpora tractable.
- Bound graph shortest-path caching by source and bypass graph searches when
  `transition_weight=0`, preventing pointwise evaluation from materializing an
  all-pairs distance cache.
- Accelerate `KNNGraph` shortest paths with SciPy's compiled sparse-graph
  implementation and compact dense-distance cache when available while
  retaining the standard-library fallback.
- Vectorize evaluation retrieval scores with NumPy and retain a bounded query
  score cache, removing per-dimension Python loops from full-corpus baselines.
- Publish the reproducible TopiOCQA n=25 MiniLM/kNN micro-corpus result:
  map-matched β=1.0 lifts follow-up nDCG@3 by `+0.084` with paired 95% CI
  `[+0.046, +0.128]`; two runs produced byte-identical reports.
- Bootstrap confidence intervals: conversation-level percentile bootstrap for
  nDCG@3 per slice (`--bootstrap-samples`, default 0; use 1000 for publishable
  runs). CIs appear in JSON reports and the markdown table.
- Full-ranking eval: the decoder now exposes the current turn's candidates
  ranked by trajectory (final-turn cumulative) score via `decode_ranked` and
  `RetrievalResult.candidate_ranking` / `CandidateScore`. The eval re-ranks the
  whole candidate window by that score (`--ranking-mode full`, default) instead
  of only hoisting the decoded chunk to rank 1 (`--ranking-mode rank1`), so
  nDCG@k reflects the trajectory across the top-k. With `transition_weight=0`
  the re-rank reproduces the raw similarity order (pointwise parity).
- Structured `SectionGraph`: `Passage.group_key` carries a structural key (the
  TopiOCQA loader sets it to the Wikipedia article title); `build_section_graph`
  connects same-key passages and clamps cross-group distance to
  `maximum_distance` (a coherent-drill / expensive-jump prior). Selectable with
  `--graph-source {knn,section}`.
- `--candidate-limit` CLI flag (default 100) sizes the re-rankable window.
- Add a numpy/sentence-transformers mypy override (numpy 2.5 stubs use 3.12
  `type` syntax; the project's mypy targets 3.10).
- Fix the TopiOCQA loader: read the released JSON/JSONL directly (HuggingFace
  `datasets` dropped the custom dataset script) via `data_path` / the
  `MAPMATCHED_TOPIOCQA_PATH` env var; drop the unused `datasets` dependency.
- Eval slice fairness: one shared entropy threshold (from pointwise trace
  entropies) and one provider/graph build per run so all methods are compared
  on the same follow-up vs standalone slices.
- Fix the TREC CAsT 2019 loader: use the correct ir-datasets id
  `trec-cast/v1/2019/judged`, load real passage text from the collection
  `docs_store()` (previously the doc id was used as the text), read
  `raw_utterance` / `manual_rewritten_utterance`, populate resolved queries,
  install TREC CAR support, and stop retrying a failed docstore build per passage.
- Add an optional `SentenceTransformerEmbedder` (extra: `[st]`) and an
  `--embedder {hash,sentence-transformers}` CLI flag so eval runs can use real
  semantic embeddings instead of the deterministic hash fixture. Both embedders
  run locally — no API tokens.
- Add optional `mapmatched.eval` harness with TopiOCQA and TREC CAsT 2019 micro
  loaders, entropy-sliced H1/H0 reporting, β ablations, and baselines (pointwise,
  history concat, Maximal Marginal Relevance, resolved oracle).
- Add deterministic hash embedder, synthetic CI fixtures, eval CLI, and
  `examples/05_eval_demo.py`.
- Add an optional FAISS candidate provider with injected query embeddings,
  explicit similarity/distance score handling, and index-to-chunk validation.
- Add deterministic weighted cosine k-nearest-neighbor corpus graph construction
  from caller-supplied embeddings.
- Add dependency-free terminal rendering for retrieval traces.
- Add Python-version CI, strict quality checks, wheel verification, runnable
  examples, and an explicit composable-model-graph parity job.
- Make the optional CMG example exit successfully when CMG is not installed.

## 0.1.0 - 2026-07-12

- Add the first Python vertical slice with typed retrieval, path, and trace models.
- Add dependency-free full Viterbi and fixed-lag decoders behind a mapmatched-owned
  protocol.
- Add an optional composable-model-graph backend boundary without a runtime or Git
  dependency.
- Add deterministic bounded in-memory graph distance, neighborhoods, clamping, and
  caching.
- Add z-score, centered, and unmodified score handling plus stable softmax entropy.
- Add provider-backed and direct-candidate sessions, prior-revision reporting, and
  deterministic context expansion.
- Add runnable examples, theory documentation, and deterministic tests.
