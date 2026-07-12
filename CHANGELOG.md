# Changelog

All notable changes to this project are documented here.

## Unreleased

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
