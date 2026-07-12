# Map-Matched Retrieval

Map-matched retrieval treats a multi-turn conversation as trajectory estimation.
Each turn's retrieval candidates are states in a trellis, corpus-graph distance is
the transition cost, and decoding returns the maximum-score path rather than an
independent winner for every turn.

It is a small retrieval decoder and trace library. It is not a vector store,
embedder, agent framework, query rewriter, graph builder for large corpora, or
context-ranking algorithm. In particular, the decoded MAP chunk and expanded
context are separate outputs; context expansion does not imply multiple decoded
paths.

## Install

Python 3.10 or newer is required.

```console
pip install map-matched-retrieval
```

The core has no runtime dependencies. Install `map-matched-retrieval[graph]` to
build a k-nearest-neighbor corpus graph from embeddings, or
`map-matched-retrieval[faiss]` to add both graph construction and FAISS
retrieval. Until composable-model-graph has a stable release, users who already
have a compatible installation may explicitly choose `CMGDecoder`; mapmatched
never installs or exposes its types.

## Direct candidates

```python
from mapmatched import InMemoryCorpusGraph, MapMatchedRetriever, ScoredCandidate

graph = InMemoryCorpusGraph.from_edges(
    [("overview", "details"), ("details", "failure-modes")],
    maximum_distance=4.0,
)
session = MapMatchedRetriever(
    graph,
    transition_weight=0.7,
    score_normalization="zscore",
).session()

first = session.retrieve_candidates([
    ScoredCandidate("overview", 0.82),
    ScoredCandidate("failure-modes", 0.78),
])
second = session.retrieve_candidates([
    ScoredCandidate("details", 0.63),
    ScoredCandidate("failure-modes", 0.65),
])

print(second.chunk_id)
print(second.context_chunk_ids)
print(second.trace.to_json(indent=2))
```

For an existing retriever, implement `CandidateProvider.candidates(query, limit)`
and pass it as `provider=...`; then call `session.retrieve(query)`.

## FAISS session

Mapmatched accepts caller-supplied embeddings but does not choose or download an
embedding model:

```python
from mapmatched import FAISSProvider, KNNGraph, MapMatchedRetriever

graph = KNNGraph.from_embeddings(
    chunk_ids,
    chunk_embeddings.tolist(),
    neighbor_count=10,
)
provider = FAISSProvider(
    faiss_index,
    chunk_ids,
    embed_query=my_embedding_function,
)
session = MapMatchedRetriever(
    graph,
    provider=provider,
    transition_weight=0.5,
).session()

session.retrieve("How does token refresh work?")
result = session.retrieve("What happens when it expires?")

print(result.chunk_id)
print(result.context_chunk_ids)
print(result.trace.render())
```

Use `score_mode="similarity"` for inner-product or cosine indexes. Use
`score_mode="distance"` for L2 indexes so lower FAISS distances become higher
retrieval scores. `FAISSProvider` verifies that index rows and chunk IDs stay
aligned. Normalize indexed and query vectors before using an inner-product index
as cosine search.

`KNNGraph` normalizes embeddings and uses weighted cosine distance. Neighbor ties
are resolved by chunk ID, identical vectors receive a small positive edge
distance, and disconnected or over-cutoff paths still clamp to
`maximum_distance`.

## Behavior and choices

- The objective is `emission_weight * normalized_score - transition_weight *
  graph_distance`, accumulated over the path.
- `zscore` is the safe normalization default. It makes each turn's score scale
  comparable and maps a constant candidate set to zeros without division by zero.
  `center` removes only the per-turn mean; `none` preserves the provider's scale
  when scores are already calibrated. Raw and normalized scores remain in traces.
- Entropy is computed from a numerically stable softmax of weighted normalized
  emissions. The margin is the chosen normalized emission minus the best
  alternative; it can be negative when graph coherence overrules pointwise rank.
- `transition_weight=0` exactly reproduces deterministic per-turn argmax.
  Candidate input order resolves score ties.
- Full decoding can revise any prior turn when evidence arrives. Set
  `fixed_lag=L` to commit a turn after `L` later turns. The trace reports both
  revised indices and the committed boundary.
- Context order is deterministic: current decoded chunk, graph neighbors ordered
  by distance and ID, then current candidates in provider order, with stable
  deduplication.

## Suitable uses and limits

This slice is intended for conversational documentation retrieval, linked
knowledge bases, section graphs, and other corpora where local movement has
meaning. Graph quality bounds retrieval quality. The in-memory graph uses bounded
Dijkstra searches and a distance cache; unreachable and beyond-cutoff pairs clamp
to `maximum_distance`. It is suitable for bounded candidate sets and modest
graphs, not an all-pairs graph service.

Candidate providers should return a small, high-recall set. Decoding costs
`O(turns * candidates²)` graph lookups, reduced in practice by caching. A
standalone query in a long session can be over-smoothed by prior context; start a
new session for unrelated queries or reduce the transition weight. This library
deliberately has no adaptive weighting, asynchronous API, or multiple-path
decoding in the core package.

## Evaluation (optional)

Install the eval harness to run entropy-sliced benchmark reports:

```console
pip install map-matched-retrieval[eval,graph]
python examples/05_eval_demo.py
```

See [`docs/eval.md`](docs/eval.md) for TopiOCQA / TREC CAsT micro-corpus runs,
ablation grids, and reproduction steps. The built-in hash embedder is for tests
only; published numbers require a caller-supplied embedding model.

## Benchmark results (dev slice)

| Benchmark | Slice | Method | β | nDCG@3 | nDCG@5 | Recall | Δ vs β=0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | follow_up | mapmatched | 0.50 | TBD | TBD | TBD | TBD |
| synthetic | standalone | mapmatched | 0.50 | TBD | TBD | TBD | TBD |
| topiocqa (micro) | follow_up | mapmatched | 0.50 | TBD | TBD | TBD | TBD |
| cast2019 (micro) | follow_up | mapmatched | 0.50 | TBD | TBD | TBD | TBD |

Run `python -m mapmatched.eval --benchmark synthetic` to populate the synthetic
row locally. Tier B rows require network access and a real embedder for
publishable values.

See [`docs/theory.md`](docs/theory.md) for the objective and semantics and
[`examples`](examples) for complete runs.
