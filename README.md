# Map-Matched Retrieval

[![CI](https://github.com/operatorstack/map-matched-retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/operatorstack/map-matched-retrieval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Typed](https://img.shields.io/badge/typing-mypy_strict-blue)](https://mypy.readthedocs.io/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)](#project-status)

**Trajectory-aware retrieval for multi-turn conversations.**

Most retrievers score every turn independently. Map-matched retrieval instead
decodes the conversation as a path through a corpus graph: retrieval scores say
where the conversation might be, while graph distance says how plausible each
move is.

The result is a small, retriever-agnostic Python library that sits between your
candidate provider and your RAG pipeline. It returns both the selected chunks and
an inspectable trace showing the emission/transition trade-off behind each
decision.

> [!IMPORTANT]
> This project is an **alpha**. The dependency-free core, typed public API,
> FAISS adapter, graph builders, traces, examples, and evaluation harness are
> implemented and tested. Benchmark coverage and third-party adapters are still
> being expanded, so APIs may evolve before 1.0.

## Why trajectory decoding?

| Pointwise retrieval | Map-matched retrieval |
| --- | --- |
| Chooses the highest-scoring chunk at each turn | Chooses the highest-scoring path across turns |
| Discards conversational location | Carries location through a corpus graph |
| Provides a score for the current result | Provides an emission/transition trace |
| Can jump on an ambiguous follow-up | Penalizes implausible jumps while preserving strong evidence |

For candidates \(x_t\) at turn \(t\), the decoder maximizes:

```text
Σ emission_weight × normalized_score(query_t, x_t)
  − transition_weight × graph_distance(x_t−1, x_t)
```

Setting `transition_weight=0` exactly recovers deterministic pointwise retrieval.
Full Viterbi decoding can revise earlier turns; fixed-lag decoding provides a
bounded-revision streaming mode.

## Highlights

- Standard-library-only core with no runtime dependencies
- Typed API checked with strict mypy
- Full and fixed-lag Viterbi decoders
- Bring-your-own retriever through a minimal `CandidateProvider` protocol
- In-memory weighted graphs and embedding-derived kNN graphs
- Optional FAISS adapter for cosine, inner-product, and L2 indexes
- JSON and terminal traces with scores, graph costs, entropy, and revisions
- Reproducible evaluation harness with ablations, baselines, and bootstrap CIs
- CI across Python 3.10, 3.11, and 3.12

## Install

Python 3.10 or newer is required. The package is not yet published to PyPI;
install the public alpha from source:

```console
git clone https://github.com/operatorstack/map-matched-retrieval.git
cd map-matched-retrieval
python -m pip install -e .
```

Optional extras keep the base package small:

```console
python -m pip install -e ".[graph]"
python -m pip install -e ".[faiss]"
python -m pip install -e ".[eval,graph]"
python -m pip install -e ".[st]"
```

| Extra | Adds |
| --- | --- |
| `graph` | Embedding-derived kNN graphs |
| `faiss` | kNN graphs and the FAISS candidate provider |
| `eval` | Benchmark loaders, baselines, metrics, and reports |
| `gemini` | Gemini conversational query rewrite baseline |
| `st` | Sentence-transformer embeddings for evaluation |

## Quickstart

Supply scored candidates directly to see the decoder without a vector database:

```python
from mapmatched import InMemoryCorpusGraph, MapMatchedRetriever, ScoredCandidate

graph = InMemoryCorpusGraph.from_edges(
    [("hmm", "noise"), ("noise", "road-jumps")],
    maximum_distance=4.0,
)
session = MapMatchedRetriever(
    graph,
    score_normalization="none",
    transition_weight=1.0,
).session()

session.retrieve_candidates(
    [
        ScoredCandidate("hmm", 5.0),
        ScoredCandidate("noise", 1.0),
        ScoredCandidate("road-jumps", 0.0),
    ]
)
result = session.retrieve_candidates(
    [
        ScoredCandidate("hmm", 1.0),
        ScoredCandidate("noise", 3.0),
        ScoredCandidate("road-jumps", 3.5),
    ]
)

print(result.chunk_id)
print(result.context_chunk_ids)
print(result.trace.render())
```

```text
noise
('noise', 'hmm', 'road-jumps')
```

The pointwise winner on the second turn is `road-jumps`, but the decoder selects
the adjacent `noise` chunk because its slightly lower emission score is offset by
a shorter graph move. The trace records the raw and normalized emissions, graph
distance, weighted transition cost, entropy, cumulative score, and any revisions
to prior turns.

Run the complete example:

```console
python examples/01_direct_candidates.py
```

## Use an existing retriever

Implement the two-argument candidate protocol and pass the provider into
`MapMatchedRetriever`:

```python
from collections.abc import Sequence

from mapmatched import MapMatchedRetriever, ScoredCandidate


class MyCandidateProvider:
    def candidates(self, query: str, limit: int) -> Sequence[ScoredCandidate]:
        return my_retriever.search(query, limit=limit)


session = MapMatchedRetriever(
    corpus_graph,
    provider=MyCandidateProvider(),
    candidate_limit=20,
    transition_weight=0.5,
).session()

session.retrieve("How does token refresh work?")
result = session.retrieve("What happens when it expires?")
```

Providers return `ScoredCandidate` values with higher scores meaning better
matches. See
[`examples/02_custom_provider.py`](examples/02_custom_provider.py) for a complete
adapter and [`examples/04_faiss_session.py`](examples/04_faiss_session.py) for
FAISS with caller-supplied embeddings.

## Architecture

```text
query ──> CandidateProvider ──> scored candidate trellis
                                      │
corpus structure ──> CorpusGraph ─────┤
                                      ▼
                              trajectory decoder
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  RetrievalResult           RetrievalTrace
              chunk + ranked context   scores + costs + revisions
```

Mapmatched does not replace a vector store, choose an embedding model, rewrite
queries, or run an agent framework. It owns one narrow boundary: graph-aware
sequential decoding over candidate sets. The decoded MAP chunk and expanded
context are separate outputs.

## Evaluation

The evaluation harness reports nDCG@3/5 and Recall@k separately for ambiguous
follow-up turns and sharp standalone turns. It includes pointwise, history
concatenation, optional Gemini query rewriting, Maximal Marginal Relevance, and
resolved-query baselines, plus conversation-level percentile bootstrap confidence
intervals. Method deltas use paired resampling of the same conversations.

```console
python -m mapmatched.eval \
    --benchmark synthetic \
    --bootstrap-samples 200
```

The synthetic benchmark is a deterministic smoke test, not research evidence.
The pinned TopiOCQA micro-corpus profile uses 25 conversations, MiniLM
embeddings, a 10-neighbor kNN graph, full candidate ranking, and 1,000 paired
conversation-level bootstrap draws:

| Slice | Method | nDCG@3 | Delta vs pointwise | Paired delta 95% CI |
| --- | --- | ---: | ---: | ---: |
| Follow-up | Pointwise | 0.150 | +0.000 | — |
| Follow-up | Map-matched β=0.5 | 0.195 | +0.045 | [+0.018, +0.077] |
| Follow-up | Map-matched β=1.0 | 0.234 | +0.084 | [+0.046, +0.128] |
| Follow-up | MMR | 0.151 | +0.001 | [+0.000, +0.003] |
| Standalone | Map-matched β=1.0 | 0.373 | +0.031 | [+0.009, +0.055] |

Both runs produced byte-identical reports. The positive paired intervals are
evidence for this fixed micro-corpus, not a full-Wikipedia or cross-benchmark
claim. Structured section graphs also underperform on topic-switch-heavy
TopiOCQA, an important negative result rather than a hidden one. See the
[`committed result`](results/topiocqa_n25_minilm_knn.md) and
[`evaluation guide`](docs/eval.md) for provenance, all baselines, limitations,
and reproduction commands.

Reproduce the pinned n=25 MiniLM/kNN profile after downloading the validation
split:

```console
python -m pip install -e ".[eval,graph,st]"
./scripts/reproduce_topiocqa_n25.sh data/topiocqa_valid.jsonl
```

## Design choices and limits

- Per-turn z-score normalization is the safe default; `center` and `none` are
  available when provider scores already have a meaningful scale.
- Candidate providers should return a small, high-recall set. Decoding requires
  `O(turns × candidates²)` graph-distance lookups, reduced by distance caching.
- Graph quality bounds retrieval quality. The in-memory graph uses bounded
  Dijkstra search and clamps unreachable or over-cutoff distances.
- A long session can over-smooth unrelated queries. Start a new session or lower
  `transition_weight` when the topic changes.
- The alpha does not yet provide adaptive weighting, an asynchronous API,
  multiple-path decoding, or managed graph infrastructure.

The strongest current use cases are conversational documentation retrieval,
linked knowledge bases, section graphs, and other corpora where local movement
has semantic meaning.

## Project status

- [x] Dependency-free decoder, graph protocol, and retrieval session
- [x] Full and fixed-lag decoding with inspectable traces
- [x] kNN graph builder and FAISS candidate provider
- [x] Synthetic, TopiOCQA, and TREC CAsT evaluation paths
- [x] Full candidate ranking and conversation-level bootstrap CIs
- [ ] Lock reproducible full-corpus benchmark results
- [ ] Add LangChain/LlamaIndex and hosted vector-store adapters
- [ ] Add graph construction tooling for larger corpora
- [ ] Stabilize the public API for a non-alpha release

See [`PLAN.md`](PLAN.md) for the longer roadmap and [`CHANGELOG.md`](CHANGELOG.md)
for the implementation history.

## Documentation and examples

- [`docs/theory.md`](docs/theory.md) — objective, normalization, graph distance,
  and decoding semantics
- [`docs/eval.md`](docs/eval.md) — benchmark tiers, baselines, and reproduction
- [`examples/01_direct_candidates.py`](examples/01_direct_candidates.py) — core
  decoder without external dependencies
- [`examples/02_custom_provider.py`](examples/02_custom_provider.py) — custom
  candidate provider
- [`examples/03_cmg_inspectable_run.py`](examples/03_cmg_inspectable_run.py) —
  optional composable-model-graph backend
- [`examples/04_faiss_session.py`](examples/04_faiss_session.py) — FAISS and kNN
  integration
- [`examples/05_eval_demo.py`](examples/05_eval_demo.py) — offline synthetic eval

## Development

```console
python -m pip install -e ".[dev,faiss,eval,graph]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Issues and focused pull requests are welcome. For behavior changes, include tests
and update the changelog so design decisions remain visible.
