# Map-Matched Retrieval — Build Plan

Conversational RAG as trajectory estimation on a corpus graph.
Newson–Krumm HMM map matching transported to retrieval: the conversation is a
noisy track, the corpus graph is the road network, retrieval is the most likely
*path* — not T independent nearest-neighbour fixes.

Source: `map matched retrieval.pdf` (working note). This plan turns the note
into a usable open-source library.

---

## 1. Positioning

**What it is**

- A retrieval **decoder** that sits on top of any existing RAG stack.
- Input: per-turn candidate sets with similarity scores (from *your* retriever)
  plus a corpus graph (built by us or supplied by you).
- Output: the decoded trajectory `x*_1:T` — the chunk the conversation is "at"
  each turn, optionally with its graph neighbourhood — and **the trace**: the
  per-turn emission-vs-transition split that shows *why* each chunk was chosen.
- The trace is the flagship artifact. People should adopt the library for the
  lift on follow-up turns and keep it for the explainability.

**What it is not**

- Not a vector store. Not an embedder. Not a framework, agent library, or
  LangChain competitor. We wrap retrievers; we never replace them.
- Not "stateful retrieval in general" — the narrow claim is
  map-matching-over-a-corpus (see novelty boundary in the note).

**Design constraints** (inherited from the operatorstack house style)

- Small API. Theory informs design; it is never the price of admission.
- Trace/evaluation-first: every decode is inspectable.
- Core stays standard-library-only; everything vendor-specific is an
  optional adapter behind a protocol.
- CHANGELOG.md as a reasoning trail, one entry per change.

---

## 2. Naming

- The common abbreviation for this project collides with Maximal Marginal
  Relevance in IR, so always use the full map-matched retrieval name.
- Repo/package: `map-matched-retrieval`; Python import: `mapmatched`.
- GitHub org: `operatorstack` (alongside composable-model-graph / modelgraph).

---

## 3. Architecture (layers, inside → out)

### 3.1 Core decoder — `mapmatched.core` (standard library only, no I/O)

The algorithm from the note, as typed operations over candidate sequences:

- **Trellis**: per turn t, states `S_t` = top-M candidates with scores
  `s(q_t, c)`.
- **Objective** (linear-chain energy, CRF reading — not strict HMM):
  `sum_t λ·s(q_t, x_t) − sum_{t≥2} β·d_G(x_{t−1}, x_t)`.
- **Decoders**:
  - `viterbi(trellis, graph, lam, beta)` — full max-score path, O(T·M²),
    backpointers retained.
  - `fixed_lag(trellis, graph, lam, beta, lag=L)` — causal/streaming variant
    (forward filtering with an L-turn commit delay). On a line graph this
    reduces to a leaky integrator — the cheapest sanity build and first test.
- **Degenerate modes are first-class flags** (they are the ablations):
  - `beta=0` → vanilla per-turn top-1 (the clean baseline).
  - `beta=inf` → graph geodesic, ignores evidence (over-smoothing bound).
- **Emission entropy** `H_t` = entropy of softmax(λ·s(q_t,·)) over `S_t`,
  computed for every turn and exposed in the trace — this is the statistic
  that stratifies "underspecified follow-up" (high H) from "sharp standalone"
  (low H), i.e. the H1/H0 slicing.

Gotchas to handle in core (documented, tested):

- **Score comparability across turns**: a single λ assumes `s` is on a stable
  scale; normalise per-turn scores (z-score or temperature softmax) and
  document the choice. Cosine vs inner-product matters here.
- **Disconnected graphs / unreachable pairs**: `d_G = ∞` clamps to a max
  penalty `d_max` so a jump is expensive but never impossible.
- **Re-decode semantics**: full Viterbi each turn may retroactively revise
  past `x*_t`. That is correct behaviour; `fixed_lag` exists for consumers who
  need stability. Both documented with the trade-off.

### 3.2 Graph layer — `mapmatched.graph`

Protocol:

```python
class CorpusGraph(Protocol):
    def distance(self, a: ChunkId, b: ChunkId) -> float: ...   # geodesic d_G
    def neighborhood(self, c: ChunkId, radius: int) -> list[ChunkId]: ...
```

Builders (quality order per the note: citation/section > kNN fallback):

- `CitationGraph.from_edges(edges)` — explicit links (citations, hyperlinks).
- `SectionGraph.from_documents(docs)` — section adjacency within/between docs.
- `KNNGraph.from_embeddings(E, k)` — the always-present fallback.
- `NetworkXGraph(g)` — bring-your-own adapter.

Distance computation is the O(T·M²) multiplier, so:

- BFS with cutoff `D_max` (distances beyond cutoff clamp to `d_max`) as the
  default; per-turn-pair batched lookups over the `S_{t−1} × S_t` block.
- LRU cache keyed on (a, b); optional landmark/ALT approximation for large
  graphs (later milestone).

### 3.3 Retriever adapters — `mapmatched.adapters` (all optional extras)

Protocol — deliberately minimal so *anything* can provide candidates:

```python
class CandidateProvider(Protocol):
    def candidates(self, query: str | np.ndarray, m: int) -> list[tuple[ChunkId, float]]: ...
```

Adapters (each an optional dependency: `pip install map-matched-retrieval[faiss]`):

- `FAISSProvider`, `QdrantProvider`, `ChromaProvider`, `PgvectorProvider`
- `LangChainProvider(retriever)` / `LlamaIndexProvider(retriever)` — wrap theirs
- …and the reverse: `as_langchain_retriever(session)` /
  `as_llamaindex_retriever(session)` so map-matched retrieval drops into an
  existing chain with one line. **This is the adoption lever.**

### 3.4 Session layer — `mapmatched.session`

The user-facing API:

```python
from mapmatched import FAISSProvider, KNNGraph, MapMatchedRetriever

graph = KNNGraph.from_embeddings(
    chunk_ids,
    chunk_embeddings.tolist(),
    neighbor_count=10,
)
retriever = MapMatchedRetriever(
    graph,
    provider=FAISSProvider(index, chunk_ids, embed_query),
    emission_weight=8.0,
    transition_weight=0.5,
    candidate_limit=20,
    # fixed_lag=2,
)

session = retriever.session()
r1 = session.retrieve("How does the HMM handle noisy GPS?")
r2 = session.retrieve("what about when it jumps roads?")   # follow-up → prior disambiguates

r2.chunk_id           # decoded x*_t
r2.context_chunk_ids  # decoded chunk + graph neighbourhood + current candidates
r2.trace              # per-turn emission/transition split and entropy
session.trace.to_json()
session.trace.render()
```

### 3.5 Trace — `mapmatched.trace`

Per-turn record: emission score of chosen chunk, transition cost paid,
runner-up and margin, emission entropy `H_t`, classified move (`drill` if
`d_G` small / `jump` if the emission gain out-paid a large `d_G`), and the
backpointer path. JSON-serialisable; renderable. This is the artifact the
note says people will take.

### 3.6 Eval harness — `mapmatched.eval` (extra: `[eval]`)

Benchmarks mapped to claims:

| Benchmark | Proves |
| --- | --- |
| **TREC CAsT** (2019–22) | H1: lift on underspecified follow-ups (raw vs resolved utterances ≈ built-in underspecification signal). nDCG@3/5, Recall@k. |
| **TopiOCQA** | Drill-vs-jump: explicit topic switches inside conversations exercise the λ/β trade-off. |
| **QReCC** | Head-to-head vs the dominant paradigm (query rewriting → dense retrieval). |
| **OR-QuAC / INSCIT** | Secondary multi-turn + high-ambiguity stress. |
| **BEIR** (single-turn) | Sanity floor: β=0 must match baseline; proves we broke nothing. |

Harness responsibilities:

- Dataset loaders + corpus/graph preparation (Wikipedia section+hyperlink graph
  for the structured-graph arm; kNN for the fallback arm).
- **Slice reporting is mandatory**: results stratified by emission entropy —
  follow-up slice and standalone slice reported separately, never only a
  corpus average (the note's pass/fail condition: lift the follow-up slice
  without harming the standalone slice beyond a preset tolerance).
- Ablation runners: β sweep (incl. β=0, β→∞), graph-source arm, fixed-lag vs
  full Viterbi, M sweep.
- Baselines: vanilla top-k, query-rewriting + dense retrieval, Maximal Marginal
  Relevance, optional GraphRAG-style graph-no-trajectory.
- Report generation → the README headline numbers.

Open gap (flagged, not blocking): the "citation graph" arm has no off-the-shelf
multi-turn benchmark on a citation corpus; Wikipedia link+section graphs cover
the structured arm for v0. A small constructed citation-corpus eval is optional
later scope.

---

## 4. Relationship to composable-model-graph

CMG is the domain-independent substrate (control-theory concepts as graph
primitives). As of main commit `62ec13b`, CMG already owns generic sequential
estimation through `CandidateState`, `decode_path`, and
`decode_path_fixed_lag`. This library maps retrieval concepts onto that
substrate:

| CMG concept | Here |
| --- | --- |
| Transform | embed → candidates → trellis → decode |
| Data | query track, candidate sets |
| Graph | the trellis (and the corpus graph it scores against) |
| Trace | the emission/transition split — literally our flagship artifact |
| Evaluation | benchmark metrics, per-slice |
| Feedback | future λ/β adaptation from eval results |

**Decision: build map-matched-retrieval Python-first with its own backend
boundary.** Mapmatched owns its decoder protocol and retrieval-specific models,
and ships a dependency-free standalone decoder as the default. An optional CMG
backend translates at the boundary for users with a compatible installation.
CMG dataclasses are never part of mapmatched's public API. Until CMG has a
stable tagged or PyPI release, mapmatched does not require an unstable Git
dependency.

---

## 5. Repo layout

```
map-matched-retrieval/
  README.md                 # what it is / what it is not / headline numbers
  docs/
    theory.md               # the working note as markdown (hypothesis → eval)
    philosophy.md           # why decoder-not-store; trace-first
    gotchas.md              # score normalisation, re-decode semantics, graph quality bound
  src/mapmatched/
    core/                   # trellis.py, viterbi.py, fixed_lag.py, scoring.py, entropy.py
    graph/                  # protocol.py, knn.py, section.py, citation.py, networkx.py, cache.py
    adapters/               # faiss.py, qdrant.py, chroma.py, pgvector.py, langchain.py, llamaindex.py
    session.py
    trace/                  # record.py, render.py
    eval/                   # loaders/, slices.py, ablations.py, baselines/, report.py
  tests/
  examples/
    01-line-graph-sanity/   # the leaky-integrator reduction, no deps
    02-wikipedia-demo/      # section+link graph, real corpus, trace rendering
    03-langchain-drop-in/   # one-line swap into an existing chain
  CHANGELOG.md
```

Python ≥3.10. Extras: `[faiss] [qdrant] [chroma] [pgvector] [langchain] [llamaindex] [eval]`.

---

## 6. Milestones

- **M0 — proof of algorithm (complete)**: dependency-free decoders, in-memory
  corpus graph, line-graph sanity fixtures, degenerate modes, and unit tests.
- **M1 — usable library (complete)**: session API, JSON and terminal traces,
  optional FAISS adapter, deterministic embedding-derived `KNNGraph`, entropy,
  score-normalisation options, CI, runnable examples, and docs. Embeddings remain
  caller-supplied; the base package remains dependency-free.
- **M2 — proof of claim**: eval harness with TopiOCQA + CAsT, β ablations,
  entropy-sliced H1/H0 report, baselines (β=0, query-rewrite, Maximal Marginal
  Relevance).
  Output: the README headline table. *This is the credibility milestone.*
- **M3 — adoption**: LangChain/LlamaIndex two-way wrappers, Qdrant/Chroma/
  pgvector adapters, QReCC + BEIR runs, Wikipedia graph tooling, docs site.
- **M4 — substrate integration**: adopt a stable composable-model-graph release
  behind the existing decoder boundary; wire λ/β tuning as a CMG feedback loop.
  Optional: constructed citation-corpus eval.

---

## 7. Risks / open questions

1. **Graph quality bound** — the method is only as strong as the corpus is
   structured (stated in the note). Mitigation: honest kNN-fallback numbers;
   section/link graph tooling shipped, not assumed.
2. **Score-scale drift across turns/embedders** breaks a single λ. Mitigation:
   normalisation options in core + a documented default.
3. **Latency** — O(T·M²) + graph lookups per turn. Mitigation: fixed-lag mode,
   distance cache/cutoff, M defaults tuned in eval.
4. **H0 slice degradation** beyond tolerance on some corpora. If observed:
   entropy-gated β (anneal β toward 0 on low-entropy turns) — but only if the
   data forces it; keep the mechanism minimal first.
5. **Benchmark freshness** — CAsT/TopiOCQA/QReCC are established as of the
   plan's writing; run a survey pass for 2024–26 conversational-retrieval
   benchmarks before locking the harness.
