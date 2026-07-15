# TREC CAsT 2019 Gemini Flash-Lite / MiniLM / kNN result

This is a Tier B judged-passage micro-corpus result, not a full MS MARCO/TREC CAR
retrieval result.

- Dataset: `trec-cast/v1/2019/judged`
- Conversations: 20 judged topics, 173 turns
- Corpus: 21,726 unique judged passages
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Graph: 10-neighbor embedding kNN
- Ranking: full, candidate limit 100
- Rewriter: `gemini-3.1-flash-lite`
- Rewrite prompt: `cast-standalone-v1`
- Bootstrap: 1,000 conversation-level paired draws, seed 42
- Profile: `cast2019-gemini-3.1-flash-lite-minilm-knn-full`
- Git revision: `1d8ea0fdc7face208dfab53589972818cf94cf41`

| Slice | Method | β | nDCG@3 | nDCG@3 95% CI | nDCG@5 | Recall@100 | Δ vs pointwise (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Follow-up | Pointwise | 0.0 | 0.325 | [0.232, 0.423] | 0.328 | 0.583 | +0.000 |
| Standalone | Pointwise | 0.0 | 0.200 | [0.143, 0.256] | 0.217 | 0.343 | +0.000 |
| Follow-up | Map-matched | 0.5 | 0.335 | [0.240, 0.432] | 0.340 | 0.583 | +0.009 [-0.006, +0.027] |
| Standalone | Map-matched | 0.5 | 0.204 | [0.147, 0.258] | 0.220 | 0.343 | +0.003 [-0.003, +0.010] |
| Follow-up | Map-matched | 1.0 | 0.352 | [0.263, 0.444] | 0.356 | 0.583 | +0.027 [+0.008, +0.049] |
| Standalone | Map-matched | 1.0 | 0.216 | [0.157, 0.272] | 0.227 | 0.343 | +0.015 [+0.004, +0.027] |
| Follow-up | History concat | — | 0.110 | [0.073, 0.160] | 0.112 | 0.328 | -0.215 [-0.325, -0.100] |
| Standalone | History concat | — | 0.189 | [0.134, 0.256] | 0.197 | 0.410 | -0.011 [-0.077, +0.057] |
| Follow-up | Gemini rewrite | — | 0.516 | [0.417, 0.598] | 0.507 | 0.700 | +0.191 [+0.078, +0.296] |
| Standalone | Gemini rewrite | — | 0.457 | [0.366, 0.537] | 0.464 | 0.607 | +0.257 [+0.187, +0.325] |
| Follow-up | MMR | — | 0.324 | [0.232, 0.423] | 0.327 | 0.583 | -0.001 [-0.004, +0.000] |
| Standalone | MMR | — | 0.200 | [0.143, 0.256] | 0.217 | 0.343 | +0.000 [+0.000, +0.000] |

## Interpretation

Map-matched retrieval at β=1.0 improves follow-up nDCG@3 by 0.027, with its
paired interval excluding zero, while also improving the standalone slice by
0.015. The claim gate passes.

Gemini rewriting is substantially stronger on both slices in this setup. It is
an API-backed query transformation baseline rather than a trajectory decoder,
and its hosted output can change across model revisions.

## Limitations

- The corpus contains only judged passages, so scores and recall do not estimate
  full-corpus retrieval performance.
- The CAsT query objects exposed by ir-datasets do not include manual rewrites;
  a resolved-query oracle is therefore omitted.
- Passage text comes from the locally built combined MS MARCO/TREC CAR docstore.
- Gemini rewrites were checkpointed after each successful request; the API key
  and billing data are not stored.
