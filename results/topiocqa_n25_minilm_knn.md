# TopiOCQA n=25 MiniLM/kNN result

Profile: `topiocqa-n25-minilm-knn-full`

- Dataset: `topiocqa_valid.jsonl`
- Dataset SHA-256: `1bba9512b24b2e5de22704766dd80b1dc497bb7262799f66c3912e6f413ac1c6`
- Conversations: 25
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Graph: 10-neighbor kNN
- Ranking: full, candidate limit 100
- Bootstrap: 1,000 conversation-level draws, seed 42
- Package: `map-matched-retrieval==0.1.0`
- Git revision: `60a9d694b807c3eb49da2a00743c6f1a05d52e6d`
- Full JSON report SHA-256: `35a1801b2d7502a7bc1036f90b53818783f56ca9f2ffeebdfad66fc41056ce9e`

| Slice | Method | β | nDCG@3 | nDCG@3 95% CI | Delta vs pointwise | Paired delta 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Follow-up | Pointwise | 0.00 | 0.150 | [0.109, 0.196] | +0.000 | — |
| Follow-up | Map-matched | 0.50 | 0.195 | [0.147, 0.249] | +0.045 | [+0.018, +0.077] |
| Follow-up | Map-matched | 1.00 | 0.234 | [0.186, 0.294] | +0.084 | [+0.046, +0.128] |
| Follow-up | History concat | — | 0.071 | [0.043, 0.099] | -0.079 | [-0.137, -0.027] |
| Follow-up | MMR | — | 0.151 | [0.109, 0.197] | +0.001 | [+0.000, +0.003] |
| Standalone | Pointwise | 0.00 | 0.342 | [0.282, 0.405] | +0.000 | — |
| Standalone | Map-matched | 0.50 | 0.359 | [0.295, 0.424] | +0.017 | [+0.004, +0.034] |
| Standalone | Map-matched | 1.00 | 0.373 | [0.305, 0.440] | +0.031 | [+0.009, +0.055] |
| Standalone | History concat | — | 0.134 | [0.110, 0.159] | -0.208 | [-0.261, -0.158] |
| Standalone | MMR | — | 0.340 | [0.280, 0.404] | -0.002 | [-0.004, +0.000] |

Both repeated runs produced byte-identical JSON and markdown reports. This is a
gold-passage micro-corpus result, not full-Wikipedia retrieval. The paired
intervals quantify uncertainty for these 25 conversations and should not be
generalized to other corpora without additional evaluation.
