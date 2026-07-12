# Theory and semantics

For conversation turn \(t\), a candidate provider supplies states \(S_t\) and raw
retrieval scores \(r_t(c)\). A configured normalization produces \(s_t(c)\).
Map-matched retrieval chooses one corpus chunk per turn:

\[
\hat{x}_{1:T} = \arg\max_{x_t \in S_t}
\sum_{t=1}^{T}\lambda s_t(x_t)
- \sum_{t=2}^{T}\beta d_G(x_{t-1}, x_t).
\]

Here \(\lambda\) is `emission_weight`, \(\beta\) is `transition_weight`, and
\(d_G\) is bounded shortest-path distance in the corpus graph. This is a
linear-chain energy objective; callers do not need to interpret provider scores
as probabilities.

## Normalization and uncertainty

Raw similarity scales differ across retrievers and can drift across turns. The
default population z-score normalization is

\[
s_t(c) = \frac{r_t(c)-\mu_t}{\sigma_t}.
\]

When all scores are equal, all normalized scores are zero. `center` subtracts
only the mean, while `none` is appropriate for scores already calibrated across
turns. Traces preserve both values.

Emission entropy is the Shannon entropy of
`softmax(emission_weight * normalized scores)`. The implementation subtracts the
maximum logit before exponentiation, so large finite inputs remain safe. Entropy
describes candidate ambiguity at a turn; it is not path posterior entropy.

## Graph distance

`InMemoryCorpusGraph` supports positive weighted edges. Searches stop at
`maximum_distance`; disconnected nodes and paths outside that bound receive the
same finite clamped distance. This makes jumps expensive but possible. Neighbor
expansion uses the same bounded shortest distances and orders equal-distance
neighbors by chunk ID.

## Full and fixed-lag decoding

Full Viterbi decoding reruns over the whole trellis after every turn. New evidence
can revise any earlier state, and `revised_prior_indices` makes that behavior
explicit.

Fixed-lag decoding commits state \(t-L\) after observing turn \(t\). Lag zero is
causal filtering, while a lag at least `turn_count - 1` equals full decoding for
the current trellis. Only the uncommitted tail can change as turns arrive.

With zero transition weight the recurrence separates by turn and therefore
returns pointwise argmax exactly. Strict comparisons retain the first provider
candidate on ties.

## MAP path versus context

The decoder returns one maximum-score path. Context expansion is a deterministic
post-processing step: append the decoded current chunk, its graph neighborhood,
and the current turn's candidates, then stably deduplicate. It does not represent
multiple paths or uncertainty-aware path ranking.

## Backend boundary

The `Decoder` protocol and all path models belong to mapmatched. The standalone
backend is the default. `CMGDecoder` translates candidates into
composable-model-graph's generic estimation types and translates its result back.
This boundary avoids exposing unstable external dataclasses or requiring a Git
dependency while preserving an inspectable parity route for compatible local
CMG installations.
