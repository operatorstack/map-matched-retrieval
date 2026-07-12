from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mapmatched import CorpusGraph, MapMatchedRetriever, ScoredCandidate
from mapmatched.scoring import ScoreNormalization

from ..corpus import BruteForceProvider


@dataclass(frozen=True, slots=True)
class MapMatchedMethodConfig:
    transition_weight: float = 0.5
    emission_weight: float = 1.0
    candidate_limit: int = 20
    fixed_lag: int | None = None
    score_normalization: ScoreNormalization = "zscore"


def run_mapmatched_conversation(
    *,
    graph: CorpusGraph,
    provider: BruteForceProvider,
    queries: Sequence[str],
    config: MapMatchedMethodConfig,
) -> tuple[tuple[str, ...], tuple[float | None, ...], tuple[tuple[str, ...], ...]]:
    retriever = MapMatchedRetriever(
        graph,
        provider=provider,
        transition_weight=config.transition_weight,
        emission_weight=config.emission_weight,
        candidate_limit=config.candidate_limit,
        fixed_lag=config.fixed_lag,
        score_normalization=config.score_normalization,
    )
    session = retriever.session()
    ranked_ids: list[str] = []
    entropies: list[float | None] = []
    candidate_rankings: list[tuple[str, ...]] = []
    for query in queries:
        result = session.retrieve(query)
        ranked_ids.append(result.chunk_id)
        entropies.append(result.trace.steps[-1].emission_entropy)
        # current-turn candidates ranked by trajectory score; fall back to the
        # decoded chunk when the decoder does not expose a ranking.
        if result.candidate_ranking:
            candidate_rankings.append(tuple(score.chunk_id for score in result.candidate_ranking))
        else:
            candidate_rankings.append((result.chunk_id,))
    return tuple(ranked_ids), tuple(entropies), tuple(candidate_rankings)


def run_pointwise_conversation(
    *,
    graph: CorpusGraph,
    provider: BruteForceProvider,
    queries: Sequence[str],
    config: MapMatchedMethodConfig,
) -> tuple[tuple[str, ...], tuple[float | None, ...], tuple[tuple[str, ...], ...]]:
    pointwise_config = MapMatchedMethodConfig(
        transition_weight=0.0,
        emission_weight=config.emission_weight,
        candidate_limit=config.candidate_limit,
        fixed_lag=config.fixed_lag,
        score_normalization=config.score_normalization,
    )
    return run_mapmatched_conversation(
        graph=graph,
        provider=provider,
        queries=queries,
        config=pointwise_config,
    )


def run_history_concat_conversation(
    *,
    provider: BruteForceProvider,
    queries: Sequence[str],
    candidate_limit: int,
) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    history: list[str] = []
    ranked_ids: list[str] = []
    for query in queries:
        rewritten = " ".join([*history, query]).strip()
        candidates = provider.candidates(rewritten, candidate_limit)
        if not candidates:
            raise RuntimeError("history concat baseline received no candidates")
        ranked_ids.append(candidates[0].chunk_id)
        history.append(query)
    return tuple(ranked_ids), tuple(None for _ in queries)


def run_resolved_oracle_conversation(
    *,
    provider: BruteForceProvider,
    resolved_queries: Sequence[str],
    candidate_limit: int,
) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    ranked_ids: list[str] = []
    for query in resolved_queries:
        candidates = provider.candidates(query, candidate_limit)
        if not candidates:
            raise RuntimeError("resolved oracle baseline received no candidates")
        ranked_ids.append(candidates[0].chunk_id)
    return tuple(ranked_ids), tuple(None for _ in resolved_queries)


def run_maximal_marginal_relevance_conversation(
    *,
    provider: BruteForceProvider,
    queries: Sequence[str],
    candidate_limit: int,
    diversity_weight: float = 0.5,
) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between zero and one")
    prior_passage_id: str | None = None
    ranked_ids: list[str] = []
    for query in queries:
        candidates = provider.candidates(query, candidate_limit)
        if not candidates:
            raise RuntimeError("maximal marginal relevance baseline received no candidates")
        if prior_passage_id is None:
            chosen = candidates[0]
        else:
            chosen = _select_mmr_candidate(
                candidates,
                prior_passage_id=prior_passage_id,
                diversity_weight=diversity_weight,
            )
        ranked_ids.append(chosen.chunk_id)
        prior_passage_id = chosen.chunk_id
    return tuple(ranked_ids), tuple(None for _ in queries)


def _select_mmr_candidate(
    candidates: Sequence[ScoredCandidate],
    *,
    prior_passage_id: str,
    diversity_weight: float,
) -> ScoredCandidate:
    if not candidates:
        raise ValueError("candidate set must not be empty")
    maximum_score = max(candidate.score for candidate in candidates)
    minimum_score = min(candidate.score for candidate in candidates)
    score_range = maximum_score - minimum_score
    best_candidate = candidates[0]
    best_value = float("-inf")
    for candidate in candidates:
        relevance = 1.0 if score_range == 0.0 else (candidate.score - minimum_score) / score_range
        redundancy = 1.0 if candidate.chunk_id == prior_passage_id else 0.0
        mmr_value = diversity_weight * relevance - (1.0 - diversity_weight) * redundancy
        if mmr_value > best_value:
            best_value = mmr_value
            best_candidate = candidate
    return best_candidate
