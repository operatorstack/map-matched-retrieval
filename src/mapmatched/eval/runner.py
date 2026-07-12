from __future__ import annotations

from collections.abc import Sequence

from mapmatched import CorpusGraph

from .baselines import MapMatchedMethodConfig, run_mapmatched_conversation
from .baselines.methods import run_maximal_marginal_relevance_conversation
from .corpus import (
    BruteForceProvider,
    build_knn_graph,
    build_passage_embeddings,
    build_section_graph,
)
from .embedder import TextEmbedder
from .metrics import ndcg_at_k, recall_at_k, turn_ranked_relevances
from .slices import (
    build_method_metrics,
    classify_turn_slice,
    compute_entropy_threshold,
    evaluate_claim,
)
from .types import EvalConfig, EvalConversation, EvalReport, MethodMetrics, Passage, TurnMetrics


class MethodSpec:
    __slots__ = ("fixed_lag", "name", "transition_weight")

    def __init__(
        self,
        *,
        name: str,
        transition_weight: float | None = None,
        fixed_lag: int | None = None,
    ) -> None:
        self.name = name
        self.transition_weight = transition_weight
        self.fixed_lag = fixed_lag


def rank_full_corpus(provider: BruteForceProvider, query: str) -> tuple[str, ...]:
    candidates = provider.candidates(query, limit=provider.passage_count)
    return tuple(candidate.chunk_id for candidate in candidates)


def run_method_on_conversation(
    *,
    conversation: EvalConversation,
    provider: BruteForceProvider,
    graph: CorpusGraph,
    method: MethodSpec,
    config: MapMatchedMethodConfig,
    ranking_mode: str = "full",
) -> tuple[tuple[tuple[str, ...], ...], tuple[float | None, ...]]:
    queries = [turn.query for turn in conversation.turns]
    method_config = MapMatchedMethodConfig(
        transition_weight=method.transition_weight
        if method.transition_weight is not None
        else config.transition_weight,
        emission_weight=config.emission_weight,
        candidate_limit=config.candidate_limit,
        fixed_lag=method.fixed_lag if method.fixed_lag is not None else config.fixed_lag,
        score_normalization=config.score_normalization,
    )
    if method.name in {"mapmatched", "pointwise"}:
        if method.name == "pointwise":
            method_config = MapMatchedMethodConfig(
                transition_weight=0.0,
                emission_weight=config.emission_weight,
                candidate_limit=config.candidate_limit,
                fixed_lag=method_config.fixed_lag,
                score_normalization=config.score_normalization,
            )
        decoded_ids, entropies, candidate_rankings = run_mapmatched_conversation(
            graph=graph,
            provider=provider,
            queries=queries,
            config=method_config,
        )
        if ranking_mode == "full":
            trajectory_rankings = _rankings_from_trajectory(provider, queries, candidate_rankings)
        else:
            trajectory_rankings = _rankings_with_decoded_first(provider, queries, decoded_ids)
        return trajectory_rankings, entropies
    if method.name == "history_concat":
        rankings: list[tuple[str, ...]] = []
        history: list[str] = []
        for query in queries:
            rewritten = " ".join([*history, query]).strip()
            rankings.append(rank_full_corpus(provider, rewritten))
            history.append(query)
        return tuple(rankings), tuple(None for _ in queries)
    if method.name == "resolved_oracle":
        oracle_queries = tuple(
            turn.resolved_query if turn.resolved_query is not None else turn.query
            for turn in conversation.turns
        )
        return tuple(rank_full_corpus(provider, query) for query in oracle_queries), tuple(
            None for _ in queries
        )
    if method.name == "maximal_marginal_relevance":
        decoded_ids, entropies = run_maximal_marginal_relevance_conversation(
            provider=provider,
            queries=queries,
            candidate_limit=config.candidate_limit,
        )
        return _rankings_with_decoded_first(provider, queries, decoded_ids), entropies
    raise ValueError(f"unsupported method: {method.name}")


def _rankings_with_decoded_first(
    provider: BruteForceProvider,
    queries: Sequence[str],
    decoded_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    rankings: list[tuple[str, ...]] = []
    for query, decoded_id in zip(queries, decoded_ids, strict=True):
        ranked = list(rank_full_corpus(provider, query))
        if decoded_id in ranked:
            ranked.remove(decoded_id)
        rankings.append((decoded_id, *ranked))
    return tuple(rankings)


def _rankings_from_trajectory(
    provider: BruteForceProvider,
    queries: Sequence[str],
    candidate_rankings: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    rankings: list[tuple[str, ...]] = []
    for query, candidate_ids in zip(queries, candidate_rankings, strict=True):
        seen = set(candidate_ids)
        tail = tuple(
            chunk_id for chunk_id in rank_full_corpus(provider, query) if chunk_id not in seen
        )
        rankings.append((*candidate_ids, *tail))
    return tuple(rankings)


def _build_provider_and_graph(
    passages: Sequence[Passage],
    embedder: TextEmbedder,
    *,
    graph_source: str,
) -> tuple[BruteForceProvider, CorpusGraph]:
    passage_ids, passage_embeddings = build_passage_embeddings(passages, embedder)
    provider = BruteForceProvider(passage_ids, passage_embeddings, embedder)
    if graph_source == "section":
        graph = build_section_graph(passages)
    else:
        graph = build_knn_graph(passage_ids, passage_embeddings)
    return provider, graph


def evaluate_method(
    *,
    conversations: Sequence[EvalConversation],
    provider: BruteForceProvider,
    graph: CorpusGraph,
    method: MethodSpec,
    config: MapMatchedMethodConfig,
    eval_config: EvalConfig,
    trace_entropies: Sequence[Sequence[float | None]],
    entropy_threshold: float,
    graph_mode: str,
) -> MethodMetrics:
    turn_metrics: list[TurnMetrics] = []
    for conversation, trace_entropies_for_conversation in zip(
        conversations,
        trace_entropies,
        strict=True,
    ):
        rankings, _ = run_method_on_conversation(
            conversation=conversation,
            provider=provider,
            graph=graph,
            method=method,
            config=config,
            ranking_mode=eval_config.ranking_mode,
        )
        for turn, ranking, trace_entropy in zip(
            conversation.turns,
            rankings,
            trace_entropies_for_conversation,
            strict=True,
        ):
            relevances = turn_ranked_relevances(ranking, turn.qrels)
            turn_metrics.append(
                TurnMetrics(
                    turn_index=turn.turn_index,
                    ndcg_at_3=ndcg_at_k(relevances, 3),
                    ndcg_at_5=ndcg_at_k(relevances, 5),
                    recall_at_k=recall_at_k(ranking, turn.qrels, eval_config.recall_k),
                    emission_entropy=trace_entropy,
                    slice_name="all",
                )
            )

    classified_turns = tuple(
        TurnMetrics(
            turn_index=turn.turn_index,
            ndcg_at_3=turn.ndcg_at_3,
            ndcg_at_5=turn.ndcg_at_5,
            recall_at_k=turn.recall_at_k,
            emission_entropy=turn.emission_entropy,
            slice_name=classify_turn_slice(
                turn_index=turn.turn_index,
                emission_entropy=turn.emission_entropy,
                entropy_threshold=entropy_threshold,
            ),
        )
        for turn in turn_metrics
    )

    effective_transition_weight: float | None
    if method.name == "pointwise":
        effective_transition_weight = 0.0
    elif method.name == "mapmatched":
        effective_transition_weight = (
            method.transition_weight
            if method.transition_weight is not None
            else config.transition_weight
        )
    else:
        effective_transition_weight = method.transition_weight

    return build_method_metrics(
        method_name=method.name,
        transition_weight=effective_transition_weight,
        candidate_limit=config.candidate_limit,
        fixed_lag=method.fixed_lag if method.fixed_lag is not None else config.fixed_lag,
        graph_mode=graph_mode,
        turns=classified_turns,
    )


def run_eval(
    *,
    conversations: Sequence[EvalConversation],
    passages: Sequence[Passage],
    embedder: TextEmbedder,
    methods: Sequence[MethodSpec],
    config: MapMatchedMethodConfig,
    eval_config: EvalConfig,
) -> EvalReport:
    graph_mode = eval_config.graph_source
    provider, graph = _build_provider_and_graph(
        passages,
        embedder,
        graph_source=graph_mode,
    )
    trace_method = MethodSpec(name="pointwise")
    trace_entropies = tuple(
        run_method_on_conversation(
            conversation=conversation,
            provider=provider,
            graph=graph,
            method=trace_method,
            config=config,
            ranking_mode=eval_config.ranking_mode,
        )[1]
        for conversation in conversations
    )
    flat_trace_entropies = [entropy for conversation in trace_entropies for entropy in conversation]
    entropy_threshold = (
        eval_config.entropy_threshold
        if eval_config.entropy_threshold is not None
        else compute_entropy_threshold(flat_trace_entropies)
    )
    method_metrics = tuple(
        evaluate_method(
            conversations=conversations,
            provider=provider,
            graph=graph,
            method=method,
            config=config,
            eval_config=eval_config,
            trace_entropies=trace_entropies,
            entropy_threshold=entropy_threshold,
            graph_mode=graph_mode,
        )
        for method in methods
    )
    mapmatched_methods = [method for method in methods if method.name == "mapmatched"]
    if not mapmatched_methods:
        verdict = None
    else:
        best_mapmatched = max(
            (method for method in method_metrics if method.method_name == "mapmatched"),
            key=lambda method: next(
                slice_metrics.ndcg_at_3
                for slice_metrics in method.slices
                if slice_metrics.slice_name == "follow_up"
            ),
        )
        verdict = evaluate_claim(
            method_metrics,
            mapmatched_method_name=best_mapmatched.method_name,
            baseline_method_name="pointwise",
            standalone_tolerance=eval_config.standalone_tolerance,
            follow_up_min_delta=eval_config.follow_up_min_delta,
            mapmatched_transition_weight=best_mapmatched.transition_weight,
        )
    return EvalReport(config=eval_config, methods=method_metrics, verdict=verdict)
