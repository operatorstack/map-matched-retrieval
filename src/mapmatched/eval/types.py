from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SliceName = Literal["follow_up", "standalone", "all"]


@dataclass(frozen=True, slots=True)
class Passage:
    passage_id: str
    text: str
    # Optional structural grouping key (e.g. a Wikipedia article / section id).
    # Passages sharing a group_key are adjacent in a SectionGraph. kNN ignores it.
    group_key: str | None = None

    def __post_init__(self) -> None:
        if not self.passage_id:
            raise ValueError("passage_id must not be empty")
        if not self.text:
            raise ValueError("passage text must not be empty")


@dataclass(frozen=True, slots=True)
class EvalTurn:
    turn_index: int
    query: str
    qrels: dict[str, int]
    resolved_query: str | None = None

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be nonnegative")
        if not self.query:
            raise ValueError("query must not be empty")
        if not self.qrels:
            raise ValueError("qrels must contain at least one judged passage")


@dataclass(frozen=True, slots=True)
class EvalConversation:
    conversation_id: str
    turns: tuple[EvalTurn, ...]

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")
        if not self.turns:
            raise ValueError("conversation must contain at least one turn")
        expected_indices = tuple(range(len(self.turns)))
        actual_indices = tuple(turn.turn_index for turn in self.turns)
        if actual_indices != expected_indices:
            raise ValueError("turn indices must be contiguous starting at zero")


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    conversation_id: str
    turn_index: int
    ndcg_at_3: float
    ndcg_at_5: float
    recall_at_k: float
    emission_entropy: float | None
    slice_name: SliceName


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    slice_name: SliceName
    turn_count: int
    ndcg_at_3: float
    ndcg_at_5: float
    recall_at_k: float
    ndcg_at_3_ci: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class MethodMetrics:
    method_name: str
    transition_weight: float | None
    candidate_limit: int
    fixed_lag: int | None
    graph_mode: str
    slices: tuple[SliceMetrics, ...]
    turns: tuple[TurnMetrics, ...]


@dataclass(frozen=True, slots=True)
class ComparisonSliceMetrics:
    slice_name: SliceName
    turn_count: int
    ndcg_at_3_delta: float
    ndcg_at_3_delta_ci: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class MethodComparison:
    method_name: str
    transition_weight: float | None
    baseline_method_name: str
    baseline_transition_weight: float | None
    candidate_limit: int
    fixed_lag: int | None
    graph_mode: str
    slices: tuple[ComparisonSliceMetrics, ...]


@dataclass(frozen=True, slots=True)
class EvalConfig:
    benchmark: str
    tier: str
    embedder_name: str
    recall_k: int
    entropy_threshold: float | None
    standalone_tolerance: float
    follow_up_min_delta: float
    # "full" re-ranks the candidate window by trajectory score; "rank1" only
    # hoists the decoded chunk to position 1 (the pre-fix behaviour).
    ranking_mode: str = "full"
    # "knn" (embedding fallback) or "section" (structured group_key graph).
    graph_source: str = "knn"
    knn_neighbor_count: int = 10
    bootstrap_samples: int = 0
    bootstrap_seed: int | None = 42
    profile: str | None = None
    dataset_filename: str | None = None
    dataset_sha256: str | None = None
    conversation_ids: tuple[str, ...] = ()
    embedding_model: str | None = None
    package_version: str | None = None
    git_revision: str | None = None
    query_rewrite_provider: str | None = None
    query_rewrite_model: str | None = None
    query_rewrite_prompt_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "tier": self.tier,
            "embedder_name": self.embedder_name,
            "recall_k": self.recall_k,
            "entropy_threshold": self.entropy_threshold,
            "standalone_tolerance": self.standalone_tolerance,
            "follow_up_min_delta": self.follow_up_min_delta,
            "ranking_mode": self.ranking_mode,
            "graph_source": self.graph_source,
            "knn_neighbor_count": self.knn_neighbor_count,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "profile": self.profile,
            "dataset_filename": self.dataset_filename,
            "dataset_sha256": self.dataset_sha256,
            "conversation_ids": list(self.conversation_ids),
            "embedding_model": self.embedding_model,
            "package_version": self.package_version,
            "git_revision": self.git_revision,
            "query_rewrite_provider": self.query_rewrite_provider,
            "query_rewrite_model": self.query_rewrite_model,
            "query_rewrite_prompt_version": self.query_rewrite_prompt_version,
        }


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    follow_up_lift: float
    standalone_delta: float
    follow_up_pass: bool
    standalone_pass: bool
    overall_pass: bool


@dataclass(frozen=True, slots=True)
class EvalReport:
    config: EvalConfig
    methods: tuple[MethodMetrics, ...]
    verdict: ClaimVerdict | None
    comparisons: tuple[MethodComparison, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "methods": [
                {
                    "method_name": method.method_name,
                    "transition_weight": method.transition_weight,
                    "candidate_limit": method.candidate_limit,
                    "fixed_lag": method.fixed_lag,
                    "graph_mode": method.graph_mode,
                    "slices": [
                        {
                            "slice_name": slice_metrics.slice_name,
                            "turn_count": slice_metrics.turn_count,
                            "ndcg_at_3": slice_metrics.ndcg_at_3,
                            "ndcg_at_5": slice_metrics.ndcg_at_5,
                            "recall_at_k": slice_metrics.recall_at_k,
                            "ndcg_at_3_ci": None
                            if slice_metrics.ndcg_at_3_ci is None
                            else list(slice_metrics.ndcg_at_3_ci),
                        }
                        for slice_metrics in method.slices
                    ],
                    "turns": [
                        {
                            "conversation_id": turn.conversation_id,
                            "turn_index": turn.turn_index,
                            "ndcg_at_3": turn.ndcg_at_3,
                            "ndcg_at_5": turn.ndcg_at_5,
                            "recall_at_k": turn.recall_at_k,
                            "emission_entropy": turn.emission_entropy,
                            "slice_name": turn.slice_name,
                        }
                        for turn in method.turns
                    ],
                }
                for method in self.methods
            ],
            "comparisons": [
                {
                    "method_name": comparison.method_name,
                    "transition_weight": comparison.transition_weight,
                    "baseline_method_name": comparison.baseline_method_name,
                    "baseline_transition_weight": comparison.baseline_transition_weight,
                    "candidate_limit": comparison.candidate_limit,
                    "fixed_lag": comparison.fixed_lag,
                    "graph_mode": comparison.graph_mode,
                    "slices": [
                        {
                            "slice_name": slice_metrics.slice_name,
                            "turn_count": slice_metrics.turn_count,
                            "ndcg_at_3_delta": slice_metrics.ndcg_at_3_delta,
                            "ndcg_at_3_delta_ci": None
                            if slice_metrics.ndcg_at_3_delta_ci is None
                            else list(slice_metrics.ndcg_at_3_delta_ci),
                        }
                        for slice_metrics in comparison.slices
                    ],
                }
                for comparison in self.comparisons
            ],
            "verdict": None
            if self.verdict is None
            else {
                "follow_up_lift": self.verdict.follow_up_lift,
                "standalone_delta": self.verdict.standalone_delta,
                "follow_up_pass": self.verdict.follow_up_pass,
                "standalone_pass": self.verdict.standalone_pass,
                "overall_pass": self.verdict.overall_pass,
            },
        }
