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
                        }
                        for slice_metrics in method.slices
                    ],
                    "turns": [
                        {
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
