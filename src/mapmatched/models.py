from __future__ import annotations

import json
import math
from dataclasses import dataclass


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    chunk_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("candidate chunk_id must not be empty")
        _require_finite("candidate score", self.score)


@dataclass(frozen=True, slots=True)
class DecoderCandidate:
    chunk_id: str
    raw_score: float
    normalized_score: float

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("decoder candidate chunk_id must not be empty")
        _require_finite("raw score", self.raw_score)
        _require_finite("normalized score", self.normalized_score)


@dataclass(frozen=True, slots=True)
class DecodedStep:
    turn_index: int
    chunk_id: str
    raw_emission_score: float
    normalized_emission_score: float
    weighted_emission_score: float
    graph_distance: float
    weighted_transition_cost: float
    cumulative_score: float


@dataclass(frozen=True, slots=True)
class DecodedPath:
    steps: tuple[DecodedStep, ...]
    chunk_ids: tuple[str, ...]
    total_score: float

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("decoded path must contain at least one step")
        if self.chunk_ids != tuple(step.chunk_id for step in self.steps):
            raise ValueError("decoded path chunk_ids must match its steps")
        _require_finite("decoded path total score", self.total_score)


@dataclass(frozen=True, slots=True)
class TraceStep:
    turn_index: int
    chunk_id: str
    raw_emission_score: float
    normalized_emission_score: float
    weighted_emission_score: float
    graph_distance: float
    weighted_transition_cost: float
    cumulative_score: float
    emission_entropy: float
    normalized_emission_margin: float

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "chunk_id": self.chunk_id,
            "raw_emission_score": self.raw_emission_score,
            "normalized_emission_score": self.normalized_emission_score,
            "weighted_emission_score": self.weighted_emission_score,
            "graph_distance": self.graph_distance,
            "weighted_transition_cost": self.weighted_transition_cost,
            "cumulative_score": self.cumulative_score,
            "emission_entropy": self.emission_entropy,
            "normalized_emission_margin": self.normalized_emission_margin,
        }


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    steps: tuple[TraceStep, ...]
    path_chunk_ids: tuple[str, ...]
    revised_prior_indices: tuple[int, ...]
    fixed_lag: int | None
    committed_through_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "path_chunk_ids": list(self.path_chunk_ids),
            "revised_prior_indices": list(self.revised_prior_indices),
            "fixed_lag": self.fixed_lag,
            "committed_through_index": self.committed_through_index,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def render(self) -> str:
        from .trace import render_trace

        return render_trace(self)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk_id: str
    context_chunk_ids: tuple[str, ...]
    candidates: tuple[ScoredCandidate, ...]
    path: DecodedPath
    trace: RetrievalTrace
