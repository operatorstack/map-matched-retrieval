from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from .decoder import Decoder, StandaloneDecoder
from .graph import CorpusGraph
from .models import (
    CandidateScore,
    DecodedPath,
    DecoderCandidate,
    RetrievalResult,
    RetrievalTrace,
    ScoredCandidate,
    TraceStep,
)
from .scoring import ScoreNormalization, normalize_candidates, softmax_entropy


class CandidateProvider(Protocol):
    def candidates(self, query: str, limit: int) -> Sequence[ScoredCandidate]: ...


class MapMatchedRetriever:
    def __init__(
        self,
        graph: CorpusGraph,
        *,
        provider: CandidateProvider | None = None,
        decoder: Decoder | None = None,
        candidate_limit: int = 20,
        emission_weight: float = 1.0,
        transition_weight: float = 1.0,
        score_normalization: ScoreNormalization = "zscore",
        fixed_lag: int | None = None,
        context_radius: float = 1.0,
        context_limit: int | None = None,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than zero")
        if not math.isfinite(emission_weight) or emission_weight < 0.0:
            raise ValueError("emission_weight must be finite and nonnegative")
        if not math.isfinite(transition_weight) or transition_weight < 0.0:
            raise ValueError("transition_weight must be finite and nonnegative")
        if fixed_lag is not None and fixed_lag < 0:
            raise ValueError("fixed_lag must be nonnegative")
        if not math.isfinite(context_radius) or context_radius < 0.0:
            raise ValueError("context_radius must be finite and nonnegative")
        if context_limit is not None and context_limit <= 0:
            raise ValueError("context_limit must be greater than zero")
        if score_normalization not in ("zscore", "center", "none"):
            raise ValueError(f"unsupported score normalization: {score_normalization}")
        self.graph = graph
        self.provider = provider
        self.decoder = decoder if decoder is not None else StandaloneDecoder()
        self.candidate_limit = candidate_limit
        self.emission_weight = emission_weight
        self.transition_weight = transition_weight
        self.score_normalization = score_normalization
        self.fixed_lag = fixed_lag
        self.context_radius = context_radius
        self.context_limit = context_limit

    def session(self) -> MapMatchedSession:
        return MapMatchedSession(self)


class MapMatchedSession:
    def __init__(self, retriever: MapMatchedRetriever) -> None:
        self._retriever = retriever
        self._trellis: list[tuple[DecoderCandidate, ...]] = []
        self._candidate_turns: list[tuple[ScoredCandidate, ...]] = []
        self._path: DecodedPath | None = None
        self._trace: RetrievalTrace | None = None

    @property
    def turn_count(self) -> int:
        return len(self._trellis)

    @property
    def current_chunk_id(self) -> str | None:
        if self._path is None:
            return None
        return self._path.chunk_ids[-1]

    @property
    def path(self) -> DecodedPath | None:
        return self._path

    @property
    def trace(self) -> RetrievalTrace | None:
        return self._trace

    def retrieve(self, query: str) -> RetrievalResult:
        if not query:
            raise ValueError("query must not be empty")
        provider = self._retriever.provider
        if provider is None:
            raise RuntimeError(
                "retrieve(query) requires a CandidateProvider; use retrieve_candidates instead"
            )
        candidates = provider.candidates(query, self._retriever.candidate_limit)
        return self.retrieve_candidates(candidates)

    def retrieve_candidates(
        self,
        candidates: Sequence[ScoredCandidate],
    ) -> RetrievalResult:
        limited_candidates = tuple(candidates[: self._retriever.candidate_limit])
        normalized_candidates = normalize_candidates(
            limited_candidates,
            self._retriever.score_normalization,
        )
        previous_path = self._path
        previous_trace = self._trace
        self._candidate_turns.append(limited_candidates)
        self._trellis.append(normalized_candidates)
        try:
            decoder = self._retriever.decoder
            candidate_ranking: tuple[CandidateScore, ...] = ()
            decode_ranked = getattr(decoder, "decode_ranked", None)
            if callable(decode_ranked):
                path, candidate_ranking = decode_ranked(
                    self._trellis,
                    graph=self._retriever.graph,
                    emission_weight=self._retriever.emission_weight,
                    transition_weight=self._retriever.transition_weight,
                    fixed_lag=self._retriever.fixed_lag,
                )
            else:
                path = decoder.decode(
                    self._trellis,
                    graph=self._retriever.graph,
                    emission_weight=self._retriever.emission_weight,
                    transition_weight=self._retriever.transition_weight,
                    fixed_lag=self._retriever.fixed_lag,
                )
            revised_indices = self._revised_indices(previous_path, path)
            if (
                previous_trace is not None
                and previous_trace.committed_through_index is not None
                and any(
                    index <= previous_trace.committed_through_index for index in revised_indices
                )
            ):
                raise RuntimeError("fixed-lag decoder revised a committed turn")
            trace = self._build_trace(path, revised_indices)
            context_chunk_ids = self._build_context(path.chunk_ids[-1], limited_candidates)
        except Exception:
            self._candidate_turns.pop()
            self._trellis.pop()
            raise
        self._path = path
        self._trace = trace
        return RetrievalResult(
            chunk_id=path.chunk_ids[-1],
            context_chunk_ids=context_chunk_ids,
            candidates=limited_candidates,
            path=path,
            trace=trace,
            candidate_ranking=candidate_ranking,
        )

    @staticmethod
    def _revised_indices(
        previous_path: DecodedPath | None,
        path: DecodedPath,
    ) -> tuple[int, ...]:
        if previous_path is None:
            return ()
        return tuple(
            index
            for index, previous_chunk_id in enumerate(previous_path.chunk_ids)
            if path.chunk_ids[index] != previous_chunk_id
        )

    def _build_trace(
        self,
        path: DecodedPath,
        revised_indices: tuple[int, ...],
    ) -> RetrievalTrace:
        trace_steps: list[TraceStep] = []
        for decoded_step, candidates in zip(path.steps, self._trellis, strict=True):
            weighted_scores = [
                self._retriever.emission_weight * candidate.normalized_score
                for candidate in candidates
            ]
            alternatives = [
                candidate.normalized_score
                for candidate in candidates
                if candidate.chunk_id != decoded_step.chunk_id
            ]
            normalized_margin = (
                decoded_step.normalized_emission_score - max(alternatives) if alternatives else 0.0
            )
            trace_steps.append(
                TraceStep(
                    turn_index=decoded_step.turn_index,
                    chunk_id=decoded_step.chunk_id,
                    raw_emission_score=decoded_step.raw_emission_score,
                    normalized_emission_score=decoded_step.normalized_emission_score,
                    weighted_emission_score=decoded_step.weighted_emission_score,
                    graph_distance=decoded_step.graph_distance,
                    weighted_transition_cost=decoded_step.weighted_transition_cost,
                    cumulative_score=decoded_step.cumulative_score,
                    emission_entropy=softmax_entropy(weighted_scores),
                    normalized_emission_margin=normalized_margin,
                )
            )
        committed_through_index = None
        if self._retriever.fixed_lag is not None:
            candidate_index = len(self._trellis) - self._retriever.fixed_lag - 1
            if candidate_index >= 0:
                committed_through_index = candidate_index
        return RetrievalTrace(
            steps=tuple(trace_steps),
            path_chunk_ids=path.chunk_ids,
            revised_prior_indices=revised_indices,
            fixed_lag=self._retriever.fixed_lag,
            committed_through_index=committed_through_index,
        )

    def _build_context(
        self,
        decoded_chunk_id: str,
        candidates: Sequence[ScoredCandidate],
    ) -> tuple[str, ...]:
        ordered_chunk_ids = [
            decoded_chunk_id,
            *self._retriever.graph.neighborhood(
                decoded_chunk_id,
                self._retriever.context_radius,
            ),
            *(candidate.chunk_id for candidate in candidates),
        ]
        context: list[str] = []
        seen: set[str] = set()
        for chunk_id in ordered_chunk_ids:
            if chunk_id in seen:
                continue
            context.append(chunk_id)
            seen.add(chunk_id)
            if (
                self._retriever.context_limit is not None
                and len(context) >= self._retriever.context_limit
            ):
                break
        return tuple(context)
