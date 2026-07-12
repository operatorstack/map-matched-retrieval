from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from .graph import CorpusGraph
from .models import DecodedPath, DecodedStep, DecoderCandidate

Trellis = Sequence[Sequence[DecoderCandidate]]


class Decoder(Protocol):
    def decode(
        self,
        trellis: Trellis,
        *,
        graph: CorpusGraph,
        emission_weight: float,
        transition_weight: float,
        fixed_lag: int | None = None,
    ) -> DecodedPath: ...


def _validate_decode(
    trellis: Trellis,
    emission_weight: float,
    transition_weight: float,
    fixed_lag: int | None,
) -> None:
    if not trellis:
        raise ValueError("trellis must contain at least one turn")
    for turn_index, candidates in enumerate(trellis):
        if not candidates:
            raise ValueError(f"trellis turn {turn_index} must contain candidates")
        chunk_ids = [candidate.chunk_id for candidate in candidates]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f"trellis turn {turn_index} contains duplicate chunk IDs")
        for candidate in candidates:
            if not math.isfinite(candidate.raw_score):
                raise ValueError("raw candidate scores must be finite")
            if not math.isfinite(candidate.normalized_score):
                raise ValueError("normalized candidate scores must be finite")
            if not math.isfinite(emission_weight * candidate.normalized_score):
                raise ValueError("weighted emission scores must be finite")
    if not math.isfinite(emission_weight) or emission_weight < 0.0:
        raise ValueError("emission_weight must be finite and nonnegative")
    if not math.isfinite(transition_weight) or transition_weight < 0.0:
        raise ValueError("transition_weight must be finite and nonnegative")
    if fixed_lag is not None and fixed_lag < 0:
        raise ValueError("fixed_lag must be nonnegative")


class StandaloneDecoder:
    def decode(
        self,
        trellis: Trellis,
        *,
        graph: CorpusGraph,
        emission_weight: float,
        transition_weight: float,
        fixed_lag: int | None = None,
    ) -> DecodedPath:
        _validate_decode(trellis, emission_weight, transition_weight, fixed_lag)
        if fixed_lag is None:
            candidate_indices = self._decode_full(
                trellis, graph, emission_weight, transition_weight, {}
            )
        else:
            candidate_indices = self._decode_fixed_lag(
                trellis, graph, emission_weight, transition_weight, fixed_lag
            )
        return self._build_path(
            trellis,
            candidate_indices,
            graph,
            emission_weight,
            transition_weight,
        )

    def _forward(
        self,
        trellis: Trellis,
        graph: CorpusGraph,
        emission_weight: float,
        transition_weight: float,
        locked: dict[int, int],
    ) -> tuple[list[list[float]], list[list[int]]]:
        cumulative_scores: list[list[float]] = []
        backpointers: list[list[int]] = []
        for turn_index, candidates in enumerate(trellis):
            cumulative_scores.append([float("-inf")] * len(candidates))
            backpointers.append([-1] * len(candidates))
            locked_index = locked.get(turn_index)
            for candidate_index, candidate in enumerate(candidates):
                if locked_index is not None and candidate_index != locked_index:
                    continue
                emission_score = emission_weight * candidate.normalized_score
                if turn_index == 0:
                    cumulative_scores[turn_index][candidate_index] = emission_score
                    continue
                best_score = float("-inf")
                best_predecessor = -1
                for predecessor_index, predecessor in enumerate(trellis[turn_index - 1]):
                    predecessor_score = cumulative_scores[turn_index - 1][predecessor_index]
                    if predecessor_score == float("-inf"):
                        continue
                    graph_distance = graph.distance(predecessor.chunk_id, candidate.chunk_id)
                    weighted_cost = transition_weight * graph_distance
                    score = predecessor_score - weighted_cost
                    if not math.isfinite(score):
                        raise ValueError("decoder accumulation produced a nonfinite score")
                    if score > best_score:
                        best_score = score
                        best_predecessor = predecessor_index
                cumulative_scores[turn_index][candidate_index] = emission_score + best_score
                backpointers[turn_index][candidate_index] = best_predecessor
        return cumulative_scores, backpointers

    @staticmethod
    def _best_index(scores: Sequence[float]) -> int:
        best_index = 0
        best_score = float("-inf")
        for index, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    @staticmethod
    def _backtrack(backpointers: Sequence[Sequence[int]], final_index: int) -> list[int]:
        indices = [0] * len(backpointers)
        indices[-1] = final_index
        for turn_index in range(len(backpointers) - 1, 0, -1):
            indices[turn_index - 1] = backpointers[turn_index][indices[turn_index]]
        return indices

    def _decode_full(
        self,
        trellis: Trellis,
        graph: CorpusGraph,
        emission_weight: float,
        transition_weight: float,
        locked: dict[int, int],
    ) -> list[int]:
        cumulative_scores, backpointers = self._forward(
            trellis, graph, emission_weight, transition_weight, locked
        )
        final_index = self._best_index(cumulative_scores[-1])
        return self._backtrack(backpointers, final_index)

    def _decode_fixed_lag(
        self,
        trellis: Trellis,
        graph: CorpusGraph,
        emission_weight: float,
        transition_weight: float,
        fixed_lag: int,
    ) -> list[int]:
        committed: dict[int, int] = {}
        for final_turn_index in range(len(trellis)):
            prefix = trellis[: final_turn_index + 1]
            prefix_path = self._decode_full(
                prefix, graph, emission_weight, transition_weight, committed
            )
            if final_turn_index >= fixed_lag:
                commit_index = final_turn_index - fixed_lag
                committed[commit_index] = prefix_path[commit_index]
        full_path = self._decode_full(trellis, graph, emission_weight, transition_weight, committed)
        return [
            committed.get(index, candidate_index) for index, candidate_index in enumerate(full_path)
        ]

    @staticmethod
    def _build_path(
        trellis: Trellis,
        candidate_indices: Sequence[int],
        graph: CorpusGraph,
        emission_weight: float,
        transition_weight: float,
    ) -> DecodedPath:
        steps: list[DecodedStep] = []
        cumulative_score = 0.0
        for turn_index, candidate_index in enumerate(candidate_indices):
            candidate = trellis[turn_index][candidate_index]
            if turn_index == 0:
                graph_distance = 0.0
            else:
                previous_candidate = trellis[turn_index - 1][candidate_indices[turn_index - 1]]
                graph_distance = graph.distance(previous_candidate.chunk_id, candidate.chunk_id)
            weighted_emission_score = emission_weight * candidate.normalized_score
            weighted_transition_cost = transition_weight * graph_distance
            cumulative_score += weighted_emission_score - weighted_transition_cost
            steps.append(
                DecodedStep(
                    turn_index=turn_index,
                    chunk_id=candidate.chunk_id,
                    raw_emission_score=candidate.raw_score,
                    normalized_emission_score=candidate.normalized_score,
                    weighted_emission_score=weighted_emission_score,
                    graph_distance=graph_distance,
                    weighted_transition_cost=weighted_transition_cost,
                    cumulative_score=cumulative_score,
                )
            )
        return DecodedPath(
            steps=tuple(steps),
            chunk_ids=tuple(step.chunk_id for step in steps),
            total_score=cumulative_score,
        )
