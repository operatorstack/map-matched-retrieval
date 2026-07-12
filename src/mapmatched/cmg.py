from __future__ import annotations

from typing import Protocol

from .decoder import Trellis, _validate_decode
from .graph import CorpusGraph
from .models import DecodedPath, DecodedStep


class CMGBackendUnavailableError(ImportError):
    pass


class _CMGCandidate(Protocol):
    id: str


class CMGDecoder:
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
        try:
            from composable_model_graph import (
                CandidateState,
                decode_path,
                decode_path_fixed_lag,
            )
        except ImportError as error:
            raise CMGBackendUnavailableError(
                "CMGDecoder requires a compatible composable-model-graph installation; "
                "no unstable Git dependency is installed by mapmatched"
            ) from error

        cmg_trellis = [
            [
                CandidateState(
                    id=candidate.chunk_id,
                    score=emission_weight * candidate.normalized_score,
                )
                for candidate in candidates
            ]
            for candidates in trellis
        ]

        def transition_cost(
            previous: _CMGCandidate,
            current: _CMGCandidate,
            step_index: int,
        ) -> float:
            del step_index
            return graph.distance(previous.id, current.id)

        if fixed_lag is None:
            cmg_path = decode_path(
                cmg_trellis,
                transition_cost=transition_cost,
                transition_weight=transition_weight,
            )
        else:
            cmg_path = decode_path_fixed_lag(
                cmg_trellis,
                fixed_lag,
                transition_cost=transition_cost,
                transition_weight=transition_weight,
            )
        state_ids = cmg_path.state_ids
        if not isinstance(state_ids, list) or not all(
            isinstance(state_id, str) for state_id in state_ids
        ):
            raise RuntimeError("CMG decoder returned invalid state IDs")
        if len(state_ids) != len(trellis):
            raise RuntimeError("CMG decoder returned a path with the wrong length")

        steps: list[DecodedStep] = []
        cumulative_score = 0.0
        for turn_index, chunk_id in enumerate(state_ids):
            candidates_by_id = {candidate.chunk_id: candidate for candidate in trellis[turn_index]}
            candidate = candidates_by_id.get(chunk_id)
            if candidate is None:
                raise RuntimeError("CMG decoder returned an unknown state ID")
            graph_distance = (
                0.0 if turn_index == 0 else graph.distance(state_ids[turn_index - 1], chunk_id)
            )
            weighted_emission_score = emission_weight * candidate.normalized_score
            weighted_transition_cost = transition_weight * graph_distance
            cumulative_score += weighted_emission_score - weighted_transition_cost
            steps.append(
                DecodedStep(
                    turn_index=turn_index,
                    chunk_id=chunk_id,
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
            chunk_ids=tuple(state_ids),
            total_score=cumulative_score,
        )
