from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from .metrics import mean
from .types import SliceName, TurnMetrics


def bootstrap_ndcg_at_3_ci(
    conversation_turns: Sequence[Sequence[TurnMetrics]],
    slice_name: SliceName,
    *,
    num_samples: int,
    seed: int | None = None,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> tuple[float, float] | None:
    if num_samples <= 0:
        return None
    contributing_conversations = _contributing_conversations(conversation_turns, slice_name)
    conversation_count = len(contributing_conversations)
    if conversation_count == 0:
        return None

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(num_samples):
        resampled = [
            contributing_conversations[rng.randrange(conversation_count)]
            for _ in range(conversation_count)
        ]
        selected_turns = _selected_turns(resampled, slice_name)
        bootstrap_means.append(mean([turn.ndcg_at_3 for turn in selected_turns]))

    return _percentile_interval(
        bootstrap_means,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )


def bootstrap_slice_cis(
    conversation_turns: Sequence[Sequence[TurnMetrics]],
    *,
    num_samples: int,
    seed: int | None = None,
) -> dict[SliceName, tuple[float, float] | None]:
    slice_names: tuple[SliceName, ...] = ("follow_up", "standalone", "all")
    return {
        slice_name: bootstrap_ndcg_at_3_ci(
            conversation_turns,
            slice_name,
            num_samples=num_samples,
            seed=seed,
        )
        for slice_name in slice_names
    }


def bootstrap_paired_ndcg_at_3_delta_ci(
    treatment_conversation_turns: Sequence[Sequence[TurnMetrics]],
    baseline_conversation_turns: Sequence[Sequence[TurnMetrics]],
    slice_name: SliceName,
    *,
    num_samples: int,
    seed: int | None = None,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> tuple[float, float] | None:
    if num_samples <= 0:
        return None
    treatment_by_id = _conversations_by_id(treatment_conversation_turns)
    baseline_by_id = _conversations_by_id(baseline_conversation_turns)
    _validate_alignment(treatment_by_id, baseline_by_id)
    contributing_ids = tuple(
        conversation_id
        for conversation_id, turns in treatment_by_id.items()
        if _selected_turns((turns,), slice_name)
    )
    if not contributing_ids:
        return None

    rng = random.Random(seed)
    bootstrap_deltas: list[float] = []
    for _ in range(num_samples):
        sampled_ids = tuple(
            contributing_ids[rng.randrange(len(contributing_ids))]
            for _ in range(len(contributing_ids))
        )
        treatment_turns = _selected_turns(
            tuple(treatment_by_id[conversation_id] for conversation_id in sampled_ids),
            slice_name,
        )
        baseline_turns = _selected_turns(
            tuple(baseline_by_id[conversation_id] for conversation_id in sampled_ids),
            slice_name,
        )
        treatment_mean = mean([turn.ndcg_at_3 for turn in treatment_turns])
        baseline_mean = mean([turn.ndcg_at_3 for turn in baseline_turns])
        bootstrap_deltas.append(treatment_mean - baseline_mean)

    return _percentile_interval(
        bootstrap_deltas,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )


def bootstrap_paired_slice_delta_cis(
    treatment_conversation_turns: Sequence[Sequence[TurnMetrics]],
    baseline_conversation_turns: Sequence[Sequence[TurnMetrics]],
    *,
    num_samples: int,
    seed: int | None = None,
) -> dict[SliceName, tuple[float, float] | None]:
    slice_names: tuple[SliceName, ...] = ("follow_up", "standalone", "all")
    return {
        slice_name: bootstrap_paired_ndcg_at_3_delta_ci(
            treatment_conversation_turns,
            baseline_conversation_turns,
            slice_name,
            num_samples=num_samples,
            seed=seed,
        )
        for slice_name in slice_names
    }


def _contributing_conversations(
    conversation_turns: Sequence[Sequence[TurnMetrics]],
    slice_name: SliceName,
) -> tuple[Sequence[TurnMetrics], ...]:
    return tuple(
        turns for turns in conversation_turns if _selected_turns((turns,), slice_name)
    )


def _selected_turns(
    conversation_turns: Sequence[Sequence[TurnMetrics]],
    slice_name: SliceName,
) -> list[TurnMetrics]:
    return [
        turn
        for conversation in conversation_turns
        for turn in conversation
        if slice_name == "all" or turn.slice_name == slice_name
    ]


def _conversations_by_id(
    conversation_turns: Sequence[Sequence[TurnMetrics]],
) -> dict[str, tuple[TurnMetrics, ...]]:
    by_id: dict[str, tuple[TurnMetrics, ...]] = {}
    for turns in conversation_turns:
        if not turns:
            raise ValueError("conversation turn metrics must not be empty")
        conversation_ids = {turn.conversation_id for turn in turns}
        if len(conversation_ids) != 1:
            raise ValueError("conversation turn metrics contain multiple conversation IDs")
        conversation_id = next(iter(conversation_ids))
        if conversation_id in by_id:
            raise ValueError(f"duplicate conversation metrics: {conversation_id}")
        by_id[conversation_id] = tuple(turns)
    return by_id


def _validate_alignment(
    treatment_by_id: Mapping[str, Sequence[TurnMetrics]],
    baseline_by_id: Mapping[str, Sequence[TurnMetrics]],
) -> None:
    if treatment_by_id.keys() != baseline_by_id.keys():
        raise ValueError("treatment and baseline conversation IDs do not match")
    for conversation_id, treatment_turns in treatment_by_id.items():
        baseline_turns = baseline_by_id[conversation_id]
        treatment_keys = tuple(
            (turn.turn_index, turn.slice_name) for turn in treatment_turns
        )
        baseline_keys = tuple((turn.turn_index, turn.slice_name) for turn in baseline_turns)
        if treatment_keys != baseline_keys:
            raise ValueError(
                f"treatment and baseline turns do not align for {conversation_id}"
            )


def _percentile_interval(
    values: Sequence[float],
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    sorted_values = sorted(values)
    return (
        _percentile(sorted_values, lower_percentile),
        _percentile(sorted_values, upper_percentile),
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[int(rank)]
    weight = rank - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight
