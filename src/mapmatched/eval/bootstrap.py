from __future__ import annotations

import math
import random
from collections.abc import Sequence

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
    conversation_count = len(conversation_turns)
    if conversation_count == 0:
        return None

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(num_samples):
        resampled = [
            conversation_turns[rng.randrange(conversation_count)] for _ in range(conversation_count)
        ]
        selected_turns = [
            turn
            for conversation in resampled
            for turn in conversation
            if turn.slice_name == slice_name
        ]
        bootstrap_means.append(mean([turn.ndcg_at_3 for turn in selected_turns]))

    sorted_means = sorted(bootstrap_means)
    return (
        _percentile(sorted_means, lower_percentile),
        _percentile(sorted_means, upper_percentile),
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
