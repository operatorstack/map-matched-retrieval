from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    total = 0.0
    for index, relevance in enumerate(relevances[:k]):
        if index == 0:
            total += relevance
        else:
            total += relevance / math.log2(index + 2)
    return total


def ndcg_at_k(ranked_relevances: Sequence[float], k: int) -> float:
    if not ranked_relevances:
        return 0.0
    ideal = sorted(ranked_relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal_dcg


def recall_at_k(
    ranked_passage_ids: Sequence[str],
    qrels: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant = {passage_id for passage_id, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_passage_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def turn_ranked_relevances(
    ranked_passage_ids: Sequence[str],
    qrels: Mapping[str, int],
) -> tuple[float, ...]:
    return tuple(float(qrels.get(passage_id, 0)) for passage_id in ranked_passage_ids)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.fsum(values) / len(values)
