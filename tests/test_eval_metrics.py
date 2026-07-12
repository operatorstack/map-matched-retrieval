import math

import pytest

from mapmatched.eval.metrics import dcg_at_k, mean, ndcg_at_k, recall_at_k


def test_dcg_at_k_matches_hand_calculation() -> None:
    expected = 3.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)
    assert dcg_at_k([3.0, 2.0, 1.0], 3) == pytest.approx(expected)


def test_ndcg_at_k_is_one_for_perfect_ranking() -> None:
    assert ndcg_at_k([3.0, 2.0, 1.0], 3) == pytest.approx(1.0)


def test_ndcg_at_k_is_zero_when_all_relevances_are_zero() -> None:
    assert ndcg_at_k([0.0, 0.0], 2) == 0.0


def test_recall_at_k_counts_any_positive_relevance() -> None:
    ranking = ("a", "b", "c")
    qrels = {"a": 0, "b": 2, "d": 1}
    assert recall_at_k(ranking, qrels, 2) == pytest.approx(0.5)


def test_mean_returns_zero_for_empty_sequence() -> None:
    assert mean([]) == 0.0
