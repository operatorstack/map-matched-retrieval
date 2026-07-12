import json
import math

import pytest

from mapmatched import (
    GraphEdge,
    InMemoryCorpusGraph,
    ScoredCandidate,
    normalize_candidates,
    softmax_entropy,
)


def test_weighted_graph_distance_neighborhood_and_clamping_are_deterministic() -> None:
    graph = InMemoryCorpusGraph(
        [
            GraphEdge("a", "b", 1.0),
            GraphEdge("a", "c", 1.0),
            GraphEdge("b", "d", 1.5),
            GraphEdge("c", "d", 1.0),
        ],
        nodes=["isolated"],
        maximum_distance=3.0,
    )

    assert graph.distance("a", "d") == 2.0
    assert graph.distance("d", "a") == 2.0
    assert graph.distance("a", "isolated") == 3.0
    assert graph.distance("missing", "missing") == 0.0
    assert graph.neighborhood("a", 2.0) == ("b", "c", "d")
    assert graph.neighborhood("a", 0.0) == ()
    assert graph.distance("a", "d") == 2.0


def test_directed_graph_and_shortest_path_cutoff() -> None:
    graph = InMemoryCorpusGraph.from_edges(
        [("a", "b"), ("b", "c")],
        directed=True,
        maximum_distance=1.5,
    )

    assert graph.distance("a", "b") == 1.0
    assert graph.distance("b", "a") == 1.5
    assert graph.distance("a", "c") == 1.5


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("none", (1.0, 2.0, 3.0)),
        ("center", (-1.0, 0.0, 1.0)),
        ("zscore", (-math.sqrt(1.5), 0.0, math.sqrt(1.5))),
    ],
)
def test_score_normalization_preserves_raw_scores(
    method: str,
    expected: tuple[float, ...],
) -> None:
    candidates = [
        ScoredCandidate("a", 1.0),
        ScoredCandidate("b", 2.0),
        ScoredCandidate("c", 3.0),
    ]

    normalized = normalize_candidates(candidates, method)

    assert tuple(candidate.raw_score for candidate in normalized) == (1.0, 2.0, 3.0)
    assert tuple(candidate.normalized_score for candidate in normalized) == pytest.approx(expected)


def test_zscore_constant_scores_is_safe() -> None:
    normalized = normalize_candidates(
        [ScoredCandidate("a", 7.0), ScoredCandidate("b", 7.0)]
    )

    assert tuple(candidate.normalized_score for candidate in normalized) == (0.0, 0.0)


def test_softmax_entropy_is_stable_for_extreme_scores() -> None:
    assert softmax_entropy([10_000.0, 10_000.0]) == pytest.approx(math.log(2.0))
    assert softmax_entropy([10_000.0, -10_000.0]) == pytest.approx(0.0)
    assert softmax_entropy([4.0]) == 0.0


def test_graph_and_scoring_invalid_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        ScoredCandidate("a", float("nan"))
    with pytest.raises(ValueError, match="unique"):
        normalize_candidates([ScoredCandidate("a", 1.0), ScoredCandidate("a", 2.0)])
    with pytest.raises(ValueError, match="at least one"):
        softmax_entropy([])
    with pytest.raises(ValueError, match="greater than zero"):
        GraphEdge("a", "b", -1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        InMemoryCorpusGraph().neighborhood("a", -1.0)


def test_trace_json_fixture_is_standard_json() -> None:
    payload = {"entropy": softmax_entropy([1.0, 2.0])}
    encoded = json.dumps(payload, allow_nan=False)

    assert json.loads(encoded) == payload
