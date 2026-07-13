import pytest

from mapmatched import (
    CMGBackendUnavailableError,
    CMGDecoder,
    InMemoryCorpusGraph,
    ScoredCandidate,
    StandaloneDecoder,
    normalize_candidates,
)


def line_graph() -> InMemoryCorpusGraph:
    return InMemoryCorpusGraph.from_edges(
        [("0", "1"), ("1", "2")],
        maximum_distance=3.0,
    )


def hand_computed_trellis() -> tuple[tuple[object, ...], ...]:
    turns = [
        [ScoredCandidate("0", 5.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 0.0)],
        [ScoredCandidate("0", 3.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 6.0)],
        [ScoredCandidate("0", 8.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 0.0)],
    ]
    return tuple(normalize_candidates(turn, "none") for turn in turns)


def test_hand_computed_viterbi_fixture() -> None:
    path = StandaloneDecoder().decode(
        hand_computed_trellis(),
        graph=line_graph(),
        emission_weight=1.0,
        transition_weight=1.0,
    )

    assert path.chunk_ids == ("0", "0", "0")
    assert path.total_score == 16.0
    assert tuple(step.cumulative_score for step in path.steps) == (5.0, 8.0, 16.0)
    assert tuple(step.graph_distance for step in path.steps) == (0.0, 0.0, 0.0)


def test_zero_transition_weight_equals_pointwise_argmax() -> None:
    class CountingDistanceGraph(InMemoryCorpusGraph):
        def __init__(self) -> None:
            super().__init__()
            self.distance_count = 0

        def distance(self, source_chunk_id: str, target_chunk_id: str) -> float:
            self.distance_count += 1
            return super().distance(source_chunk_id, target_chunk_id)

    graph = CountingDistanceGraph()
    path = StandaloneDecoder().decode(
        hand_computed_trellis(),
        graph=graph,
        emission_weight=1.0,
        transition_weight=0.0,
    )

    assert path.chunk_ids == ("0", "2", "0")
    assert graph.distance_count == 0


def test_fixed_lag_behavior() -> None:
    decoder = StandaloneDecoder()
    graph = line_graph()

    lag_zero = decoder.decode(
        hand_computed_trellis(),
        graph=graph,
        emission_weight=1.0,
        transition_weight=1.0,
        fixed_lag=0,
    )
    lag_one = decoder.decode(
        hand_computed_trellis(),
        graph=graph,
        emission_weight=1.0,
        transition_weight=1.0,
        fixed_lag=1,
    )
    lag_two = decoder.decode(
        hand_computed_trellis(),
        graph=graph,
        emission_weight=1.0,
        transition_weight=1.0,
        fixed_lag=2,
    )

    assert lag_zero.chunk_ids == ("0", "2", "0")
    assert lag_one.chunk_ids == ("0", "0", "0")
    assert lag_two.chunk_ids == ("0", "0", "0")


def test_ties_follow_candidate_input_order() -> None:
    trellis = (
        normalize_candidates([ScoredCandidate("b", 1.0), ScoredCandidate("a", 1.0)], "none"),
        normalize_candidates([ScoredCandidate("d", 1.0), ScoredCandidate("c", 1.0)], "none"),
    )

    path = StandaloneDecoder().decode(
        trellis,
        graph=InMemoryCorpusGraph(maximum_distance=1.0),
        emission_weight=1.0,
        transition_weight=0.0,
    )

    assert path.chunk_ids == ("b", "d")


@pytest.mark.parametrize(
    ("emission_weight", "transition_weight", "fixed_lag"),
    [
        (-1.0, 1.0, None),
        (1.0, -1.0, None),
        (1.0, 1.0, -1),
        (float("inf"), 1.0, None),
    ],
)
def test_invalid_decoder_parameters(
    emission_weight: float,
    transition_weight: float,
    fixed_lag: int | None,
) -> None:
    with pytest.raises(ValueError):
        StandaloneDecoder().decode(
            hand_computed_trellis(),
            graph=line_graph(),
            emission_weight=emission_weight,
            transition_weight=transition_weight,
            fixed_lag=fixed_lag,
        )


def test_empty_trellis_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        StandaloneDecoder().decode(
            (),
            graph=line_graph(),
            emission_weight=1.0,
            transition_weight=1.0,
        )


def test_optional_cmg_backend_has_parity_or_clear_unavailable_error() -> None:
    decoder = CMGDecoder()
    arguments = {
        "graph": line_graph(),
        "emission_weight": 1.0,
        "transition_weight": 1.0,
    }
    try:
        cmg_path = decoder.decode(hand_computed_trellis(), **arguments)
    except CMGBackendUnavailableError as error:
        assert "compatible composable-model-graph installation" in str(error)
        return

    standalone_path = StandaloneDecoder().decode(hand_computed_trellis(), **arguments)
    assert cmg_path == standalone_path
