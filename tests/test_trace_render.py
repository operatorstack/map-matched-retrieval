from mapmatched import (
    InMemoryCorpusGraph,
    MapMatchedRetriever,
    RetrievalTrace,
    ScoredCandidate,
    render_trace,
)


def test_trace_render_includes_path_mode_and_diagnostics() -> None:
    graph = InMemoryCorpusGraph.from_edges(
        [("a", "b"), ("b", "c")],
        maximum_distance=3.0,
    )
    session = MapMatchedRetriever(
        graph,
        score_normalization="none",
        transition_weight=2.0,
    ).session()
    session.retrieve_candidates([ScoredCandidate("a", 5.0), ScoredCandidate("c", 0.0)])
    result = session.retrieve_candidates([ScoredCandidate("b", 0.0), ScoredCandidate("c", 2.0)])

    rendered = result.trace.render()

    assert rendered == render_trace(result.trace)
    assert "path: a -> b" in rendered
    assert "mode: full" in rendered
    assert "revised prior turns: none" in rendered
    assert "turn  chunk" in rendered
    assert "entropy" in rendered
    assert "     -2" in rendered


def test_fixed_lag_render_shows_commitment_and_revisions() -> None:
    graph = InMemoryCorpusGraph.from_edges([("a", "b")])
    session = MapMatchedRetriever(
        graph,
        score_normalization="none",
        fixed_lag=0,
    ).session()

    result = session.retrieve_candidates([ScoredCandidate("a", 1.0)])

    assert "mode: fixed-lag 0" in result.trace.render()
    assert "committed through turn: 0" in result.trace.render()


def test_full_decode_render_lists_revised_prior_turns() -> None:
    graph = InMemoryCorpusGraph.from_edges(
        [("0", "1"), ("1", "2")],
        maximum_distance=3.0,
    )
    session = MapMatchedRetriever(
        graph,
        score_normalization="none",
    ).session()
    session.retrieve_candidates(
        [ScoredCandidate("0", 5.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 0.0)]
    )
    session.retrieve_candidates(
        [ScoredCandidate("0", 3.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 6.0)]
    )
    result = session.retrieve_candidates(
        [ScoredCandidate("0", 8.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 0.0)]
    )

    assert "revised prior turns: 1" in result.trace.render()


def test_empty_trace_renders_without_a_table() -> None:
    trace = RetrievalTrace(
        steps=(),
        path_chunk_ids=(),
        revised_prior_indices=(1, 3),
        fixed_lag=None,
        committed_through_index=None,
    )

    assert trace.render() == (
        "path: (empty)\nmode: full\nrevised prior turns: 1, 3\ncommitted through turn: none"
    )
