import json

import pytest

from mapmatched import (
    InMemoryCorpusGraph,
    MapMatchedRetriever,
    ScoredCandidate,
)


class FixtureProvider:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def candidates(self, query: str, limit: int) -> list[ScoredCandidate]:
        self.queries.append((query, limit))
        return [
            ScoredCandidate("0", 5.0),
            ScoredCandidate("1", 0.0),
            ScoredCandidate("2", 0.0),
        ][:limit]


def line_graph() -> InMemoryCorpusGraph:
    return InMemoryCorpusGraph.from_edges(
        [("0", "1"), ("1", "2")],
        maximum_distance=3.0,
    )


def conversation_turns() -> list[list[ScoredCandidate]]:
    return [
        [ScoredCandidate("0", 5.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 0.0)],
        [ScoredCandidate("0", 3.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 6.0)],
        [ScoredCandidate("0", 8.0), ScoredCandidate("1", 0.0), ScoredCandidate("2", 0.0)],
    ]


def test_provider_api_and_direct_candidates_api() -> None:
    provider = FixtureProvider()
    retriever = MapMatchedRetriever(
        line_graph(),
        provider=provider,
        score_normalization="none",
        candidate_limit=2,
    )

    provider_result = retriever.session().retrieve("question")
    direct_result = retriever.session().retrieve_candidates(conversation_turns()[0])

    assert provider.queries == [("question", 2)]
    assert provider_result.candidates == tuple(conversation_turns()[0][:2])
    assert direct_result.candidates == tuple(conversation_turns()[0][:2])


def test_full_decode_reports_prior_revisions_and_current_map_chunk() -> None:
    session = MapMatchedRetriever(
        line_graph(),
        score_normalization="none",
    ).session()

    first, second, third = (
        session.retrieve_candidates(candidates) for candidates in conversation_turns()
    )

    assert first.trace.path_chunk_ids == ("0",)
    assert second.trace.path_chunk_ids == ("0", "2")
    assert third.chunk_id == "0"
    assert session.current_chunk_id == "0"
    assert third.trace.path_chunk_ids == ("0", "0", "0")
    assert third.trace.revised_prior_indices == (1,)
    assert third.trace.fixed_lag is None
    assert third.trace.committed_through_index is None


def test_fixed_lag_exposes_commit_boundary_and_stabilizes_history() -> None:
    session = MapMatchedRetriever(
        line_graph(),
        score_normalization="none",
        fixed_lag=0,
    ).session()

    results = [session.retrieve_candidates(candidates) for candidates in conversation_turns()]

    assert results[-1].trace.path_chunk_ids == ("0", "2", "0")
    assert results[-1].trace.revised_prior_indices == ()
    assert results[-1].trace.fixed_lag == 0
    assert results[-1].trace.committed_through_index == 2


def test_context_expansion_is_separate_ordered_and_deduplicated() -> None:
    session = MapMatchedRetriever(
        line_graph(),
        score_normalization="none",
        transition_weight=0.0,
        context_radius=1.0,
    ).session()

    result = session.retrieve_candidates([ScoredCandidate("0", 5.0), ScoredCandidate("2", 4.0)])

    assert result.chunk_id == "0"
    assert result.context_chunk_ids == ("0", "1", "2")
    assert result.path.chunk_ids == ("0",)


def test_trace_contains_scores_cost_diagnostics_and_valid_json() -> None:
    session = MapMatchedRetriever(
        line_graph(),
        score_normalization="none",
        transition_weight=2.0,
    ).session()
    session.retrieve_candidates([ScoredCandidate("0", 2.0), ScoredCandidate("2", 1.0)])
    result = session.retrieve_candidates([ScoredCandidate("1", 3.0), ScoredCandidate("2", 2.0)])

    payload = json.loads(result.trace.to_json())
    latest = payload["steps"][-1]

    assert latest["raw_emission_score"] == 3.0
    assert latest["normalized_emission_score"] == 3.0
    assert latest["weighted_transition_cost"] == 2.0
    assert latest["graph_distance"] == 1.0
    assert latest["emission_entropy"] > 0.0
    assert latest["normalized_emission_margin"] == 1.0
    assert payload["path_chunk_ids"] == ["0", "1"]


def test_failed_turn_does_not_mutate_session() -> None:
    session = MapMatchedRetriever(line_graph()).session()

    with pytest.raises(ValueError, match="must not be empty"):
        session.retrieve_candidates([])

    assert session.turn_count == 0
    assert session.path is None
    assert session.trace is None


def test_query_api_requires_provider() -> None:
    session = MapMatchedRetriever(line_graph()).session()

    with pytest.raises(RuntimeError, match="CandidateProvider"):
        session.retrieve("question")
    with pytest.raises(ValueError, match="must not be empty"):
        session.retrieve("")
