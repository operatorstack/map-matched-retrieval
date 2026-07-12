from mapmatched import (
    CMGBackendUnavailableError,
    CMGDecoder,
    InMemoryCorpusGraph,
    MapMatchedRetriever,
    ScoredCandidate,
)


graph = InMemoryCorpusGraph.from_edges(
    [("0", "1"), ("1", "2")],
    maximum_distance=3.0,
)
session = MapMatchedRetriever(
    graph,
    decoder=CMGDecoder(),
    score_normalization="none",
).session()

try:
    session.retrieve_candidates(
        [ScoredCandidate("0", 5.0), ScoredCandidate("1", 0.0)]
    )
    result = session.retrieve_candidates(
        [ScoredCandidate("0", 2.0), ScoredCandidate("2", 4.0)]
    )
except CMGBackendUnavailableError as error:
    raise SystemExit(str(error)) from error

print(result.trace.to_json(indent=2))
