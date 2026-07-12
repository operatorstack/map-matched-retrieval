from mapmatched import InMemoryCorpusGraph, MapMatchedRetriever, ScoredCandidate

graph = InMemoryCorpusGraph.from_edges(
    [("hmm", "noise"), ("noise", "road-jumps")],
    maximum_distance=4.0,
)
session = MapMatchedRetriever(
    graph,
    score_normalization="none",
    transition_weight=1.0,
).session()

session.retrieve_candidates(
    [
        ScoredCandidate("hmm", 5.0),
        ScoredCandidate("noise", 1.0),
        ScoredCandidate("road-jumps", 0.0),
    ]
)
result = session.retrieve_candidates(
    [
        ScoredCandidate("hmm", 1.0),
        ScoredCandidate("noise", 3.0),
        ScoredCandidate("road-jumps", 3.5),
    ]
)

print(result.chunk_id)
print(result.context_chunk_ids)
print(result.trace.to_json(indent=2))
