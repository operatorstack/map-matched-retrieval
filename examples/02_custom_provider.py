from mapmatched import InMemoryCorpusGraph, MapMatchedRetriever, ScoredCandidate


class KeywordProvider:
    def __init__(self, documents: dict[str, str]) -> None:
        self._documents = documents

    def candidates(self, query: str, limit: int) -> list[ScoredCandidate]:
        query_terms = set(query.lower().split())
        scored = [
            ScoredCandidate(
                chunk_id=chunk_id,
                score=float(len(query_terms.intersection(text.lower().split()))),
            )
            for chunk_id, text in self._documents.items()
        ]
        return sorted(scored, key=lambda candidate: (-candidate.score, candidate.chunk_id))[:limit]


documents = {
    "intro": "hidden markov model map matching",
    "noise": "noisy observations and candidate states",
    "jumps": "road jumps transition distance",
}
graph = InMemoryCorpusGraph.from_edges(
    [("intro", "noise"), ("noise", "jumps")],
    maximum_distance=4.0,
)
session = MapMatchedRetriever(
    graph,
    provider=KeywordProvider(documents),
    candidate_limit=3,
).session()

session.retrieve("How does map matching handle noisy observations?")
result = session.retrieve("What about jumps?")

print(result.chunk_id)
print(result.trace.path_chunk_ids)
