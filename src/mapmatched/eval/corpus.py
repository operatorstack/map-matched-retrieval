from __future__ import annotations

from collections.abc import Sequence

from mapmatched import KNNGraph, ScoredCandidate

from .embedder import PassageEmbedder, QueryEmbedder, dot_product
from .types import Passage


class BruteForceProvider:
    def __init__(
        self,
        passage_ids: Sequence[str],
        passage_embeddings: Sequence[Sequence[float]],
        embed_query: QueryEmbedder,
    ) -> None:
        ids = tuple(passage_ids)
        if not ids:
            raise ValueError("passage_ids must contain at least one ID")
        if len(ids) != len(passage_embeddings):
            raise ValueError("passage_ids and passage_embeddings length mismatch")
        self._passage_ids = ids
        self._passage_embeddings = tuple(tuple(values) for values in passage_embeddings)
        self._embed_query = embed_query

    @property
    def passage_count(self) -> int:
        return len(self._passage_ids)

    def candidates(self, query: str, limit: int) -> list[ScoredCandidate]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        query_embedding = tuple(self._embed_query.embed_query(query))
        scored = [
            (
                dot_product(query_embedding, passage_embedding),
                passage_id,
            )
            for passage_id, passage_embedding in zip(
                self._passage_ids,
                self._passage_embeddings,
                strict=True,
            )
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            ScoredCandidate(chunk_id=passage_id, score=score)
            for score, passage_id in scored[:limit]
        ]


def build_passage_embeddings(
    passages: Sequence[Passage],
    embedder: PassageEmbedder,
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    ordered_passages = tuple(passages)
    passage_ids = tuple(passage.passage_id for passage in ordered_passages)
    embeddings = tuple(
        tuple(embedder.embed_passage(passage.passage_id, passage.text))
        for passage in ordered_passages
    )
    return passage_ids, embeddings


def build_knn_graph(
    passage_ids: Sequence[str],
    passage_embeddings: Sequence[Sequence[float]],
    *,
    neighbor_count: int = 10,
) -> KNNGraph:
    return KNNGraph.from_embeddings(
        passage_ids,
        [list(values) for values in passage_embeddings],
        neighbor_count=neighbor_count,
    )
