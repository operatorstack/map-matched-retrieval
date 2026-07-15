from __future__ import annotations

import heapq
import importlib
from collections import OrderedDict
from collections.abc import Sequence

from mapmatched import KNNGraph, ScoredCandidate
from mapmatched.graph import InMemoryCorpusGraph

from .embedder import PassageEmbedder, QueryEmbedder
from .types import Passage

_SCORE_CACHE_SIZE = 512


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
        try:
            numpy = importlib.import_module("numpy")
        except ImportError as error:
            raise ImportError(
                "BruteForceProvider requires the 'eval' extra: "
                "pip install 'map-matched-retrieval[eval]'"
            ) from error
        embedding_matrix = numpy.asarray(passage_embeddings, dtype="float64")
        shape = getattr(embedding_matrix, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("passage_embeddings must be a two-dimensional matrix")
        self._passage_ids = ids
        self._embedding_matrix = embedding_matrix
        self._embedding_dimension = shape[1]
        self._numpy = numpy
        self._embed_query = embed_query
        self._query_embedding_cache: dict[str, tuple[float, ...]] = {}
        self._score_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    @property
    def passage_count(self) -> int:
        return len(self._passage_ids)

    def candidates(self, query: str, limit: int) -> list[ScoredCandidate]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        query_embedding = self._query_embedding_cache.get(query)
        if query_embedding is None:
            query_embedding = tuple(self._embed_query.embed_query(query))
            self._query_embedding_cache[query] = query_embedding
        if len(query_embedding) != self._embedding_dimension:
            raise ValueError("query and passage embedding dimensions must match")
        scores = self._score_cache.get(query)
        if scores is None:
            query_vector = self._numpy.asarray(query_embedding, dtype="float64")
            raw_scores = self._embedding_matrix @ query_vector
            tolist = getattr(raw_scores, "tolist", None)
            if not callable(tolist):
                raise TypeError("NumPy returned invalid retrieval scores")
            score_values = tolist()
            if not isinstance(score_values, list):
                raise TypeError("NumPy returned invalid retrieval scores")
            scores = tuple(float(score) for score in score_values)
            self._score_cache[query] = scores
            if len(self._score_cache) > _SCORE_CACHE_SIZE:
                self._score_cache.popitem(last=False)
        else:
            self._score_cache.move_to_end(query)
        effective_limit = min(limit, len(self._passage_ids))

        def ranking_key(index: int) -> tuple[float, str]:
            return -scores[index], self._passage_ids[index]

        if effective_limit == len(self._passage_ids):
            ranked_indices = sorted(range(len(self._passage_ids)), key=ranking_key)
        else:
            ranked_indices = heapq.nsmallest(
                effective_limit,
                range(len(self._passage_ids)),
                key=ranking_key,
            )
        return [
            ScoredCandidate(
                chunk_id=self._passage_ids[index],
                score=scores[index],
            )
            for index in ranked_indices
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


def build_section_graph(passages: Sequence[Passage]) -> InMemoryCorpusGraph:
    """A structured corpus graph: passages sharing a `group_key` (e.g. the same
    Wikipedia article / section) are connected; passages in different groups are
    unconnected, so their geodesic distance clamps to `maximum_distance` — a
    coherent-drill / expensive-jump prior. Passages without a group_key (or in a
    singleton group) are isolated nodes."""
    passage_ids = [passage.passage_id for passage in passages]
    groups: dict[str, list[str]] = {}
    for passage in passages:
        if passage.group_key is None:
            continue
        groups.setdefault(passage.group_key, []).append(passage.passage_id)
    edges: list[tuple[str, str]] = []
    for members in groups.values():
        anchor = members[0]
        for other in members[1:]:
            edges.append((anchor, other))
    return InMemoryCorpusGraph.from_edges(edges, nodes=passage_ids)
