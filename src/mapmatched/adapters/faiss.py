from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from types import ModuleType
from typing import Literal, Protocol

from ..models import ScoredCandidate

FAISSScoreMode = Literal["similarity", "distance"]
QueryEmbedder = Callable[[str], Sequence[float]]


class FAISSDependencyUnavailableError(ImportError):
    pass


class _SearchMatrix(Protocol):
    def __getitem__(self, index: int) -> _SearchMatrix: ...

    def tolist(self) -> list[object]: ...


class FAISSIndex(Protocol):
    d: int
    ntotal: int

    def search(
        self,
        query_vectors: object,
        limit: int,
    ) -> tuple[_SearchMatrix, _SearchMatrix]: ...


def _load_numpy() -> ModuleType:
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise FAISSDependencyUnavailableError(
            "FAISSProvider requires the 'faiss' extra: pip install 'map-matched-retrieval[faiss]'"
        ) from error


def _integer_attribute(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"FAISS index {name} must be an integer")
    return int(value)


class FAISSProvider:
    def __init__(
        self,
        index: FAISSIndex,
        chunk_ids: Sequence[str],
        embed_query: QueryEmbedder,
        *,
        score_mode: FAISSScoreMode = "similarity",
    ) -> None:
        ids = tuple(chunk_ids)
        if not ids:
            raise ValueError("chunk_ids must contain at least one ID")
        if any(not chunk_id for chunk_id in ids):
            raise ValueError("chunk IDs must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("chunk IDs must be unique")
        if score_mode not in ("similarity", "distance"):
            raise ValueError(f"unsupported FAISS score mode: {score_mode}")
        if not callable(embed_query):
            raise TypeError("embed_query must be callable")

        dimension = _integer_attribute(index.d, "dimension")
        if dimension <= 0:
            raise ValueError("FAISS index dimension must be greater than zero")
        indexed_count = _integer_attribute(index.ntotal, "ntotal")
        if indexed_count != len(ids):
            raise ValueError(
                f"FAISS index contains {indexed_count} vectors but received {len(ids)} chunk IDs"
            )

        self._index = index
        self._chunk_ids = ids
        self._embed_query = embed_query
        self._score_mode = score_mode
        self._dimension = dimension

    def candidates(self, query: str, limit: int) -> tuple[ScoredCandidate, ...]:
        if not query:
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        indexed_count = _integer_attribute(self._index.ntotal, "ntotal")
        if indexed_count != len(self._chunk_ids):
            raise RuntimeError("FAISS index size changed after provider construction")

        vector = self._query_vector(query)
        numpy = _load_numpy()
        query_vectors = numpy.asarray([vector], dtype="float32")
        result_limit = min(limit, len(self._chunk_ids))
        score_matrix, index_matrix = self._index.search(query_vectors, result_limit)
        score_values = score_matrix[0].tolist()
        index_values = index_matrix[0].tolist()
        if len(score_values) != len(index_values):
            raise RuntimeError("FAISS returned different score and index counts")

        ranked: list[tuple[ScoredCandidate, int]] = []
        seen_indices: set[int] = set()
        for raw_score, raw_index in zip(score_values, index_values, strict=True):
            index_position = _integer_attribute(raw_index, "result index")
            if index_position < 0 or index_position >= len(self._chunk_ids):
                raise RuntimeError(f"FAISS returned invalid result index {index_position}")
            if index_position in seen_indices:
                raise RuntimeError(f"FAISS returned duplicate result index {index_position}")
            if not isinstance(raw_score, Real):
                raise RuntimeError("FAISS returned a non-numeric score")
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("FAISS returned a non-finite score")
            if self._score_mode == "distance":
                score = -score
            ranked.append(
                (
                    ScoredCandidate(
                        chunk_id=self._chunk_ids[index_position],
                        score=score,
                    ),
                    index_position,
                )
            )
            seen_indices.add(index_position)

        ranked.sort(key=lambda item: (-item[0].score, item[1]))
        return tuple(candidate for candidate, _ in ranked)

    def _query_vector(self, query: str) -> tuple[float, ...]:
        try:
            vector = tuple(float(value) for value in self._embed_query(query))
        except (TypeError, ValueError) as error:
            raise ValueError("embed_query must return a numeric vector") from error
        if len(vector) != self._dimension:
            raise ValueError(
                f"query vector has dimension {len(vector)}; expected {self._dimension}"
            )
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("query vector values must be finite")
        return vector
