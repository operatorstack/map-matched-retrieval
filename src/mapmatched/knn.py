from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from types import ModuleType

from .graph import GraphEdge, InMemoryCorpusGraph

_SIMILARITY_BLOCK_SIZE = 256


class GraphDependencyUnavailableError(ImportError):
    pass


def _load_numpy() -> ModuleType:
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise GraphDependencyUnavailableError(
            "KNNGraph requires the 'graph' extra: pip install 'map-matched-retrieval[graph]'"
        ) from error


def _coerce_embedding_rows(
    embeddings: Sequence[Sequence[float]],
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    numpy = _load_numpy()
    matrix: object = numpy.asarray(embeddings, dtype="float64")
    ndim: object = getattr(matrix, "ndim", None)
    shape: object = getattr(matrix, "shape", None)
    tolist: object = getattr(matrix, "tolist", None)
    if ndim != 2 or not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if shape[0] != expected_count:
        raise ValueError(f"received {shape[0]} embeddings for {expected_count} chunk IDs")
    if shape[1] == 0:
        raise ValueError("embedding dimension must be greater than zero")
    if not callable(tolist):
        raise TypeError("NumPy returned an invalid embedding matrix")
    raw_rows: object = tolist()
    if not isinstance(raw_rows, list):
        raise TypeError("NumPy returned an invalid embedding matrix")

    rows: list[tuple[float, ...]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, list):
            raise TypeError("NumPy returned an invalid embedding row")
        row = tuple(float(value) for value in raw_row)
        if any(not math.isfinite(value) for value in row):
            raise ValueError("embedding values must be finite")
        magnitude = math.sqrt(sum(value * value for value in row))
        if magnitude == 0.0:
            raise ValueError("embedding vectors must have nonzero magnitude")
        rows.append(tuple(value / magnitude for value in row))
    return tuple(rows)


class KNNGraph(InMemoryCorpusGraph):
    @classmethod
    def from_embeddings(
        cls,
        chunk_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        *,
        neighbor_count: int = 10,
        maximum_distance: float = 10.0,
        minimum_edge_distance: float = 1e-6,
    ) -> KNNGraph:
        ids = tuple(chunk_ids)
        if not ids:
            raise ValueError("chunk_ids must contain at least one ID")
        if any(not chunk_id for chunk_id in ids):
            raise ValueError("chunk IDs must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("chunk IDs must be unique")
        if neighbor_count <= 0:
            raise ValueError("neighbor_count must be greater than zero")
        if not math.isfinite(maximum_distance) or maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be finite and greater than zero")
        if (
            not math.isfinite(minimum_edge_distance)
            or minimum_edge_distance <= 0.0
            or minimum_edge_distance > maximum_distance
        ):
            raise ValueError(
                "minimum_edge_distance must be finite, greater than zero, "
                "and no greater than maximum_distance"
            )

        numpy = _load_numpy()
        normalized_rows = _coerce_embedding_rows(embeddings, len(ids))
        normalized = numpy.asarray(normalized_rows, dtype="float64")
        del normalized_rows
        edge_distances: dict[tuple[int, int], float] = {}
        effective_count = min(neighbor_count, max(0, len(ids) - 1))
        for source_start in range(0, len(ids), _SIMILARITY_BLOCK_SIZE):
            source_end = min(source_start + _SIMILARITY_BLOCK_SIZE, len(ids))
            similarities = normalized[source_start:source_end] @ normalized.T
            numpy.clip(similarities, -1.0, 1.0, out=similarities)
            for block_index, source_index in enumerate(range(source_start, source_end)):
                if effective_count == 0:
                    continue
                source_similarities = similarities[block_index]
                source_similarities[source_index] = float("-inf")
                partition = numpy.argpartition(
                    -source_similarities,
                    effective_count - 1,
                )[:effective_count]
                cutoff_similarity = float(numpy.min(source_similarities[partition]))
                candidate_indices = numpy.flatnonzero(
                    source_similarities >= cutoff_similarity
                ).tolist()
                ranked_neighbors = sorted(
                    (
                        (
                            1.0 - float(source_similarities[target_index]),
                            ids[target_index],
                            target_index,
                        )
                        for target_index in candidate_indices
                    ),
                    key=lambda item: (item[0], item[1]),
                )
                for cosine_distance, _, target_index in ranked_neighbors[:effective_count]:
                    pair = (
                        (source_index, target_index)
                        if source_index < target_index
                        else (target_index, source_index)
                    )
                    edge_distance = min(
                        max(cosine_distance, minimum_edge_distance),
                        maximum_distance,
                    )
                    existing = edge_distances.get(pair)
                    if existing is None or edge_distance < existing:
                        edge_distances[pair] = edge_distance

        edges = (
            GraphEdge(ids[source_index], ids[target_index], distance)
            for (source_index, target_index), distance in sorted(edge_distances.items())
        )
        return cls(
            edges,
            nodes=ids,
            directed=False,
            maximum_distance=maximum_distance,
        )
