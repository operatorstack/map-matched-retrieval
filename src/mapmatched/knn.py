from __future__ import annotations

import importlib
import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
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


def _load_scipy_sparse() -> tuple[ModuleType, ModuleType] | None:
    try:
        return (
            importlib.import_module("scipy.sparse"),
            importlib.import_module("scipy.sparse.csgraph"),
        )
    except ImportError:
        return None


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
    def __init__(
        self,
        edges: Iterable[GraphEdge] = (),
        *,
        nodes: Iterable[str] = (),
        directed: bool = False,
        maximum_distance: float = 10.0,
        distance_cache_size: int = 256,
    ) -> None:
        super().__init__(
            edges,
            nodes=nodes,
            directed=directed,
            maximum_distance=maximum_distance,
            distance_cache_size=distance_cache_size,
        )
        self._ordered_nodes = tuple(sorted(self._adjacency))
        self._node_indices = {chunk_id: index for index, chunk_id in enumerate(self._ordered_nodes)}
        self._sparse_distance_cache: OrderedDict[str, object] = OrderedDict()
        scipy_sparse = _load_scipy_sparse()
        if scipy_sparse is None:
            self._sparse_adjacency = None
            self._scipy_csgraph = None
            return
        sparse, self._scipy_csgraph = scipy_sparse
        rows: list[int] = []
        columns: list[int] = []
        distances: list[float] = []
        for source_id, neighbors in self._adjacency.items():
            source_index = self._node_indices[source_id]
            for target_id, distance in neighbors.items():
                rows.append(source_index)
                columns.append(self._node_indices[target_id])
                distances.append(distance)
        csr_matrix = getattr(sparse, "csr_matrix", None)
        if not callable(csr_matrix):
            self._sparse_adjacency = None
            self._scipy_csgraph = None
            return
        self._sparse_adjacency = csr_matrix(
            (distances, (rows, columns)),
            shape=(len(self._ordered_nodes), len(self._ordered_nodes)),
        )

    def distance(self, source_chunk_id: str, target_chunk_id: str) -> float:
        if self._sparse_adjacency is None or self._scipy_csgraph is None:
            return super().distance(source_chunk_id, target_chunk_id)
        if not source_chunk_id or not target_chunk_id:
            raise ValueError("distance chunk IDs must not be empty")
        if source_chunk_id == target_chunk_id:
            return 0.0
        target_index = self._node_indices.get(target_chunk_id)
        if target_index is None:
            return self._maximum_distance
        distances = self._cached_sparse_distances(source_chunk_id)
        if distances is not None:
            return self._sparse_distance_value(distances, target_index)
        if not self._directed:
            reverse_distances = self._cached_sparse_distances(target_chunk_id)
            source_index = self._node_indices.get(source_chunk_id)
            if reverse_distances is not None and source_index is not None:
                return self._sparse_distance_value(reverse_distances, source_index)
        distances = self._compute_sparse_distances(source_chunk_id, self._maximum_distance)
        if distances is None:
            return self._maximum_distance
        self._sparse_distance_cache[source_chunk_id] = distances
        self._sparse_distance_cache.move_to_end(source_chunk_id)
        if len(self._sparse_distance_cache) > self._distance_cache_size:
            self._sparse_distance_cache.popitem(last=False)
        return self._sparse_distance_value(distances, target_index)

    def _cached_sparse_distances(self, source: str) -> object | None:
        distances = self._sparse_distance_cache.get(source)
        if distances is not None:
            self._sparse_distance_cache.move_to_end(source)
        return distances

    def _sparse_distance_value(self, distances: object, target_index: int) -> float:
        get_item = getattr(distances, "__getitem__", None)
        if not callable(get_item):
            return self._maximum_distance
        distance = float(get_item(target_index))
        if not math.isfinite(distance):
            return self._maximum_distance
        return min(distance, self._maximum_distance)

    def _compute_sparse_distances(self, source: str, cutoff: float) -> object | None:
        if self._sparse_adjacency is None or self._scipy_csgraph is None:
            return None
        source_index = self._node_indices.get(source)
        if source_index is None:
            return None
        dijkstra = getattr(self._scipy_csgraph, "dijkstra", None)
        if not callable(dijkstra):
            return None
        return dijkstra(
            self._sparse_adjacency,
            directed=self._directed,
            indices=source_index,
            limit=cutoff,
        )

    def _bounded_distances(self, source: str, cutoff: float) -> dict[str, float]:
        if self._sparse_adjacency is None or self._scipy_csgraph is None:
            return super()._bounded_distances(source, cutoff)
        raw_distances = self._compute_sparse_distances(source, cutoff)
        if raw_distances is None:
            return {source: 0.0}
        tolist = getattr(raw_distances, "tolist", None)
        if not callable(tolist):
            return super()._bounded_distances(source, cutoff)
        values = tolist()
        if not isinstance(values, list):
            return super()._bounded_distances(source, cutoff)
        return {
            chunk_id: float(distance)
            for chunk_id, distance in zip(self._ordered_nodes, values, strict=True)
            if isinstance(distance, (int, float)) and math.isfinite(distance) and distance <= cutoff
        }

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
