from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class CorpusGraph(Protocol):
    @property
    def maximum_distance(self) -> float: ...

    def distance(self, source_chunk_id: str, target_chunk_id: str) -> float: ...

    def neighborhood(self, chunk_id: str, radius: float) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source_chunk_id: str
    target_chunk_id: str
    distance: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_chunk_id or not self.target_chunk_id:
            raise ValueError("graph edge chunk IDs must not be empty")
        if not math.isfinite(self.distance) or self.distance <= 0.0:
            raise ValueError("graph edge distance must be finite and greater than zero")


class InMemoryCorpusGraph:
    def __init__(
        self,
        edges: Iterable[GraphEdge] = (),
        *,
        nodes: Iterable[str] = (),
        directed: bool = False,
        maximum_distance: float = 10.0,
        distance_cache_size: int = 256,
    ) -> None:
        if not math.isfinite(maximum_distance) or maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be finite and greater than zero")
        if distance_cache_size <= 0:
            raise ValueError("distance_cache_size must be greater than zero")
        self._maximum_distance = maximum_distance
        self._directed = directed
        self._distance_cache_size = distance_cache_size
        self._adjacency: dict[str, dict[str, float]] = {}
        self._distance_cache: OrderedDict[str, dict[str, float]] = OrderedDict()

        for node in nodes:
            if not node:
                raise ValueError("graph node IDs must not be empty")
            self._adjacency.setdefault(node, {})
        for edge in edges:
            self._add_edge(edge.source_chunk_id, edge.target_chunk_id, edge.distance)
            if not directed:
                self._add_edge(edge.target_chunk_id, edge.source_chunk_id, edge.distance)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str]],
        *,
        nodes: Iterable[str] = (),
        directed: bool = False,
        maximum_distance: float = 10.0,
        distance_cache_size: int = 256,
    ) -> InMemoryCorpusGraph:
        return cls(
            (GraphEdge(source, target) for source, target in edges),
            nodes=nodes,
            directed=directed,
            maximum_distance=maximum_distance,
            distance_cache_size=distance_cache_size,
        )

    @property
    def maximum_distance(self) -> float:
        return self._maximum_distance

    def _add_edge(self, source: str, target: str, distance: float) -> None:
        neighbors = self._adjacency.setdefault(source, {})
        self._adjacency.setdefault(target, {})
        current = neighbors.get(target)
        if current is None or distance < current:
            neighbors[target] = distance

    def distance(self, source_chunk_id: str, target_chunk_id: str) -> float:
        if not source_chunk_id or not target_chunk_id:
            raise ValueError("distance chunk IDs must not be empty")
        if source_chunk_id == target_chunk_id:
            return 0.0
        distances = self._cached_distances(source_chunk_id)
        if distances is not None:
            return min(
                distances.get(target_chunk_id, self._maximum_distance),
                self._maximum_distance,
            )
        if not self._directed:
            reverse_distances = self._cached_distances(target_chunk_id)
            if reverse_distances is not None:
                return min(
                    reverse_distances.get(source_chunk_id, self._maximum_distance),
                    self._maximum_distance,
                )
        distances = self._bounded_distances(source_chunk_id, self._maximum_distance)
        self._distance_cache[source_chunk_id] = distances
        self._distance_cache.move_to_end(source_chunk_id)
        if len(self._distance_cache) > self._distance_cache_size:
            self._distance_cache.popitem(last=False)
        return min(
            distances.get(target_chunk_id, self._maximum_distance),
            self._maximum_distance,
        )

    def _cached_distances(self, source: str) -> dict[str, float] | None:
        distances = self._distance_cache.get(source)
        if distances is not None:
            self._distance_cache.move_to_end(source)
        return distances

    def neighborhood(self, chunk_id: str, radius: float) -> tuple[str, ...]:
        if not chunk_id:
            raise ValueError("neighborhood chunk_id must not be empty")
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("neighborhood radius must be finite and nonnegative")
        effective_radius = min(radius, self._maximum_distance)
        distances = self._bounded_distances(chunk_id, effective_radius)
        ordered = sorted(
            (
                (distance, neighbor)
                for neighbor, distance in distances.items()
                if neighbor != chunk_id and distance <= effective_radius
            ),
            key=lambda item: (item[0], item[1]),
        )
        return tuple(neighbor for _, neighbor in ordered)

    def _bounded_distances(self, source: str, cutoff: float) -> dict[str, float]:
        if source not in self._adjacency:
            return {source: 0.0}
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, chunk_id = heapq.heappop(queue)
            if distance != distances[chunk_id]:
                continue
            for neighbor, edge_distance in sorted(self._adjacency[chunk_id].items()):
                candidate_distance = distance + edge_distance
                if candidate_distance > cutoff:
                    continue
                previous_distance = distances.get(neighbor)
                if previous_distance is None or candidate_distance < previous_distance:
                    distances[neighbor] = candidate_distance
                    heapq.heappush(queue, (candidate_distance, neighbor))
        return distances
