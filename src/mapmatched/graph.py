from __future__ import annotations

import heapq
import math
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
    ) -> None:
        if not math.isfinite(maximum_distance) or maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be finite and greater than zero")
        self._maximum_distance = maximum_distance
        self._directed = directed
        self._adjacency: dict[str, dict[str, float]] = {}
        self._distance_cache: dict[tuple[str, str], float] = {}

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
    ) -> InMemoryCorpusGraph:
        return cls(
            (GraphEdge(source, target) for source, target in edges),
            nodes=nodes,
            directed=directed,
            maximum_distance=maximum_distance,
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

    def _cache_key(self, source: str, target: str) -> tuple[str, str]:
        if self._directed or source <= target:
            return source, target
        return target, source

    def distance(self, source_chunk_id: str, target_chunk_id: str) -> float:
        if not source_chunk_id or not target_chunk_id:
            raise ValueError("distance chunk IDs must not be empty")
        if source_chunk_id == target_chunk_id:
            return 0.0
        cache_key = self._cache_key(source_chunk_id, target_chunk_id)
        cached = self._distance_cache.get(cache_key)
        if cached is not None:
            return cached
        distances = self._bounded_distances(source_chunk_id, self._maximum_distance)
        for chunk_id in self._adjacency:
            distance = min(
                distances.get(chunk_id, self._maximum_distance),
                self._maximum_distance,
            )
            self._distance_cache[self._cache_key(source_chunk_id, chunk_id)] = distance
        if cache_key not in self._distance_cache:
            self._distance_cache[cache_key] = min(
                distances.get(target_chunk_id, self._maximum_distance),
                self._maximum_distance,
            )
        return self._distance_cache[cache_key]

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
