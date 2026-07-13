from __future__ import annotations

import importlib
import math

import pytest

from mapmatched import GraphDependencyUnavailableError, KNNGraph
from mapmatched import knn as knn_module

pytest.importorskip("numpy")


def test_knn_graph_builds_weighted_cosine_edges() -> None:
    graph = KNNGraph.from_embeddings(
        ["a", "b", "c"],
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        neighbor_count=1,
        maximum_distance=3.0,
    )

    distance_ab = 1.0 - 0.8 / math.sqrt(0.8**2 + 0.2**2)
    distance_bc = 1.0 - 0.2 / math.sqrt(0.8**2 + 0.2**2)

    assert graph.distance("a", "b") == pytest.approx(distance_ab)
    assert graph.distance("b", "c") == pytest.approx(distance_bc)
    assert graph.distance("a", "c") == pytest.approx(distance_ab + distance_bc)


def test_identical_embeddings_use_minimum_edge_distance() -> None:
    graph = KNNGraph.from_embeddings(
        ["a", "b"],
        [[1.0, 0.0], [2.0, 0.0]],
        neighbor_count=1,
        minimum_edge_distance=0.01,
    )

    assert graph.distance("a", "b") == pytest.approx(0.01)


def test_tie_breaking_is_stable_across_input_order() -> None:
    first = KNNGraph.from_embeddings(
        ["a", "b", "c"],
        [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        neighbor_count=1,
        maximum_distance=3.0,
    )
    second = KNNGraph.from_embeddings(
        ["c", "a", "b"],
        [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
        neighbor_count=1,
        maximum_distance=3.0,
    )

    for source in ("a", "b", "c"):
        for target in ("a", "b", "c"):
            assert first.distance(source, target) == second.distance(source, target)


def test_knn_graph_connects_neighbors_across_similarity_blocks() -> None:
    chunk_ids = [f"chunk-{index:03d}" for index in range(258)]
    embeddings = [[1.0, index / 1000.0] for index in range(258)]

    graph = KNNGraph.from_embeddings(
        chunk_ids,
        embeddings,
        neighbor_count=1,
        maximum_distance=3.0,
    )

    assert graph.distance("chunk-255", "chunk-256") < graph.maximum_distance


@pytest.mark.parametrize(
    ("chunk_ids", "embeddings", "message"),
    [
        ([], [], "at least one"),
        (["a", "a"], [[1.0], [2.0]], "unique"),
        (["a"], [], "two-dimensional"),
        (["a"], [[]], "dimension"),
        (["a"], [[0.0, 0.0]], "nonzero"),
        (["a"], [[float("nan")]], "finite"),
    ],
)
def test_invalid_embedding_inputs_are_rejected(
    chunk_ids: list[str],
    embeddings: list[list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        KNNGraph.from_embeddings(chunk_ids, embeddings)


def test_missing_numpy_has_install_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = importlib.import_module

    def import_without_numpy(name: str, package: str | None = None) -> object:
        if name == "numpy":
            raise ImportError("missing")
        return original_import_module(name, package)

    monkeypatch.setattr(knn_module.importlib, "import_module", import_without_numpy)

    with pytest.raises(GraphDependencyUnavailableError, match=r"\[graph\]"):
        KNNGraph.from_embeddings(["a"], [[1.0]])
