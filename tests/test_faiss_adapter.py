from __future__ import annotations

import importlib
from collections.abc import Sequence

import pytest

from mapmatched import FAISSDependencyUnavailableError, FAISSProvider
from mapmatched.adapters import faiss as faiss_adapter

faiss = pytest.importorskip("faiss")
numpy = pytest.importorskip("numpy")


def test_similarity_index_maps_ids_and_orders_ties_by_index() -> None:
    index = faiss.IndexFlatIP(2)
    index.add(numpy.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype="float32"))
    provider = FAISSProvider(
        index,
        ["exact", "first-tie", "second-tie"],
        lambda query: [1.0, 0.0],
    )

    candidates = provider.candidates("query", 3)

    assert tuple(candidate.chunk_id for candidate in candidates) == (
        "exact",
        "first-tie",
        "second-tie",
    )
    assert tuple(candidate.score for candidate in candidates) == (1.0, 0.0, 0.0)


def test_distance_index_converts_lower_distance_to_higher_score() -> None:
    index = faiss.IndexFlatL2(2)
    index.add(numpy.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    provider = FAISSProvider(
        index,
        ["near", "far"],
        lambda query: [0.9, 0.1],
        score_mode="distance",
    )

    candidates = provider.candidates("query", 2)

    assert tuple(candidate.chunk_id for candidate in candidates) == ("near", "far")
    assert candidates[0].score > candidates[1].score


def test_provider_validates_index_mapping_and_query_vectors() -> None:
    index = faiss.IndexFlatIP(2)
    index.add(numpy.asarray([[1.0, 0.0]], dtype="float32"))

    with pytest.raises(ValueError, match=r"1 vectors.*2 chunk IDs"):
        FAISSProvider(index, ["a", "b"], lambda query: [1.0, 0.0])

    wrong_dimension = FAISSProvider(index, ["a"], lambda query: [1.0])
    with pytest.raises(ValueError, match=r"dimension 1.*expected 2"):
        wrong_dimension.candidates("query", 1)

    nonfinite = FAISSProvider(index, ["a"], lambda query: [float("nan"), 0.0])
    with pytest.raises(ValueError, match="finite"):
        nonfinite.candidates("query", 1)


class FixtureIndex:
    d = 2
    ntotal = 1

    def search(self, query_vectors: object, limit: int) -> tuple[object, object]:
        raise AssertionError("search must not run without NumPy")


class FixtureMatrix:
    def __init__(self, values: list[object] | list[list[object]]) -> None:
        self._values = values

    def __getitem__(self, index: int) -> FixtureMatrix:
        value = self._values[index]
        if not isinstance(value, list):
            raise TypeError("matrix row must be a list")
        return FixtureMatrix(value)

    def tolist(self) -> list[object]:
        return list(self._values)


class SearchResultIndex:
    d = 2

    def __init__(
        self,
        scores: list[object],
        indices: list[object],
        *,
        indexed_count: int,
    ) -> None:
        self.ntotal = indexed_count
        self._scores = scores
        self._indices = indices

    def search(
        self,
        query_vectors: object,
        limit: int,
    ) -> tuple[FixtureMatrix, FixtureMatrix]:
        return FixtureMatrix([self._scores]), FixtureMatrix([self._indices])


@pytest.mark.parametrize(
    ("scores", "indices", "indexed_count", "chunk_ids", "message"),
    [
        ([float("nan")], [0], 1, ["a"], "non-finite"),
        ([1.0], [2], 1, ["a"], "invalid result index"),
        ([1.0, 0.5], [0, 0], 2, ["a", "b"], "duplicate result index"),
    ],
)
def test_invalid_search_results_are_rejected(
    scores: list[object],
    indices: list[object],
    indexed_count: int,
    chunk_ids: list[str],
    message: str,
) -> None:
    provider = FAISSProvider(
        SearchResultIndex(scores, indices, indexed_count=indexed_count),
        chunk_ids,
        lambda query: [1.0, 0.0],
    )

    with pytest.raises(RuntimeError, match=message):
        provider.candidates("query", len(chunk_ids))


def test_missing_optional_dependency_has_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FAISSProvider(FixtureIndex(), ["a"], lambda query: [1.0, 0.0])
    original_import_module = importlib.import_module

    def import_without_numpy(name: str, package: str | None = None) -> object:
        if name == "numpy":
            raise ImportError("missing")
        return original_import_module(name, package)

    monkeypatch.setattr(faiss_adapter.importlib, "import_module", import_without_numpy)

    with pytest.raises(FAISSDependencyUnavailableError, match=r"\[faiss\]"):
        provider.candidates("query", 1)


@pytest.mark.parametrize("chunk_ids", [[], ["a", "a"], [""]])
def test_invalid_chunk_ids_are_rejected(chunk_ids: Sequence[str]) -> None:
    index = faiss.IndexFlatIP(2)
    if chunk_ids:
        index.add(numpy.zeros((len(chunk_ids), 2), dtype="float32"))

    with pytest.raises(ValueError):
        FAISSProvider(index, chunk_ids, lambda query: [1.0, 0.0])
