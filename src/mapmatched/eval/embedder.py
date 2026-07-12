from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> Sequence[float]: ...


class PassageEmbedder(Protocol):
    def embed_passage(self, passage_id: str, text: str) -> Sequence[float]: ...


class TextEmbedder(QueryEmbedder, PassageEmbedder, Protocol):
    def embed_text(self, text: str) -> Sequence[float]: ...


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    magnitude = math.sqrt(math.fsum(value * value for value in vector))
    if magnitude == 0.0:
        if not vector:
            raise ValueError("embedding dimension must be greater than zero")
        unit = [0.0] * len(vector)
        unit[0] = 1.0
        return tuple(unit)
    return tuple(value / magnitude for value in vector)


def hash_embedding(text: str, *, dimension: int = 32) -> tuple[float, ...]:
    if dimension <= 0:
        raise ValueError("dimension must be greater than zero")
    values = [0.0] * dimension
    normalized_text = text.strip().lower()
    if not normalized_text:
        return _normalize(values)
    for token in normalized_text.split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign
    return _normalize(values)


class DeterministicHashEmbedder:
    def __init__(self, *, dimension: int = 32) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension

    @property
    def name(self) -> str:
        return f"deterministic-hash-{self.dimension}"

    def embed_text(self, text: str) -> tuple[float, ...]:
        return hash_embedding(text, dimension=self.dimension)

    def embed_query(self, query: str) -> tuple[float, ...]:
        return self.embed_text(query)

    def embed_passage(self, passage_id: str, text: str) -> tuple[float, ...]:
        del passage_id
        return self.embed_text(text)


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    return math.fsum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
