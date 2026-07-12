from __future__ import annotations

import math
from typing import Literal, Sequence

from .models import DecoderCandidate, ScoredCandidate

ScoreNormalization = Literal["zscore", "center", "none"]


def normalize_candidates(
    candidates: Sequence[ScoredCandidate],
    method: ScoreNormalization = "zscore",
) -> tuple[DecoderCandidate, ...]:
    if not candidates:
        raise ValueError("candidate set must not be empty")
    chunk_ids = [candidate.chunk_id for candidate in candidates]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("candidate chunk_ids must be unique within a turn")

    scores = [candidate.score for candidate in candidates]
    mean = math.fsum(scores) / len(scores)
    if method == "zscore":
        variance = math.fsum((score - mean) ** 2 for score in scores) / len(scores)
        standard_deviation = math.sqrt(variance)
        normalized = (
            [0.0 for _ in scores]
            if standard_deviation == 0.0
            else [(score - mean) / standard_deviation for score in scores]
        )
    elif method == "center":
        normalized = [score - mean for score in scores]
    elif method == "none":
        normalized = scores
    else:
        raise ValueError(f"unsupported score normalization: {method}")

    return tuple(
        DecoderCandidate(
            chunk_id=candidate.chunk_id,
            raw_score=candidate.score,
            normalized_score=normalized_score,
        )
        for candidate, normalized_score in zip(candidates, normalized, strict=True)
    )


def softmax_entropy(scores: Sequence[float]) -> float:
    if not scores:
        raise ValueError("entropy requires at least one score")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("entropy scores must be finite")
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = math.fsum(exponentials)
    probabilities = [value / total for value in exponentials]
    return -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
