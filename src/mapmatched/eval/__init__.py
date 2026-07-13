from __future__ import annotations

from .ablations import default_method_grid, run_ablation_grid
from .embedder import DeterministicHashEmbedder, QueryEmbedder, hash_embedding
from .loaders import load_cast2019_micro, load_synthetic_fixture, load_topiocqa_micro
from .report import render_json, render_markdown_table
from .runner import MethodSpec, run_eval
from .types import (
    ComparisonSliceMetrics,
    EvalConfig,
    EvalConversation,
    EvalReport,
    MethodComparison,
    Passage,
)

__all__ = [
    "DeterministicHashEmbedder",
    "ComparisonSliceMetrics",
    "EvalConfig",
    "EvalConversation",
    "EvalReport",
    "MethodSpec",
    "MethodComparison",
    "Passage",
    "QueryEmbedder",
    "default_method_grid",
    "hash_embedding",
    "load_cast2019_micro",
    "load_synthetic_fixture",
    "load_topiocqa_micro",
    "render_json",
    "render_markdown_table",
    "run_ablation_grid",
    "run_eval",
]
