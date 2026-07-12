from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose

from .metrics import mean
from .types import ClaimVerdict, MethodMetrics, SliceMetrics, SliceName, TurnMetrics


def compute_entropy_threshold(entropies: Sequence[float | None]) -> float:
    finite = [value for value in entropies if value is not None]
    if not finite:
        return 0.0
    sorted_values = sorted(finite)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def classify_turn_slice(
    *,
    turn_index: int,
    emission_entropy: float | None,
    entropy_threshold: float,
) -> SliceName:
    if turn_index == 0:
        return "standalone"
    if emission_entropy is None:
        return "follow_up"
    if emission_entropy >= entropy_threshold:
        return "follow_up"
    return "standalone"


def aggregate_slice_metrics(
    turns: Sequence[TurnMetrics],
    slice_name: SliceName,
    *,
    ndcg_at_3_ci: tuple[float, float] | None = None,
) -> SliceMetrics:
    selected = [turn for turn in turns if turn.slice_name == slice_name]
    return SliceMetrics(
        slice_name=slice_name,
        turn_count=len(selected),
        ndcg_at_3=mean([turn.ndcg_at_3 for turn in selected]),
        ndcg_at_5=mean([turn.ndcg_at_5 for turn in selected]),
        recall_at_k=mean([turn.recall_at_k for turn in selected]),
        ndcg_at_3_ci=ndcg_at_3_ci,
    )


def build_method_metrics(
    *,
    method_name: str,
    transition_weight: float | None,
    candidate_limit: int,
    fixed_lag: int | None,
    graph_mode: str,
    turns: Sequence[TurnMetrics],
    ndcg_at_3_cis: Mapping[SliceName, tuple[float, float] | None] | None = None,
) -> MethodMetrics:
    cis = ndcg_at_3_cis or {}
    all_slice = SliceMetrics(
        slice_name="all",
        turn_count=len(turns),
        ndcg_at_3=mean([turn.ndcg_at_3 for turn in turns]),
        ndcg_at_5=mean([turn.ndcg_at_5 for turn in turns]),
        recall_at_k=mean([turn.recall_at_k for turn in turns]),
        ndcg_at_3_ci=cis.get("all"),
    )
    slice_metrics = (
        aggregate_slice_metrics(turns, "follow_up", ndcg_at_3_ci=cis.get("follow_up")),
        aggregate_slice_metrics(turns, "standalone", ndcg_at_3_ci=cis.get("standalone")),
        all_slice,
    )
    return MethodMetrics(
        method_name=method_name,
        transition_weight=transition_weight,
        candidate_limit=candidate_limit,
        fixed_lag=fixed_lag,
        graph_mode=graph_mode,
        slices=slice_metrics,
        turns=tuple(turns),
    )


def _slice_value(method: MethodMetrics, slice_name: SliceName) -> float:
    for slice_metrics in method.slices:
        if slice_metrics.slice_name == slice_name:
            return slice_metrics.ndcg_at_3
    return 0.0


def evaluate_claim(
    methods: Sequence[MethodMetrics],
    *,
    mapmatched_method_name: str,
    baseline_method_name: str,
    standalone_tolerance: float,
    follow_up_min_delta: float,
    mapmatched_transition_weight: float | None = None,
) -> ClaimVerdict | None:
    mapmatched = _select_method(
        methods,
        method_name=mapmatched_method_name,
        transition_weight=mapmatched_transition_weight,
    )
    baseline = _select_method(
        methods,
        method_name=baseline_method_name,
        transition_weight=0.0 if baseline_method_name == "pointwise" else None,
    )
    if mapmatched is None or baseline is None:
        return None

    follow_up_lift = _slice_value(mapmatched, "follow_up") - _slice_value(baseline, "follow_up")
    standalone_delta = _slice_value(mapmatched, "standalone") - _slice_value(baseline, "standalone")
    follow_up_pass = follow_up_lift >= follow_up_min_delta
    standalone_pass = standalone_delta >= -standalone_tolerance
    return ClaimVerdict(
        follow_up_lift=follow_up_lift,
        standalone_delta=standalone_delta,
        follow_up_pass=follow_up_pass,
        standalone_pass=standalone_pass,
        overall_pass=follow_up_pass and standalone_pass,
    )


def _select_method(
    methods: Sequence[MethodMetrics],
    *,
    method_name: str,
    transition_weight: float | None,
) -> MethodMetrics | None:
    matches = [method for method in methods if method.method_name == method_name]
    if transition_weight is not None:
        weighted = [
            method
            for method in matches
            if method.transition_weight is not None
            and isclose(method.transition_weight, transition_weight, rel_tol=1e-9, abs_tol=1e-12)
        ]
        if weighted:
            return weighted[0]
        return None
    if not matches:
        return None
    return matches[0]
