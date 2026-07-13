from mapmatched.eval.slices import (
    aggregate_slice_metrics,
    classify_turn_slice,
    compute_entropy_threshold,
    evaluate_claim,
)
from mapmatched.eval.types import MethodMetrics, SliceMetrics, TurnMetrics


def test_classify_turn_slice_uses_first_turn_as_standalone() -> None:
    assert (
        classify_turn_slice(
            turn_index=0,
            emission_entropy=2.0,
            entropy_threshold=1.0,
        )
        == "standalone"
    )


def test_classify_turn_slice_uses_entropy_threshold_for_follow_up_turns() -> None:
    assert (
        classify_turn_slice(
            turn_index=1,
            emission_entropy=1.5,
            entropy_threshold=1.0,
        )
        == "follow_up"
    )
    assert (
        classify_turn_slice(
            turn_index=1,
            emission_entropy=0.5,
            entropy_threshold=1.0,
        )
        == "standalone"
    )


def test_compute_entropy_threshold_uses_median() -> None:
    assert compute_entropy_threshold([0.0, 1.0, 2.0, 3.0]) == 1.5


def test_evaluate_claim_checks_follow_up_and_standalone_gates() -> None:
    methods = (
        _method("pointwise", follow_up=0.40, standalone=0.80),
        _method("mapmatched", follow_up=0.50, standalone=0.79),
    )
    verdict = evaluate_claim(
        methods,
        mapmatched_method_name="mapmatched",
        baseline_method_name="pointwise",
        standalone_tolerance=0.02,
        follow_up_min_delta=0.05,
    )
    assert verdict is not None
    assert verdict.follow_up_pass is True
    assert verdict.standalone_pass is True
    assert verdict.overall_pass is True


def test_aggregate_slice_metrics_skips_other_slices() -> None:
    turns = (
        TurnMetrics("conversation-1", 0, 1.0, 1.0, 1.0, 0.0, "standalone"),
        TurnMetrics("conversation-1", 1, 0.5, 0.5, 0.5, 1.0, "follow_up"),
    )
    follow_up = aggregate_slice_metrics(turns, "follow_up")
    assert follow_up.turn_count == 1
    assert follow_up.ndcg_at_3 == 0.5


def _method(name: str, *, follow_up: float, standalone: float) -> MethodMetrics:
    return MethodMetrics(
        method_name=name,
        transition_weight=0.0 if name == "pointwise" else 0.5,
        candidate_limit=20,
        fixed_lag=None,
        graph_mode="knn",
        slices=(
            SliceMetrics("follow_up", 1, follow_up, follow_up, follow_up),
            SliceMetrics("standalone", 1, standalone, standalone, standalone),
            SliceMetrics("all", 2, (follow_up + standalone) / 2, 0.0, 0.0),
        ),
        turns=(),
    )
