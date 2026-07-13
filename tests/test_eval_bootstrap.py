import pytest

from mapmatched.eval.bootstrap import (
    bootstrap_ndcg_at_3_ci,
    bootstrap_paired_ndcg_at_3_delta_ci,
    bootstrap_slice_cis,
)
from mapmatched.eval.types import SliceName, TurnMetrics


def _turn(
    ndcg_at_3: float,
    slice_name: SliceName,
    *,
    conversation_id: str = "conversation-1",
    turn_index: int = 0,
) -> TurnMetrics:
    return TurnMetrics(
        conversation_id=conversation_id,
        turn_index=turn_index,
        ndcg_at_3=ndcg_at_3,
        ndcg_at_5=ndcg_at_3,
        recall_at_k=ndcg_at_3,
        emission_entropy=1.0,
        slice_name=slice_name,
    )


def test_bootstrap_disabled_returns_none() -> None:
    conversation_turns = (
        (_turn(1.0, "follow_up"), _turn(0.0, "standalone")),
        (_turn(0.5, "follow_up"),),
    )
    assert bootstrap_ndcg_at_3_ci(conversation_turns, "follow_up", num_samples=0) is None


def test_bootstrap_is_deterministic_with_seed() -> None:
    conversation_turns = (
        (_turn(1.0, "follow_up"), _turn(0.0, "standalone")),
        (_turn(0.5, "follow_up"),),
        (_turn(0.25, "follow_up"), _turn(0.75, "standalone")),
    )
    first = bootstrap_ndcg_at_3_ci(
        conversation_turns,
        "follow_up",
        num_samples=200,
        seed=7,
    )
    second = bootstrap_ndcg_at_3_ci(
        conversation_turns,
        "follow_up",
        num_samples=200,
        seed=7,
    )
    assert first == second
    assert first is not None
    assert first[0] <= first[1]


def test_bootstrap_slice_cis_returns_all_slices() -> None:
    conversation_turns = (
        (_turn(1.0, "follow_up"), _turn(0.0, "standalone")),
        (_turn(0.5, "follow_up"),),
    )
    cis = bootstrap_slice_cis(conversation_turns, num_samples=50, seed=1)
    assert set(cis) == {"follow_up", "standalone", "all"}
    assert cis["follow_up"] is not None
    assert cis["follow_up"][0] <= cis["follow_up"][1]
    assert cis["all"] is not None
    assert cis["all"] != (0.0, 0.0)


def test_paired_bootstrap_positive_delta_excludes_zero() -> None:
    treatment = (
        (_turn(1.0, "follow_up", conversation_id="conversation-1"),),
        (_turn(0.8, "follow_up", conversation_id="conversation-2"),),
    )
    baseline = (
        (_turn(0.2, "follow_up", conversation_id="conversation-1"),),
        (_turn(0.3, "follow_up", conversation_id="conversation-2"),),
    )
    ci = bootstrap_paired_ndcg_at_3_delta_ci(
        treatment,
        baseline,
        "follow_up",
        num_samples=200,
        seed=7,
    )
    assert ci is not None
    assert ci[0] > 0.0


def test_paired_bootstrap_zero_delta_is_exact() -> None:
    treatment = (
        (_turn(0.2, "standalone", conversation_id="conversation-1"),),
        (_turn(0.8, "standalone", conversation_id="conversation-2"),),
    )
    ci = bootstrap_paired_ndcg_at_3_delta_ci(
        treatment,
        treatment,
        "standalone",
        num_samples=100,
        seed=11,
    )
    assert ci == (0.0, 0.0)


def test_paired_bootstrap_negative_delta_excludes_zero() -> None:
    treatment = (
        (_turn(0.1, "follow_up", conversation_id="conversation-1"),),
        (_turn(0.2, "follow_up", conversation_id="conversation-2"),),
    )
    baseline = (
        (_turn(0.8, "follow_up", conversation_id="conversation-1"),),
        (_turn(0.9, "follow_up", conversation_id="conversation-2"),),
    )
    ci = bootstrap_paired_ndcg_at_3_delta_ci(
        treatment,
        baseline,
        "follow_up",
        num_samples=100,
        seed=3,
    )
    assert ci is not None
    assert ci[1] < 0.0


def test_paired_bootstrap_rejects_misaligned_conversations() -> None:
    treatment = ((_turn(1.0, "follow_up", conversation_id="conversation-1"),),)
    baseline = ((_turn(1.0, "follow_up", conversation_id="conversation-2"),),)
    with pytest.raises(ValueError, match="conversation IDs do not match"):
        bootstrap_paired_ndcg_at_3_delta_ci(
            treatment,
            baseline,
            "follow_up",
            num_samples=10,
        )


def test_paired_bootstrap_ignores_conversations_without_requested_slice() -> None:
    treatment = (
        (_turn(0.8, "follow_up", conversation_id="conversation-1"),),
        (_turn(0.1, "standalone", conversation_id="conversation-2"),),
    )
    baseline = (
        (_turn(0.3, "follow_up", conversation_id="conversation-1"),),
        (_turn(0.9, "standalone", conversation_id="conversation-2"),),
    )
    ci = bootstrap_paired_ndcg_at_3_delta_ci(
        treatment,
        baseline,
        "follow_up",
        num_samples=50,
        seed=5,
    )
    assert ci == (0.5, 0.5)


def test_synthetic_eval_populates_bootstrap_cis() -> None:
    from mapmatched.eval import (
        DeterministicHashEmbedder,
        EvalConfig,
        MethodSpec,
        load_synthetic_fixture,
        run_eval,
    )
    from mapmatched.eval.baselines import MapMatchedMethodConfig

    conversations, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    report = run_eval(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        methods=(MethodSpec(name="pointwise", transition_weight=0.0),),
        config=MapMatchedMethodConfig(candidate_limit=4),
        eval_config=EvalConfig(
            benchmark="synthetic",
            tier="synthetic",
            embedder_name=embedder.name,
            recall_k=10,
            entropy_threshold=None,
            standalone_tolerance=0.02,
            follow_up_min_delta=0.0,
            bootstrap_samples=50,
            bootstrap_seed=99,
        ),
    )
    follow_up = next(
        slice_metrics
        for slice_metrics in report.methods[0].slices
        if slice_metrics.slice_name == "follow_up"
    )
    assert follow_up.ndcg_at_3_ci is not None
    assert follow_up.ndcg_at_3_ci[0] <= follow_up.ndcg_at_3 <= follow_up.ndcg_at_3_ci[1]
    assert report.methods[0].turns[0].conversation_id
