import pytest

from mapmatched.eval import (
    DeterministicHashEmbedder,
    EvalConfig,
    MethodSpec,
    load_synthetic_fixture,
    render_markdown_table,
    run_eval,
)
from mapmatched.eval.baselines import MapMatchedMethodConfig


def test_synthetic_eval_runs_end_to_end() -> None:
    conversations, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    report = run_eval(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        methods=(
            MethodSpec(name="pointwise", transition_weight=0.0),
            MethodSpec(name="mapmatched", transition_weight=0.5),
            MethodSpec(name="history_concat"),
            MethodSpec(name="maximal_marginal_relevance"),
        ),
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
            bootstrap_seed=42,
        ),
    )
    assert report.methods
    markdown = render_markdown_table(report)
    assert "Benchmark results" in markdown
    assert "pointwise" in markdown
    assert report.comparisons
    mapmatched_comparison = next(
        comparison for comparison in report.comparisons if comparison.method_name == "mapmatched"
    )
    follow_up = next(
        slice_metrics
        for slice_metrics in mapmatched_comparison.slices
        if slice_metrics.slice_name == "follow_up"
    )
    assert follow_up.ndcg_at_3_delta_ci is not None


def test_resolved_oracle_requires_every_turn_to_be_resolved() -> None:
    conversations, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()

    with pytest.raises(ValueError, match="resolved query for every turn"):
        run_eval(
            conversations=conversations,
            passages=passages,
            embedder=embedder,
            methods=(MethodSpec(name="resolved_oracle"),),
            config=MapMatchedMethodConfig(candidate_limit=4),
            eval_config=EvalConfig(
                benchmark="synthetic",
                tier="synthetic",
                embedder_name=embedder.name,
                recall_k=10,
                entropy_threshold=None,
                standalone_tolerance=0.02,
                follow_up_min_delta=0.0,
            ),
        )
