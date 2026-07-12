from mapmatched.eval import (
    DeterministicHashEmbedder,
    EvalConfig,
    MethodSpec,
    load_synthetic_fixture,
    run_eval,
)
from mapmatched.eval.baselines import MapMatchedMethodConfig


def test_all_methods_share_pointwise_entropy_slices() -> None:
    conversations, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    report = run_eval(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        methods=(
            MethodSpec(name="pointwise", transition_weight=0.0),
            MethodSpec(name="mapmatched", transition_weight=0.5),
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
            ranking_mode="full",
            graph_source="knn",
        ),
    )
    pointwise = next(method for method in report.methods if method.method_name == "pointwise")
    mapmatched = next(method for method in report.methods if method.method_name == "mapmatched")
    pointwise_slices = [turn.slice_name for turn in pointwise.turns]
    mapmatched_slices = [turn.slice_name for turn in mapmatched.turns]
    assert pointwise_slices == mapmatched_slices
