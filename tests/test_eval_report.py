from mapmatched.eval.report import render_json, render_markdown_table
from mapmatched.eval.types import EvalConfig, EvalReport, MethodMetrics, SliceMetrics


def test_report_json_and_markdown_are_deterministic() -> None:
    report = EvalReport(
        config=EvalConfig(
            benchmark="synthetic",
            tier="synthetic",
            embedder_name="deterministic-hash-32",
            recall_k=100,
            entropy_threshold=1.0,
            standalone_tolerance=0.02,
            follow_up_min_delta=0.0,
        ),
        methods=(
            MethodMetrics(
                method_name="pointwise",
                transition_weight=0.0,
                candidate_limit=20,
                fixed_lag=None,
                graph_mode="knn",
                slices=(
                    SliceMetrics("follow_up", 2, 0.4, 0.4, 0.4),
                    SliceMetrics("standalone", 1, 0.8, 0.8, 0.8),
                    SliceMetrics("all", 3, 0.533, 0.533, 0.533),
                ),
                turns=(),
            ),
        ),
        verdict=None,
    )
    json_payload = render_json(report)
    markdown = render_markdown_table(report)
    assert '"benchmark": "synthetic"' in json_payload
    assert "| synthetic | follow_up | pointwise |" in markdown
    assert "nDCG@3 95% CI" in markdown
