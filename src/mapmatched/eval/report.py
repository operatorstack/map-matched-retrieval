from __future__ import annotations

import json

from .types import EvalReport, MethodComparison, MethodMetrics, SliceName


def render_markdown_table(report: EvalReport) -> str:
    metadata = [
        f"Benchmark: `{report.config.benchmark}`",
        f"Tier: `{report.config.tier}`",
        f"Embedder: `{report.config.embedder_name}`",
    ]
    if report.config.profile is not None:
        metadata.append(f"Profile: `{report.config.profile}`")
    lines = [
        "## Benchmark results (dev slice)",
        "",
        " · ".join(metadata),
        "",
    ]
    if report.config.dataset_filename is not None:
        lines.extend(
            [
                f"Data: `{report.config.dataset_filename}` · "
                f"SHA-256: `{report.config.dataset_sha256}` · "
                f"Conversations: `{len(report.config.conversation_ids)}`",
                "",
            ]
        )
    lines.extend(
        [
            "| Benchmark | Slice | Method | β | nDCG@3 | nDCG@3 95% CI | "
            "nDCG@5 | Recall | Δ vs β=0 (95% CI) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    baseline_by_slice = _baseline_ndcg(report.methods)
    comparison_by_slice = _comparison_delta_cis(report.comparisons)
    for method in report.methods:
        for slice_metrics in method.slices:
            if slice_metrics.slice_name == "all":
                continue
            if slice_metrics.turn_count == 0:
                continue
            baseline = baseline_by_slice.get(
                (method.candidate_limit, method.fixed_lag, slice_metrics.slice_name),
            )
            delta_ci = comparison_by_slice.get(
                (
                    method.method_name,
                    method.transition_weight,
                    method.candidate_limit,
                    method.fixed_lag,
                    method.graph_mode,
                    slice_metrics.slice_name,
                )
            )
            delta = _format_delta(slice_metrics.ndcg_at_3, baseline, delta_ci)
            beta = "—" if method.transition_weight is None else f"{method.transition_weight:.2f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        report.config.benchmark,
                        slice_metrics.slice_name,
                        method.method_name,
                        beta,
                        f"{slice_metrics.ndcg_at_3:.3f}",
                        _format_ci(slice_metrics.ndcg_at_3_ci),
                        f"{slice_metrics.ndcg_at_5:.3f}",
                        f"{slice_metrics.recall_at_k:.3f}",
                        delta,
                    ]
                )
                + " |"
            )
    lines.append("")
    if report.verdict is not None:
        lines.extend(
            [
                "### Claim check",
                "",
                f"- Follow-up lift vs β=0: `{report.verdict.follow_up_lift:+.3f}` "
                f"({'pass' if report.verdict.follow_up_pass else 'fail'})",
                f"- Standalone delta vs β=0: `{report.verdict.standalone_delta:+.3f}` "
                f"({'pass' if report.verdict.standalone_pass else 'fail'})",
                f"- Overall: `{'pass' if report.verdict.overall_pass else 'fail'}`",
                "",
            ]
        )
    return "\n".join(lines)


def render_json(report: EvalReport, *, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, sort_keys=True)


def _baseline_ndcg(
    methods: tuple[MethodMetrics, ...],
) -> dict[tuple[int, int | None, SliceName], float]:
    baseline: dict[tuple[int, int | None, SliceName], float] = {}
    for method in methods:
        if method.method_name != "pointwise":
            continue
        if method.transition_weight not in (0.0, None):
            continue
        for slice_metrics in method.slices:
            if slice_metrics.slice_name == "all":
                continue
            baseline[(method.candidate_limit, method.fixed_lag, slice_metrics.slice_name)] = (
                slice_metrics.ndcg_at_3
            )
    return baseline


def _comparison_delta_cis(
    comparisons: tuple[MethodComparison, ...],
) -> dict[
    tuple[str, float | None, int, int | None, str, SliceName],
    tuple[float, float] | None,
]:
    by_slice: dict[
        tuple[str, float | None, int, int | None, str, SliceName],
        tuple[float, float] | None,
    ] = {}
    for comparison in comparisons:
        for slice_metrics in comparison.slices:
            by_slice[
                (
                    comparison.method_name,
                    comparison.transition_weight,
                    comparison.candidate_limit,
                    comparison.fixed_lag,
                    comparison.graph_mode,
                    slice_metrics.slice_name,
                )
            ] = slice_metrics.ndcg_at_3_delta_ci
    return by_slice


def _format_delta(
    value: float,
    baseline: float | None,
    ci: tuple[float, float] | None,
) -> str:
    if baseline is None:
        return "—"
    delta = f"{value - baseline:+.3f}"
    if ci is None:
        return delta
    return f"{delta} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _format_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "—"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
