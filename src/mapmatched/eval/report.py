from __future__ import annotations

import json

from .types import EvalReport, MethodMetrics, SliceName


def render_markdown_table(report: EvalReport) -> str:
    lines = [
        "## Benchmark results (dev slice)",
        "",
        f"Benchmark: `{report.config.benchmark}` · Tier: `{report.config.tier}` · "
        f"Embedder: `{report.config.embedder_name}`",
        "",
        "| Benchmark | Slice | Method | β | nDCG@3 | nDCG@3 95% CI | nDCG@5 | Recall | Δ vs β=0 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    baseline_by_slice = _baseline_ndcg(report.methods)
    for method in report.methods:
        for slice_metrics in method.slices:
            if slice_metrics.slice_name == "all":
                continue
            if slice_metrics.turn_count == 0:
                continue
            baseline = baseline_by_slice.get(
                (method.candidate_limit, method.fixed_lag, slice_metrics.slice_name),
            )
            delta = _format_delta(slice_metrics.ndcg_at_3, baseline)
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


def _format_delta(value: float, baseline: float | None) -> str:
    if baseline is None:
        return "—"
    return f"{value - baseline:+.3f}"


def _format_ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "—"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
