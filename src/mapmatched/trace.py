from __future__ import annotations

from collections.abc import Sequence

from .models import RetrievalTrace


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)
    ]
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(
            value.ljust(widths[index]) if index == 1 else value.rjust(widths[index])
            for index, value in enumerate(row)
        )
        for row in rows
    ]
    return [header_line, divider, *body]


def render_trace(trace: RetrievalTrace) -> str:
    path = " -> ".join(trace.path_chunk_ids) if trace.path_chunk_ids else "(empty)"
    revisions = (
        ", ".join(str(index) for index in trace.revised_prior_indices)
        if trace.revised_prior_indices
        else "none"
    )
    if trace.fixed_lag is None:
        mode = "full"
        commitment = "none"
    else:
        mode = f"fixed-lag {trace.fixed_lag}"
        commitment = (
            str(trace.committed_through_index)
            if trace.committed_through_index is not None
            else "none"
        )

    lines = [
        f"path: {path}",
        f"mode: {mode}",
        f"revised prior turns: {revisions}",
        f"committed through turn: {commitment}",
    ]
    if not trace.steps:
        return "\n".join(lines)

    headers = (
        "turn",
        "chunk",
        "raw",
        "normalized",
        "emission",
        "distance",
        "transition",
        "entropy",
        "margin",
        "cumulative",
    )
    rows = [
        (
            str(step.turn_index),
            step.chunk_id,
            _format_number(step.raw_emission_score),
            _format_number(step.normalized_emission_score),
            _format_number(step.weighted_emission_score),
            _format_number(step.graph_distance),
            _format_number(step.weighted_transition_cost),
            _format_number(step.emission_entropy),
            _format_number(step.normalized_emission_margin),
            _format_number(step.cumulative_score),
        )
        for step in trace.steps
    ]
    return "\n".join([*lines, "", *_format_table(headers, rows)])
