#!/usr/bin/env python3
from __future__ import annotations

from mapmatched.eval import (
    DeterministicHashEmbedder,
    EvalConfig,
    load_synthetic_fixture,
    render_markdown_table,
    run_ablation_grid,
)


def main() -> None:
    conversations, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    report = run_ablation_grid(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        eval_config=EvalConfig(
            benchmark="synthetic",
            tier="synthetic",
            embedder_name=embedder.name,
            recall_k=10,
            entropy_threshold=None,
            standalone_tolerance=0.02,
            follow_up_min_delta=0.0,
        ),
        transition_weights=(0.0, 0.5),
        include_mmr=True,
    )
    print(render_markdown_table(report))


if __name__ == "__main__":
    main()
