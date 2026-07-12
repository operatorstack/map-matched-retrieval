from __future__ import annotations

from collections.abc import Sequence

from .baselines import MapMatchedMethodConfig
from .embedder import TextEmbedder
from .runner import MethodSpec, run_eval
from .types import EvalConfig, EvalConversation, EvalReport, Passage


def default_method_grid(
    *,
    transition_weights: Sequence[float] = (0.0, 0.5, 1.0),
    include_mmr: bool = True,
    include_resolved_oracle: bool = False,
) -> tuple[MethodSpec, ...]:
    methods: list[MethodSpec] = [
        MethodSpec(name="pointwise", transition_weight=0.0),
    ]
    for beta in transition_weights:
        if beta == 0.0:
            continue
        methods.append(MethodSpec(name="mapmatched", transition_weight=beta))
    methods.append(MethodSpec(name="history_concat"))
    if include_mmr:
        methods.append(MethodSpec(name="maximal_marginal_relevance"))
    if include_resolved_oracle:
        methods.append(MethodSpec(name="resolved_oracle"))
    return tuple(methods)


def run_ablation_grid(
    *,
    conversations: Sequence[EvalConversation],
    passages: Sequence[Passage],
    embedder: TextEmbedder,
    eval_config: EvalConfig,
    transition_weights: Sequence[float] = (0.0, 0.5, 1.0),
    candidate_limit: int = 20,
    fixed_lag: int | None = None,
    include_mmr: bool = True,
    include_resolved_oracle: bool = False,
) -> EvalReport:
    methods = default_method_grid(
        transition_weights=transition_weights,
        include_mmr=include_mmr,
        include_resolved_oracle=include_resolved_oracle,
    )
    config = MapMatchedMethodConfig(
        candidate_limit=candidate_limit,
        fixed_lag=fixed_lag,
    )
    return run_eval(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        methods=methods,
        config=config,
        eval_config=eval_config,
    )
