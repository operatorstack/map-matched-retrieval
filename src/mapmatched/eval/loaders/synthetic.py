from __future__ import annotations

from ..types import EvalConversation, EvalTurn, Passage


def load_synthetic_fixture() -> tuple[tuple[EvalConversation, ...], tuple[Passage, ...]]:
    passages = (
        Passage("alpha-overview", "alpha overview introduction map matched retrieval"),
        Passage("alpha-details", "alpha details transition graph distance decoding"),
        Passage("beta-overview", "beta overview unrelated topic switching benchmark"),
        Passage("beta-details", "beta details follow up underspecified query entropy"),
    )
    conversations = (
        EvalConversation(
            conversation_id="synthetic-drill",
            turns=(
                EvalTurn(
                    turn_index=0,
                    query="alpha overview introduction",
                    qrels={"alpha-overview": 3, "alpha-details": 1},
                ),
                EvalTurn(
                    turn_index=1,
                    query="details transition graph",
                    qrels={"alpha-details": 3, "alpha-overview": 1},
                ),
                EvalTurn(
                    turn_index=2,
                    query="decoding follow up",
                    qrels={"alpha-details": 3, "beta-details": 1},
                ),
            ),
        ),
        EvalConversation(
            conversation_id="synthetic-switch",
            turns=(
                EvalTurn(
                    turn_index=0,
                    query="beta overview topic",
                    qrels={"beta-overview": 3, "beta-details": 1},
                ),
                EvalTurn(
                    turn_index=1,
                    query="switch alpha overview",
                    qrels={"alpha-overview": 3, "beta-overview": 1},
                    resolved_query="alpha overview introduction map matched retrieval",
                ),
            ),
        ),
    )
    return conversations, passages
