from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..types import EvalConversation


class ConversationQueryRewriter(Protocol):
    def rewrite(self, *, history: Sequence[str], query: str) -> str: ...


def rewrite_conversation_queries(
    conversation: EvalConversation,
    rewriter: ConversationQueryRewriter,
) -> tuple[str, ...]:
    history: list[str] = []
    rewritten_queries: list[str] = []
    for turn in conversation.turns:
        rewritten_query = rewriter.rewrite(history=tuple(history), query=turn.query).strip()
        if not rewritten_query:
            raise RuntimeError(
                f"query rewriter returned an empty query for "
                f"{conversation.conversation_id} turn {turn.turn_index}"
            )
        rewritten_queries.append(rewritten_query)
        history.append(turn.query)
    return tuple(rewritten_queries)
