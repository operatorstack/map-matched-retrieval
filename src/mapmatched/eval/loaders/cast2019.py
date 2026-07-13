from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType

from ..types import EvalConversation, EvalTurn, Passage

# The judged TREC CAsT 2019 topics (ir-datasets id). The passage text lives in
# the CAsT document collection (MS MARCO + TREC CAR); ir-datasets downloads it
# on first use of docs_store(), which is multi-GB, so a full run is heavy.
_CAST_2019_DATASET = "trec-cast/v1/2019/judged"


class EvalDependencyUnavailableError(ImportError):
    pass


def _load_ir_datasets() -> ModuleType:
    try:
        return importlib.import_module("ir_datasets")
    except ImportError as error:
        raise EvalDependencyUnavailableError(
            "CAsT loader requires the 'eval' extra: pip install 'map-matched-retrieval[eval]'"
        ) from error


def _first_text_attr(obj: object, names: Sequence[str]) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def load_cast2019_micro() -> tuple[tuple[EvalConversation, ...], tuple[Passage, ...]]:
    ir_datasets = _load_ir_datasets()
    load = getattr(ir_datasets, "load", None)
    if not callable(load):
        raise EvalDependencyUnavailableError("ir_datasets.load is unavailable")
    dataset = load(_CAST_2019_DATASET)

    qrels_by_query: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        query_id = getattr(qrel, "query_id", None)
        doc_id = getattr(qrel, "doc_id", None)
        relevance = getattr(qrel, "relevance", None)
        if not isinstance(query_id, str) or not isinstance(doc_id, str):
            continue
        if not isinstance(relevance, int):
            continue
        qrels_by_query.setdefault(query_id, {})[doc_id] = relevance

    # CAsT query objects expose raw_utterance and (for judged) a manual rewrite;
    # topic_number / turn_number order the conversation.
    topics_by_number: dict[int, list[dict[str, object]]] = {}
    for topic in dataset.queries_iter():
        topic_number = getattr(topic, "topic_number", None)
        query_id = getattr(topic, "query_id", None)
        raw = _first_text_attr(topic, ("raw_utterance", "utterance"))
        resolved = _first_text_attr(
            topic, ("manual_rewritten_utterance", "automatic_rewritten_utterance")
        )
        if not isinstance(topic_number, int) or not isinstance(query_id, str) or raw is None:
            continue
        turn_number = getattr(topic, "turn_number", None)
        topics_by_number.setdefault(topic_number, []).append(
            {
                "query_id": query_id,
                "utterance": raw,
                "resolved": resolved,
                "turn_number": turn_number if isinstance(turn_number, int) else 0,
            }
        )

    # Pull real passage text for the judged documents from the collection store.
    judged_doc_ids = sorted({doc_id for qrels in qrels_by_query.values() for doc_id in qrels})
    passages_by_id: dict[str, Passage] = {}
    docs_store = None
    store_factory = getattr(dataset, "docs_store", None)
    if callable(store_factory):
        try:
            docs_store = store_factory()
        except Exception:
            docs_store = None
    for doc_id in judged_doc_ids:
        text: str | None = None
        if docs_store is not None:
            try:
                doc = docs_store.get(doc_id)
            except Exception:
                docs_store = None
                doc = None
            if doc is not None:
                text = _first_text_attr(doc, ("text", "body"))
        passages_by_id[doc_id] = Passage(doc_id, text or doc_id)

    conversations: list[EvalConversation] = []
    for topic_number in sorted(topics_by_number):
        raw_turns = sorted(
            topics_by_number[topic_number],
            key=lambda item: item["turn_number"] if isinstance(item["turn_number"], int) else 0,
        )
        eval_turns: list[EvalTurn] = []
        for turn in raw_turns:
            query_id = turn.get("query_id")
            utterance = turn.get("utterance")
            resolved_value = turn.get("resolved")
            if not isinstance(query_id, str) or not isinstance(utterance, str):
                continue
            qrels = qrels_by_query.get(query_id)
            if not qrels:
                continue
            eval_turns.append(
                EvalTurn(
                    turn_index=len(eval_turns),
                    query=utterance,
                    qrels=qrels,
                    resolved_query=resolved_value if isinstance(resolved_value, str) else None,
                )
            )
        if eval_turns:
            conversations.append(
                EvalConversation(
                    conversation_id=f"cast2019-{topic_number}",
                    turns=tuple(eval_turns),
                )
            )
    return tuple(conversations), tuple(passages_by_id.values())


def load_cast2019_resolved_queries(conversation: EvalConversation) -> tuple[str, ...] | None:
    resolved = tuple(
        turn.resolved_query for turn in conversation.turns if turn.resolved_query is not None
    )
    if len(resolved) != len(conversation.turns):
        return None
    return resolved


def merge_passages(passage_groups: Sequence[Sequence[Passage]]) -> tuple[Passage, ...]:
    merged: dict[str, Passage] = {}
    for group in passage_groups:
        for passage in group:
            merged[passage.passage_id] = passage
    return tuple(merged[passage_id] for passage_id in sorted(merged))
