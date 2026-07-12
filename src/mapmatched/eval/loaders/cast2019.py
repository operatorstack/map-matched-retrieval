from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType

from ..types import EvalConversation, EvalTurn, Passage


class EvalDependencyUnavailableError(ImportError):
    pass


def _load_ir_datasets() -> ModuleType:
    try:
        return importlib.import_module("ir_datasets")
    except ImportError as error:
        raise EvalDependencyUnavailableError(
            "CAsT loader requires the 'eval' extra: pip install 'map-matched-retrieval[eval]'"
        ) from error


def load_cast2019_micro() -> tuple[tuple[EvalConversation, ...], tuple[Passage, ...]]:
    ir_datasets = _load_ir_datasets()
    load = getattr(ir_datasets, "load", None)
    if not callable(load):
        raise EvalDependencyUnavailableError("ir_datasets.load is unavailable")
    dataset = load("trec-cast-2019/train")
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

    topics_by_number: dict[int, dict[str, object]] = {}
    for topic in dataset.queries_iter():
        topic_number = getattr(topic, "topic_number", None)
        query_id = getattr(topic, "query_id", None)
        utterance = getattr(topic, "utterance", None)
        if not isinstance(topic_number, int) or not isinstance(query_id, str):
            continue
        if not isinstance(utterance, str) or not utterance:
            continue
        topics_by_number.setdefault(topic_number, {"turns": []})
        turns = topics_by_number[topic_number]["turns"]
        if isinstance(turns, list):
            turns.append({"query_id": query_id, "utterance": utterance})

    passages_by_id: dict[str, Passage] = {}
    for doc_id in sorted({doc_id for qrels in qrels_by_query.values() for doc_id in qrels}):
        passages_by_id[doc_id] = Passage(doc_id, doc_id)

    conversations: list[EvalConversation] = []
    for topic_number in sorted(topics_by_number):
        topic = topics_by_number[topic_number]
        turns_value = topic.get("turns")
        if not isinstance(turns_value, list):
            continue
        eval_turns: list[EvalTurn] = []
        for turn_index, turn in enumerate(turns_value):
            if not isinstance(turn, dict):
                continue
            query_id = turn.get("query_id")
            utterance = turn.get("utterance")
            if not isinstance(query_id, str) or not isinstance(utterance, str):
                continue
            qrels = qrels_by_query.get(query_id)
            if not qrels:
                continue
            eval_turns.append(
                EvalTurn(
                    turn_index=turn_index,
                    query=utterance,
                    qrels=qrels,
                )
            )
        if eval_turns:
            conversations.append(
                EvalConversation(
                    conversation_id=f"cast2019-{topic_number}",
                    turns=tuple(
                        EvalTurn(
                            turn_index=index,
                            query=turn.query,
                            qrels=turn.qrels,
                            resolved_query=turn.resolved_query,
                        )
                        for index, turn in enumerate(eval_turns)
                    ),
                )
            )
    return tuple(conversations), tuple(passages_by_id.values())


def load_cast2019_resolved_queries(conversation: EvalConversation) -> tuple[str, ...] | None:
    del conversation
    return None


def merge_passages(passage_groups: Sequence[Sequence[Passage]]) -> tuple[Passage, ...]:
    merged: dict[str, Passage] = {}
    for group in passage_groups:
        for passage in group:
            merged[passage.passage_id] = passage
    return tuple(merged[passage_id] for passage_id in sorted(merged))
