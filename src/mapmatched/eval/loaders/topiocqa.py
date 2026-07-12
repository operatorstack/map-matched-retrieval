from __future__ import annotations

import importlib
from types import ModuleType

from ..types import EvalConversation, EvalTurn, Passage


class EvalDependencyUnavailableError(ImportError):
    pass


def _load_datasets() -> ModuleType:
    try:
        return importlib.import_module("datasets")
    except ImportError as error:
        raise EvalDependencyUnavailableError(
            "TopiOCQA loader requires the 'eval' extra: pip install 'map-matched-retrieval[eval]'"
        ) from error


def load_topiocqa_micro(
    *,
    conversation_limit: int = 50,
) -> tuple[tuple[EvalConversation, ...], tuple[Passage, ...]]:
    if conversation_limit <= 0:
        raise ValueError("conversation_limit must be greater than zero")
    datasets = _load_datasets()
    load_dataset = getattr(datasets, "load_dataset", None)
    if not callable(load_dataset):
        raise EvalDependencyUnavailableError("datasets.load_dataset is unavailable")
    dataset = load_dataset("McGill-NLP/TopiOCQA", split="validation")
    conversations: list[EvalConversation] = []
    passages_by_id: dict[str, Passage] = {}
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in dataset:
        if not isinstance(row, dict):
            continue
        conversation_number = row.get("Conversation_no")
        if not isinstance(conversation_number, int):
            continue
        grouped.setdefault(conversation_number, []).append(row)
        if len(grouped) >= conversation_limit and conversation_number not in grouped:
            break

    selected_numbers = sorted(grouped.keys())[:conversation_limit]
    for conversation_number in selected_numbers:
        rows = sorted(
            grouped[conversation_number],
            key=lambda item: _turn_number(item),
        )
        turns: list[EvalTurn] = []
        for turn_index, row in enumerate(rows):
            question = row.get("Question")
            if not isinstance(question, str) or not question:
                continue
            qrels: dict[str, int] = {}
            gold_passage = row.get("Gold_passage")
            if isinstance(gold_passage, dict):
                passage_id = gold_passage.get("id")
                title = gold_passage.get("title")
                text = gold_passage.get("text")
                if isinstance(passage_id, str) and passage_id:
                    passage_text = _join_title_text(title, text)
                    passages_by_id[passage_id] = Passage(passage_id, passage_text)
                    qrels[passage_id] = 3
            additional_answers = row.get("Additional_answers")
            if isinstance(additional_answers, list):
                for answer in additional_answers:
                    if not isinstance(answer, dict):
                        continue
                    answer_text = answer.get("Answer")
                    topic = answer.get("Topic")
                    if isinstance(answer_text, str) and answer_text:
                        synthetic_id = (
                            f"topiocqa-extra-{conversation_number}-{turn_index}-{len(qrels)}"
                        )
                        passages_by_id[synthetic_id] = Passage(
                            synthetic_id,
                            _join_title_text(
                                topic if isinstance(topic, str) else None,
                                answer_text,
                            ),
                        )
                        qrels[synthetic_id] = 2
            if not qrels:
                continue
            turns.append(
                EvalTurn(
                    turn_index=turn_index,
                    query=question,
                    qrels=qrels,
                )
            )
        if turns:
            conversations.append(
                EvalConversation(
                    conversation_id=f"topiocqa-{conversation_number}",
                    turns=tuple(
                        EvalTurn(
                            turn_index=index,
                            query=turn.query,
                            qrels=turn.qrels,
                            resolved_query=turn.resolved_query,
                        )
                        for index, turn in enumerate(turns)
                    ),
                )
            )
    return tuple(conversations), tuple(passages_by_id.values())


def _turn_number(row: dict[str, object]) -> int:
    turn_number = row.get("Turn_no")
    if isinstance(turn_number, int):
        return turn_number
    return 0


def _join_title_text(title: object, text: object) -> str:
    title_text = title if isinstance(title, str) else ""
    body_text = text if isinstance(text, str) else ""
    if title_text and body_text:
        return f"{title_text}. {body_text}"
    return title_text or body_text or "empty passage"
