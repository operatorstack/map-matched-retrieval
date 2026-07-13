from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from ..types import EvalConversation, EvalTurn, Passage


class EvalDependencyUnavailableError(ImportError):
    pass


_PATH_ENV_VAR = "MAPMATCHED_TOPIOCQA_PATH"


def _resolve_data_path(data_path: str | os.PathLike[str] | None) -> Path:
    """Locate the TopiOCQA JSON/JSONL split.

    HuggingFace `datasets` dropped support for the custom dataset *script* that
    `McGill-NLP/TopiOCQA` ships, so we read the released JSON directly instead.
    Point this at the downloaded validation split (``topiocqa_valid.jsonl``) via the
    ``data_path`` argument or the ``MAPMATCHED_TOPIOCQA_PATH`` environment
    variable.
    """
    candidate = data_path if data_path is not None else os.environ.get(_PATH_ENV_VAR)
    if not candidate:
        raise EvalDependencyUnavailableError(
            "TopiOCQA loader needs the dataset JSONL. Download data/topiocqa_valid.jsonl "
            "from https://huggingface.co/datasets/McGill-NLP/TopiOCQA and "
            f"pass data_path=... or set {_PATH_ENV_VAR}."
        )
    path = Path(candidate)
    if not path.is_file():
        raise EvalDependencyUnavailableError(f"TopiOCQA data file not found: {path}")
    return path


def _iter_rows(path: Path) -> Iterator[dict[str, object]]:
    """Yield row dicts from either a JSON array file or a JSON-lines file."""
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise EvalDependencyUnavailableError("TopiOCQA JSON must be a list of rows")
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if isinstance(row, dict):
            yield row


def load_topiocqa_micro(
    *,
    conversation_limit: int = 50,
    data_path: str | os.PathLike[str] | None = None,
) -> tuple[tuple[EvalConversation, ...], tuple[Passage, ...]]:
    if conversation_limit <= 0:
        raise ValueError("conversation_limit must be greater than zero")
    path = _resolve_data_path(data_path)

    # Group rows by conversation, keeping the first `conversation_limit`
    # conversations in file order (rows of an already-selected conversation are
    # always collected, even when interleaved).
    grouped: dict[int, list[dict[str, object]]] = {}
    selected_order: list[int] = []
    for row in _iter_rows(path):
        conversation_number = row.get("Conversation_no")
        if not isinstance(conversation_number, int):
            continue
        if conversation_number not in grouped:
            if len(selected_order) >= conversation_limit:
                continue
            selected_order.append(conversation_number)
        grouped.setdefault(conversation_number, []).append(row)

    conversations: list[EvalConversation] = []
    passages_by_id: dict[str, Passage] = {}
    for conversation_number in selected_order:
        rows = sorted(grouped[conversation_number], key=_turn_number)
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
                    group_key = title if isinstance(title, str) and title else None
                    passages_by_id[passage_id] = Passage(
                        passage_id, passage_text, group_key=group_key
                    )
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
