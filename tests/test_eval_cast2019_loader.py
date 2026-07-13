from types import SimpleNamespace

import pytest

from mapmatched.eval.loaders import cast2019


def test_cast_loader_stops_using_docstore_after_retrieval_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingDocStore:
        def __init__(self) -> None:
            self.get_count = 0

        def get(self, doc_id: str) -> object:
            del doc_id
            self.get_count += 1
            raise ImportError("missing corpus dependency")

    docstore = FailingDocStore()
    dataset = SimpleNamespace(
        qrels_iter=lambda: iter(
            (
                SimpleNamespace(query_id="1_1", doc_id="doc-1", relevance=2),
                SimpleNamespace(query_id="1_1", doc_id="doc-2", relevance=1),
            )
        ),
        queries_iter=lambda: iter(
            (
                SimpleNamespace(
                    topic_number=1,
                    turn_number=1,
                    query_id="1_1",
                    raw_utterance="What is renewable energy?",
                    manual_rewritten_utterance="renewable energy definition",
                ),
            )
        ),
        docs_store=lambda: docstore,
    )
    ir_datasets = SimpleNamespace(load=lambda dataset_id: dataset)
    monkeypatch.setattr(cast2019, "_load_ir_datasets", lambda: ir_datasets)

    conversations, passages = cast2019.load_cast2019_micro()

    assert docstore.get_count == 1
    assert conversations[0].turns[0].resolved_query == "renewable energy definition"
    assert tuple((passage.passage_id, passage.text) for passage in passages) == (
        ("doc-1", "doc-1"),
        ("doc-2", "doc-2"),
    )
