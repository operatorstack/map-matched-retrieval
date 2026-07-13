import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from mapmatched.eval import DeterministicHashEmbedder, EvalConfig, MethodSpec, run_eval
from mapmatched.eval.__main__ import main
from mapmatched.eval.baselines import (
    GeminiQueryRewriter,
    GeminiRewriteConfig,
    MapMatchedMethodConfig,
    create_gemini_query_rewriter,
)
from mapmatched.eval.loaders.synthetic import load_synthetic_fixture


class RecordingQueryRewriter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def rewrite(self, *, history: Sequence[str], query: str) -> str:
        self.calls.append((tuple(history), query))
        return " ".join([*history, query])


def test_gemini_query_rewriter_sends_history_and_returns_text() -> None:
    requests: list[dict[str, object]] = []

    class Response:
        text = "  standalone query  "

    def generate_content(**request: object) -> object:
        requests.append(request)
        return Response()

    rewriter = GeminiQueryRewriter(
        generate_content=generate_content,
        config=GeminiRewriteConfig(model="test-model"),
    )
    rewritten = rewriter.rewrite(history=("first question",), query="what about it?")

    assert rewritten == "standalone query"
    assert requests[0]["model"] == "test-model"
    assert requests[0]["config"] == {
        "thinking_config": {"thinking_level": "minimal"}
    }
    assert '"prior_user_utterances": ["first question"]' in str(requests[0]["contents"])


def test_create_gemini_query_rewriter_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_gemini_query_rewriter()


def test_gemini_rewrite_runs_end_to_end_with_paired_comparison() -> None:
    conversations, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    rewriter = RecordingQueryRewriter()
    report = run_eval(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        methods=(
            MethodSpec(name="pointwise", transition_weight=0.0),
            MethodSpec(name="gemini_rewrite"),
        ),
        config=MapMatchedMethodConfig(candidate_limit=4),
        eval_config=EvalConfig(
            benchmark="synthetic",
            tier="synthetic",
            embedder_name=embedder.name,
            recall_k=10,
            entropy_threshold=None,
            standalone_tolerance=0.02,
            follow_up_min_delta=0.0,
            bootstrap_samples=20,
            bootstrap_seed=42,
        ),
        query_rewriter=rewriter,
    )

    expected_turn_count = sum(len(conversation.turns) for conversation in conversations)
    assert len(rewriter.calls) == expected_turn_count
    assert rewriter.calls[0][0] == ()
    assert rewriter.calls[1][0] == (conversations[0].turns[0].query,)
    comparison = next(
        comparison
        for comparison in report.comparisons
        if comparison.method_name == "gemini_rewrite"
    )
    assert all(slice_metrics.ndcg_at_3_delta_ci is not None for slice_metrics in comparison.slices)


def test_cli_records_gemini_rewrite_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rewriter = RecordingQueryRewriter()
    monkeypatch.setattr(
        "mapmatched.eval.__main__.create_gemini_query_rewriter",
        lambda *, model: rewriter,
    )
    output_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--benchmark",
            "synthetic",
            "--tier",
            "synthetic",
            "--include-gemini-rewrite",
            "--gemini-model",
            "test-model",
            "--candidate-limit",
            "4",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["config"]["query_rewrite_provider"] == "gemini"
    assert payload["config"]["query_rewrite_model"] == "test-model"
    assert payload["config"]["query_rewrite_prompt_version"] == "cast-standalone-v1"
    assert any(method["method_name"] == "gemini_rewrite" for method in payload["methods"])
