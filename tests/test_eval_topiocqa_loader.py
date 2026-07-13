import hashlib
import json
from pathlib import Path

from mapmatched.eval import DeterministicHashEmbedder, EvalConfig, MethodSpec, run_eval
from mapmatched.eval.__main__ import main
from mapmatched.eval.baselines import MapMatchedMethodConfig
from mapmatched.eval.loaders import load_topiocqa_micro

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "topiocqa_micro_sample.jsonl"


def test_topiocqa_loader_limits_conversations_in_file_order() -> None:
    conversations, passages = load_topiocqa_micro(
        conversation_limit=2,
        data_path=FIXTURE_PATH,
    )
    assert tuple(conversation.conversation_id for conversation in conversations) == (
        "topiocqa-1",
        "topiocqa-2",
    )
    assert tuple(len(conversation.turns) for conversation in conversations) == (2, 1)
    assert {passage.passage_id for passage in passages} == {
        "orbit-1",
        "topiocqa-extra-1-0-1",
        "orbit-2",
        "plant-1",
    }
    assert conversations[0].turns[0].qrels == {
        "orbit-1": 3,
        "topiocqa-extra-1-0-1": 2,
    }


def test_topiocqa_fixture_runs_end_to_end_offline() -> None:
    conversations, passages = load_topiocqa_micro(
        conversation_limit=2,
        data_path=FIXTURE_PATH,
    )
    embedder = DeterministicHashEmbedder()
    report = run_eval(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        methods=(
            MethodSpec(name="pointwise", transition_weight=0.0),
            MethodSpec(name="mapmatched", transition_weight=0.5),
        ),
        config=MapMatchedMethodConfig(candidate_limit=4),
        eval_config=EvalConfig(
            benchmark="topiocqa",
            tier="micro",
            embedder_name=embedder.name,
            recall_k=4,
            entropy_threshold=None,
            standalone_tolerance=0.02,
            follow_up_min_delta=0.0,
            bootstrap_samples=20,
            bootstrap_seed=42,
            conversation_ids=tuple(conversation.conversation_id for conversation in conversations),
        ),
    )
    assert report.comparisons
    assert report.config.conversation_ids == ("topiocqa-1", "topiocqa-2")
    assert all(turn.conversation_id for method in report.methods for turn in method.turns)


def test_topiocqa_cli_records_reproduction_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--profile",
            "fixture-profile",
            "--benchmark",
            "topiocqa",
            "--data-path",
            str(FIXTURE_PATH),
            "--conversation-limit",
            "2",
            "--knn-neighbors",
            "2",
            "--candidate-limit",
            "4",
            "--bootstrap-samples",
            "10",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    expected_sha256 = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert exit_code == 0
    assert payload["config"]["profile"] == "fixture-profile"
    assert payload["config"]["dataset_filename"] == FIXTURE_PATH.name
    assert payload["config"]["dataset_sha256"] == expected_sha256
    assert payload["config"]["conversation_ids"] == ["topiocqa-1", "topiocqa-2"]
    assert payload["config"]["knn_neighbor_count"] == 2
    assert payload["config"]["package_version"] == "0.1.0"
    assert payload["config"]["git_revision"]
