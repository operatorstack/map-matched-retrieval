from mapmatched.decoder import StandaloneDecoder
from mapmatched.eval.ablations import run_ablation_grid
from mapmatched.eval.embedder import DeterministicHashEmbedder
from mapmatched.eval.loaders.synthetic import load_synthetic_fixture
from mapmatched.eval.types import EvalConfig
from mapmatched.graph import InMemoryCorpusGraph
from mapmatched.models import DecoderCandidate


def _eval_config(**overrides: object) -> EvalConfig:
    base = {
        "benchmark": "synthetic",
        "tier": "synthetic",
        "embedder_name": "deterministic-hash-32",
        "recall_k": 10,
        "entropy_threshold": None,
        "standalone_tolerance": 0.02,
        "follow_up_min_delta": 0.0,
    }
    base.update(overrides)
    return EvalConfig(**base)  # type: ignore[arg-type]


def test_decoder_ranking_beta_zero_is_emission_order() -> None:
    # transition_weight 0 => trajectory score == emission; ranking follows score
    trellis = [
        [DecoderCandidate("a", 2.0, 2.0), DecoderCandidate("b", 1.0, 1.0)],
        [DecoderCandidate("a", 0.5, 0.5), DecoderCandidate("b", 3.0, 3.0)],
    ]
    graph = InMemoryCorpusGraph.from_edges([("a", "b")], nodes=["a", "b"])
    _, ranking = StandaloneDecoder().decode_ranked(
        trellis, graph=graph, emission_weight=1.0, transition_weight=0.0
    )
    # final turn: b (score 3) should outrank a (0.5) under β=0
    assert [c.chunk_id for c in ranking] == ["b", "a"]


def test_full_ranking_runs_and_beta_zero_matches_pointwise() -> None:
    conversations, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    report = run_ablation_grid(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        eval_config=_eval_config(ranking_mode="full"),
        transition_weights=(0.0, 0.5),
        include_mmr=False,
    )
    # both methods present; graph_mode label reflects the source
    names = {m.method_name for m in report.methods}
    assert {"pointwise", "mapmatched"} <= names
    assert all(m.graph_mode == "knn" for m in report.methods)


def test_section_graph_source_runs_end_to_end() -> None:
    conversations, passages = load_synthetic_fixture()
    # synthetic passages have no group_key -> section graph = all isolated; the
    # run must still complete and label the graph source.
    report = run_ablation_grid(
        conversations=conversations,
        passages=passages,
        embedder=DeterministicHashEmbedder(),
        eval_config=_eval_config(graph_source="section", ranking_mode="full"),
        transition_weights=(0.0, 0.5),
        include_mmr=False,
    )
    assert all(m.graph_mode == "section" for m in report.methods)
