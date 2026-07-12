from mapmatched.eval.baselines import MapMatchedMethodConfig
from mapmatched.eval.baselines.methods import (
    run_history_concat_conversation,
    run_maximal_marginal_relevance_conversation,
    run_pointwise_conversation,
)
from mapmatched.eval.corpus import BruteForceProvider, build_knn_graph, build_passage_embeddings
from mapmatched.eval.embedder import DeterministicHashEmbedder
from mapmatched.eval.loaders.synthetic import load_synthetic_fixture


def test_history_concat_changes_query_sequence() -> None:
    _, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    passage_ids, passage_embeddings = build_passage_embeddings(passages, embedder)
    provider = BruteForceProvider(passage_ids, passage_embeddings, embedder)
    graph = build_knn_graph(passage_ids, passage_embeddings)
    queries = ("alpha overview introduction", "details transition graph")
    pointwise_ids, _ = run_pointwise_conversation(
        graph=graph,
        provider=provider,
        queries=queries,
        config=MapMatchedMethodConfig(candidate_limit=4, score_normalization="none"),
    )
    history_ids, _ = run_history_concat_conversation(
        provider=provider,
        queries=queries,
        candidate_limit=4,
    )
    assert pointwise_ids[0] == history_ids[0]
    assert len(history_ids) == 2


def test_maximal_marginal_relevance_returns_one_chunk_per_turn() -> None:
    _, passages = load_synthetic_fixture()
    embedder = DeterministicHashEmbedder()
    passage_ids, passage_embeddings = build_passage_embeddings(passages, embedder)
    provider = BruteForceProvider(passage_ids, passage_embeddings, embedder)
    ranked_ids, _ = run_maximal_marginal_relevance_conversation(
        provider=provider,
        queries=("alpha overview introduction", "details transition graph"),
        candidate_limit=4,
    )
    assert len(ranked_ids) == 2
