from __future__ import annotations

from collections.abc import Sequence

from mapmatched import FAISSProvider, KNNGraph, MapMatchedRetriever


def main() -> None:
    try:
        import faiss
        import numpy as np
    except ImportError:
        print("Skipped: install the 'faiss' extra to run this example")
        return

    chunk_ids = ("overview", "details", "failure-modes", "unrelated")
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.7, 0.3],
            [0.0, 1.0],
        ],
        dtype="float32",
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(2)
    index.add(embeddings)
    query_embeddings = {
        "How does retrieval work?": (1.0, 0.0),
        "What happens when it fails?": (0.6, 0.4),
    }

    def embed_query(query: str) -> Sequence[float]:
        return query_embeddings[query]

    graph = KNNGraph.from_embeddings(
        chunk_ids,
        embeddings.tolist(),
        neighbor_count=2,
        maximum_distance=2.0,
    )
    provider = FAISSProvider(index, chunk_ids, embed_query)
    session = MapMatchedRetriever(
        graph,
        provider=provider,
        transition_weight=0.5,
        context_radius=0.25,
    ).session()

    session.retrieve("How does retrieval work?")
    result = session.retrieve("What happens when it fails?")

    print(f"MAP chunk: {result.chunk_id}")
    print(f"context: {', '.join(result.context_chunk_ids)}")
    print(result.trace.render())


if __name__ == "__main__":
    main()
