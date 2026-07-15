from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

from mapmatched import __version__

from .ablations import run_ablation_grid
from .baselines import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_REWRITE_PROMPT_VERSION,
    ConversationQueryRewriter,
    create_gemini_query_rewriter,
    rewrite_conversation_queries,
)
from .embedder import DeterministicHashEmbedder, SentenceTransformerEmbedder
from .loaders import load_cast2019_micro, load_synthetic_fixture, load_topiocqa_micro
from .report import render_json, render_markdown_table
from .types import EvalConfig, EvalConversation, Passage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run map-matched retrieval evaluation.")
    parser.add_argument(
        "--benchmark",
        choices=("synthetic", "topiocqa", "cast2019"),
        default="synthetic",
    )
    parser.add_argument("--tier", choices=("micro", "synthetic"), default="micro")
    parser.add_argument("--conversation-limit", type=int, default=50)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to a local dataset file (e.g. TopiOCQA dev JSON).",
    )
    parser.add_argument("--output", type=Path, default=Path("eval-report.json"))
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument(
        "--embedder",
        choices=("hash", "sentence-transformers"),
        default="hash",
        help="hash = deterministic CI fixture; sentence-transformers = real semantic embeddings.",
    )
    parser.add_argument(
        "--st-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name when --embedder sentence-transformers.",
    )
    parser.add_argument(
        "--graph-source",
        choices=("knn", "section"),
        default="knn",
        help="knn = embedding fallback graph; section = structured group_key graph.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=10,
        help="Neighbors per passage when --graph-source knn.",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("full", "rank1"),
        default="full",
        help="full = re-rank the candidate window by trajectory score; rank1 = legacy hoist.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=100,
        help="Per-turn candidate window fed to the decoder / re-rank.",
    )
    parser.add_argument("--recall-k", type=int, default=100)
    parser.add_argument("--standalone-tolerance", type=float, default=0.02)
    parser.add_argument("--follow-up-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Conversation-level bootstrap resamples for nDCG@3 95%% CIs (0 = disabled).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Stable run profile name stored in report metadata.",
    )
    parser.add_argument("--include-resolved-oracle", action="store_true")
    parser.add_argument(
        "--include-gemini-rewrite",
        action="store_true",
        help="Evaluate a Gemini conversational query rewrite baseline.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model used by --include-gemini-rewrite.",
    )
    parser.add_argument(
        "--gemini-rewrite-cache",
        type=Path,
        default=None,
        help="JSON checkpoint for completed Gemini rewrites.",
    )
    parser.add_argument(
        "--gemini-min-request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between Gemini requests.",
    )
    parser.add_argument(
        "--gemini-prefetch-only",
        action="store_true",
        help="Checkpoint Gemini rewrites without running retrieval evaluation.",
    )
    return parser


def load_benchmark(
    name: str,
    *,
    conversation_limit: int,
    data_path: Path | None = None,
) -> tuple[tuple[EvalConversation, ...], tuple[Passage, ...]]:
    if name == "synthetic":
        return load_synthetic_fixture()
    if name == "topiocqa":
        return load_topiocqa_micro(
            conversation_limit=conversation_limit,
            data_path=data_path,
        )
    if name == "cast2019":
        conversations, passages = load_cast2019_micro()
        return conversations[:conversation_limit], passages
    raise ValueError(f"unsupported benchmark: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    query_rewriter: ConversationQueryRewriter | None = None
    if args.include_gemini_rewrite:
        query_rewriter = create_gemini_query_rewriter(
            model=args.gemini_model,
            cache_path=args.gemini_rewrite_cache,
            minimum_request_interval=args.gemini_min_request_interval,
        )
    data_path = _effective_data_path(args.benchmark, args.data_path)
    conversations, passages = load_benchmark(
        args.benchmark,
        conversation_limit=args.conversation_limit,
        data_path=data_path,
    )
    if args.gemini_prefetch_only:
        if query_rewriter is None:
            parser.error("--gemini-prefetch-only requires --include-gemini-rewrite")
        for conversation in conversations:
            rewrite_conversation_queries(conversation, query_rewriter)
        return 0
    embedder: DeterministicHashEmbedder | SentenceTransformerEmbedder
    if args.embedder == "sentence-transformers":
        embedder = SentenceTransformerEmbedder(args.st_model)
    else:
        embedder = DeterministicHashEmbedder()
    eval_config = EvalConfig(
        benchmark=args.benchmark,
        tier=args.tier,
        embedder_name=embedder.name,
        recall_k=args.recall_k,
        entropy_threshold=None,
        standalone_tolerance=args.standalone_tolerance,
        follow_up_min_delta=args.follow_up_min_delta,
        ranking_mode=args.ranking_mode,
        graph_source=args.graph_source,
        knn_neighbor_count=args.knn_neighbors,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        profile=args.profile,
        dataset_filename=None if data_path is None else data_path.name,
        dataset_sha256=None if data_path is None else _sha256(data_path),
        conversation_ids=tuple(conversation.conversation_id for conversation in conversations),
        embedding_model=args.st_model
        if args.embedder == "sentence-transformers"
        else embedder.name,
        package_version=__version__,
        git_revision=_git_revision(),
        query_rewrite_provider="gemini" if args.include_gemini_rewrite else None,
        query_rewrite_model=args.gemini_model if args.include_gemini_rewrite else None,
        query_rewrite_prompt_version=GEMINI_REWRITE_PROMPT_VERSION
        if args.include_gemini_rewrite
        else None,
        query_rewrite_cache_filename=args.gemini_rewrite_cache.name
        if args.gemini_rewrite_cache is not None
        else None,
    )
    report = run_ablation_grid(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        eval_config=eval_config,
        candidate_limit=args.candidate_limit,
        include_gemini_rewrite=args.include_gemini_rewrite,
        include_resolved_oracle=args.include_resolved_oracle or args.benchmark == "cast2019",
        query_rewriter=query_rewriter,
    )
    args.output.write_text(render_json(report), encoding="utf-8")
    markdown = render_markdown_table(report)
    if args.markdown_output is not None:
        args.markdown_output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


def _effective_data_path(benchmark: str, data_path: Path | None) -> Path | None:
    if data_path is not None:
        return data_path
    if benchmark != "topiocqa":
        return None
    environment_path = os.environ.get("MAPMATCHED_TOPIOCQA_PATH")
    if environment_path is None:
        return None
    return Path(environment_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for block in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    revision = result.stdout.strip()
    return revision or None


if __name__ == "__main__":
    raise SystemExit(main())
