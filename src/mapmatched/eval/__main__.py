from __future__ import annotations

import argparse
from pathlib import Path

from .ablations import run_ablation_grid
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
    parser.add_argument("--recall-k", type=int, default=100)
    parser.add_argument("--standalone-tolerance", type=float, default=0.02)
    parser.add_argument("--follow-up-min-delta", type=float, default=0.0)
    parser.add_argument("--include-resolved-oracle", action="store_true")
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
        return conversations, passages
    raise ValueError(f"unsupported benchmark: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    conversations, passages = load_benchmark(
        args.benchmark,
        conversation_limit=args.conversation_limit,
        data_path=args.data_path,
    )
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
    )
    report = run_ablation_grid(
        conversations=conversations,
        passages=passages,
        embedder=embedder,
        eval_config=eval_config,
        include_resolved_oracle=args.include_resolved_oracle or args.benchmark == "cast2019",
    )
    args.output.write_text(render_json(report), encoding="utf-8")
    markdown = render_markdown_table(report)
    if args.markdown_output is not None:
        args.markdown_output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
