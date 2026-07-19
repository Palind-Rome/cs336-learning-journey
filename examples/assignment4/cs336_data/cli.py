"""Reproducible command-line entry points for Assignment 4."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

from xopen import xopen

from .pipeline import PipelineConfig, PipelineFactory, process_wet_files_parallel
from .quality import train_quality_classifier
from .tokenize import tokenize_jsonl


def _expand(patterns: Iterable[str]) -> list[str]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    if not paths:
        raise FileNotFoundError(f"no files matched: {list(patterns)}")
    return paths


def _iter_documents(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        with xopen(path, "rt", encoding="utf-8") as file:
            if ".jsonl" in Path(path).name:
                for line in file:
                    record = json.loads(line)
                    yield record["text"]
            else:
                yield file.read()


def _read_domains(path: str | None) -> tuple[str, ...]:
    if path is None:
        return ()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return tuple(
        sorted(
            {
                line.strip().lower().rstrip(".")
                for line in lines
                if line.strip() and not line.lstrip().startswith("#")
            }
        )
    )


def _train_quality(args: argparse.Namespace) -> None:
    positive_paths = _expand(args.positive)
    negative_paths = _expand(args.negative)
    train_quality_classifier(
        _iter_documents(positive_paths),
        _iter_documents(negative_paths),
        args.output,
        epoch=args.epoch,
    )


def _filter(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        english_threshold=args.english_threshold,
        nsfw_threshold=args.nsfw_threshold,
        toxicity_threshold=args.toxicity_threshold,
        quality_threshold=args.quality_threshold,
        target_domains=_read_domains(args.target_domains),
    )
    factory = PipelineFactory(
        language_model_path=args.lid_model,
        nsfw_model_path=args.nsfw_model,
        toxicity_model_path=args.toxicity_model,
        quality_model_path=args.quality_model,
        config=config,
    )
    process_wet_files_parallel(
        _expand(args.input),
        args.output_directory,
        factory,
        args.workers,
        audit_per_reason=args.audit_per_reason,
        full_dump_bytes=args.full_dump_bytes,
    )


def _tokenize(args: argparse.Namespace) -> None:
    count = tokenize_jsonl(_expand(args.input), args.output, workers=args.workers)
    print(json.dumps({"output": args.output, "tokens": count}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    quality = commands.add_parser("train-quality")
    quality.add_argument("--positive", action="append", required=True)
    quality.add_argument("--negative", action="append", required=True)
    quality.add_argument("--output", required=True)
    quality.add_argument("--epoch", type=int, default=10)
    quality.set_defaults(run=_train_quality)

    filtering = commands.add_parser("filter")
    filtering.add_argument("--input", action="append", required=True)
    filtering.add_argument("--output-directory", required=True)
    filtering.add_argument("--lid-model", required=True)
    filtering.add_argument("--nsfw-model", required=True)
    filtering.add_argument("--toxicity-model", required=True)
    filtering.add_argument("--quality-model", required=True)
    filtering.add_argument("--target-domains")
    filtering.add_argument("--workers", type=int, default=8)
    filtering.add_argument("--audit-per-reason", type=int, default=5)
    filtering.add_argument("--full-dump-bytes", type=int)
    filtering.add_argument("--english-threshold", type=float, default=0.70)
    filtering.add_argument("--nsfw-threshold", type=float, default=0.80)
    filtering.add_argument("--toxicity-threshold", type=float, default=0.80)
    filtering.add_argument("--quality-threshold", type=float, default=0.55)
    filtering.set_defaults(run=_filter)

    tokenize = commands.add_parser("tokenize")
    tokenize.add_argument("--input", action="append", required=True)
    tokenize.add_argument("--output", required=True)
    tokenize.add_argument("--workers", type=int)
    tokenize.set_defaults(run=_tokenize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
