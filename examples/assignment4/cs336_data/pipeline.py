"""Parallel, auditable WET-to-JSONL filtering pipeline."""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

from fastwarc.warc import ArchiveIterator, WarcRecordType

from .classifiers import LanguageIdentifier, load_nsfw_classifier, load_toxicity_classifier
from .pii import mask_emails, mask_ipv4_addresses, mask_phone_numbers
from .quality import FastTextQualityClassifier, passes_gopher_quality_filter


TextScorer = Callable[[str], tuple[str, float]]


@dataclass
class PipelineConfig:
    english_threshold: float = 0.70
    nsfw_threshold: float = 0.80
    toxicity_threshold: float = 0.80
    quality_threshold: float = 0.55
    target_domains: tuple[str, ...] = ()


@dataclass
class FileResult:
    input_path: str
    output_path: str
    counters: dict[str, int]
    elapsed_seconds: float
    input_bytes: int
    audit_samples: list[dict[str, str]]


@dataclass
class PipelineFactory:
    """Picklable factory: every process loads its four models exactly once."""

    language_model_path: str
    nsfw_model_path: str
    toxicity_model_path: str
    quality_model_path: str
    config: PipelineConfig

    def __call__(self) -> "DocumentPipeline":
        language = LanguageIdentifier(self.language_model_path)
        nsfw = load_nsfw_classifier(self.nsfw_model_path)
        toxicity = load_toxicity_classifier(self.toxicity_model_path)
        quality = FastTextQualityClassifier(self.quality_model_path)
        return DocumentPipeline(
            language_identifier=language.identify,
            nsfw_classifier=nsfw.classify,
            toxicity_classifier=toxicity.classify,
            quality_classifier=quality.classify,
            config=self.config,
        )


def _host_matches(url: str, domains: tuple[str, ...]) -> bool:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in domains)


class DocumentPipeline:
    def __init__(
        self,
        language_identifier: TextScorer,
        nsfw_classifier: TextScorer,
        toxicity_classifier: TextScorer,
        quality_classifier: TextScorer,
        config: PipelineConfig | None = None,
    ):
        self.language_identifier = language_identifier
        self.nsfw_classifier = nsfw_classifier
        self.toxicity_classifier = toxicity_classifier
        self.quality_classifier = quality_classifier
        self.config = config or PipelineConfig()

    def process(self, text: str, url: str = "") -> tuple[str | None, str]:
        if self.config.target_domains and not _host_matches(url, self.config.target_domains):
            return None, "domain"
        language, language_score = self.language_identifier(text)
        if language != "en" or language_score < self.config.english_threshold:
            return None, "language"
        if not passes_gopher_quality_filter(text):
            return None, "gopher"
        nsfw_label, nsfw_score = self.nsfw_classifier(text)
        if nsfw_label == "nsfw" and nsfw_score >= self.config.nsfw_threshold:
            return None, "nsfw"
        toxic_label, toxic_score = self.toxicity_classifier(text)
        if toxic_label == "toxic" and toxic_score >= self.config.toxicity_threshold:
            return None, "toxic"
        quality_label, quality_score = self.quality_classifier(text)
        if quality_label != "wiki" or quality_score < self.config.quality_threshold:
            return None, "learned_quality"

        text, email_count = mask_emails(text)
        text, phone_count = mask_phone_numbers(text)
        text, ip_count = mask_ipv4_addresses(text)
        reason = f"kept:email={email_count},phone={phone_count},ip={ip_count}"
        return text, reason


def _audit_key(sample: dict[str, str]) -> str:
    payload = "\0".join((sample["decision"], sample["url"], sample["snippet"]))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _keep_smallest_samples(
    samples: list[dict[str, str]], per_reason: int
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for sample in samples:
        grouped.setdefault(sample["decision"], []).append(sample)
    kept: list[dict[str, str]] = []
    for reason in sorted(grouped):
        kept.extend(sorted(grouped[reason], key=_audit_key)[:per_reason])
    return kept


def process_wet_file(
    input_path: str | Path,
    output_path: str | Path,
    pipeline_factory: Callable[[], DocumentPipeline],
    audit_per_reason: int = 5,
) -> FileResult:
    """Process one compressed WET file; create classifiers once per worker."""

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    started = time.perf_counter()
    samples: list[dict[str, str]] = []
    pipeline = pipeline_factory()
    with gzip.open(input_path, "rb") as source, output_path.open("w", encoding="utf-8") as target:
        for record in ArchiveIterator(source, record_types=WarcRecordType.conversion):
            counters["seen"] += 1
            text = record.reader.read().decode("utf-8", errors="replace")
            url = record.headers.get("WARC-Target-URI", "")
            filtered, decision = pipeline.process(text, url)
            reason = decision.split(":", 1)[0]
            counters[reason] += 1
            if reason == "kept":
                for item in decision.split(":", 1)[1].split(","):
                    name, count = item.split("=", 1)
                    counters[f"masked_{name}"] += int(count)
            samples.append(
                {
                    "decision": reason,
                    "url": url,
                    "snippet": text[:500].replace("\x00", "�"),
                }
            )
            samples = _keep_smallest_samples(samples, audit_per_reason)
            if filtered is None:
                continue
            target.write(
                json.dumps(
                    {
                        "url": url,
                        "text": filtered,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return FileResult(
        str(input_path),
        str(output_path),
        dict(counters),
        time.perf_counter() - started,
        input_path.stat().st_size,
        samples,
    )


def process_wet_files_parallel(
    input_paths: Iterable[str | Path],
    output_directory: str | Path,
    pipeline_factory: Callable[[], DocumentPipeline],
    workers: int,
    *,
    audit_per_reason: int = 5,
    full_dump_bytes: int | None = None,
) -> Counter[str]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    started = time.perf_counter()
    input_bytes = 0
    worker_seconds = 0.0
    audit_samples: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_wet_file,
                path,
                output_directory / (Path(path).name + ".jsonl"),
                pipeline_factory,
                audit_per_reason,
            )
            for path in input_paths
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            totals.update(result.counters)
            input_bytes += result.input_bytes
            worker_seconds += result.elapsed_seconds
            audit_samples.extend(result.audit_samples)
            audit_samples = _keep_smallest_samples(audit_samples, audit_per_reason)
    wall_seconds = time.perf_counter() - started
    bytes_per_second = input_bytes / wall_seconds if wall_seconds else 0.0
    report = {
        "counts": dict(totals),
        "wall_seconds": wall_seconds,
        "worker_seconds": worker_seconds,
        "input_bytes": input_bytes,
        "bytes_per_second": bytes_per_second,
        "projected_full_dump_seconds": (
            full_dump_bytes / bytes_per_second
            if full_dump_bytes is not None and bytes_per_second
            else None
        ),
    }
    (output_directory / "filter-stats.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_directory / "audit-samples.jsonl").open("w", encoding="utf-8") as file:
        for sample in _keep_smallest_samples(audit_samples, audit_per_reason):
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return totals
