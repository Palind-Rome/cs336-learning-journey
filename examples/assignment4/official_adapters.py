"""Adapter bodies for the official tests; classifier paths come from env vars."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from cs336_data.classifiers import LanguageIdentifier, load_nsfw_classifier, load_toxicity_classifier
from cs336_data.deduplication import exact_line_deduplicate, minhash_deduplicate
from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.pii import mask_emails, mask_ipv4_addresses, mask_phone_numbers
from cs336_data.quality import FastTextQualityClassifier, passes_gopher_quality_filter


def _model_path(environment_name: str, shared_name: str) -> Path:
    path = Path(os.environ.get(environment_name, f"/shared-data/classifiers/{shared_name}"))
    if not path.exists():
        raise FileNotFoundError(f"set {environment_name}; model not found at {path}")
    return path


@lru_cache(maxsize=None)
def _language():
    return LanguageIdentifier(_model_path("CS336_LID_MODEL", "lid.176.bin"))


@lru_cache(maxsize=None)
def _nsfw():
    return load_nsfw_classifier(
        _model_path("CS336_NSFW_MODEL", "dolma_fasttext_nsfw_jigsaw_model.bin")
    )


@lru_cache(maxsize=None)
def _toxicity():
    return load_toxicity_classifier(
        _model_path("CS336_TOXICITY_MODEL", "dolma_fasttext_hatespeech_jigsaw_model.bin")
    )


@lru_cache(maxsize=None)
def _quality():
    return FastTextQualityClassifier(_model_path("CS336_QUALITY_MODEL", "quality.bin"))


def run_extract_text_from_html_bytes(html_bytes):
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text):
    return _language().identify(text)


def run_mask_emails(text):
    return mask_emails(text)


def run_mask_phone_numbers(text):
    return mask_phone_numbers(text)


def run_mask_ips(text):
    return mask_ipv4_addresses(text)


def run_classify_nsfw(text):
    return _nsfw().classify(text)


def run_classify_toxic_speech(text):
    return _toxicity().classify(text)


def run_classify_quality(text):
    return _quality().classify(text)


def run_gopher_quality_filter(text):
    return passes_gopher_quality_filter(text)


def run_exact_line_deduplication(input_files, output_directory):
    return exact_line_deduplicate(input_files, output_directory)


def run_minhash_deduplication(
    input_files,
    num_hashes,
    num_bands,
    ngrams,
    jaccard_threshold,
    output_directory,
):
    return minhash_deduplicate(
        input_files,
        num_hashes,
        num_bands,
        ngrams,
        jaccard_threshold,
        output_directory,
    )
