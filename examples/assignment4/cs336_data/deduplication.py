"""Exact-line and MinHash+LSH fuzzy deduplication."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from xopen import xopen


def _line_key(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()


def exact_line_deduplicate(
    input_files: list[str | Path], output_directory: str | Path
) -> None:
    """Two passes: count fixed-size line hashes, then retain corpus-unique lines."""

    paths = [Path(path) for path in input_files]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frequency: Counter[bytes] = Counter()
    for path in paths:
        with xopen(path, "rt", encoding="utf-8") as file:
            for line in file:
                frequency[_line_key(line)] += 1

    for path in paths:
        output_path = output_directory / path.name
        with xopen(path, "rt", encoding="utf-8") as source, xopen(
            output_path, "wt", encoding="utf-8"
        ) as target:
            for line in source:
                if frequency[_line_key(line)] == 1:
                    target.write(line)


def normalize_for_deduplication(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s]", " ", without_accents)
    return " ".join(without_punctuation.split())


def word_ngrams(text: str, n: int) -> set[str]:
    tokens = normalize_for_deduplication(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


_PRIME = np.uint64(4_294_967_311)


def _base_hash(value: str) -> np.uint64:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def minhash_signature(ngrams: set[str], num_hashes: int, seed: int = 336) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coefficients_a = rng.integers(1, _PRIME, size=num_hashes, dtype=np.uint64)
    coefficients_b = rng.integers(0, _PRIME, size=num_hashes, dtype=np.uint64)
    signature = np.full(num_hashes, _PRIME, dtype=np.uint64)
    for ngram in ngrams:
        value = _base_hash(ngram)
        permuted = (coefficients_a * value + coefficients_b) % _PRIME
        signature = np.minimum(signature, permuted)
    return signature


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _candidate_pairs(signatures: list[np.ndarray], num_bands: int) -> set[tuple[int, int]]:
    rows_per_band = len(signatures[0]) // num_bands
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for document_index, signature in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band].tobytes())
            for previous in buckets[key]:
                candidates.add((previous, document_index))
            buckets[key].append(document_index)
    return candidates


def minhash_deduplicate(
    input_files: list[str | Path],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: str | Path,
) -> None:
    if num_hashes <= 0 or num_bands <= 0 or num_hashes % num_bands:
        raise ValueError("num_hashes must be positive and divisible by num_bands")
    paths = sorted((Path(path) for path in input_files), key=lambda path: path.name)
    documents: list[str] = []
    for path in paths:
        with xopen(path, "rt", encoding="utf-8") as file:
            documents.append(file.read())

    ngram_sets = [word_ngrams(document, ngrams) for document in documents]
    signatures = [minhash_signature(values, num_hashes) for values in ngram_sets]
    union_find = UnionFind(len(paths))
    for left, right in _candidate_pairs(signatures, num_bands):
        if jaccard(ngram_sets[left], ngram_sets[right]) >= jaccard_threshold:
            union_find.union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(paths)):
        clusters[union_find.find(index)].append(index)
    # Deterministically keep the lexicographically first path. Reproducibility is
    # preferable to an unrecorded random choice, while still retaining one per cluster.
    keep = {min(indices) for indices in clusters.values()}
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for index in sorted(keep):
        with xopen(output_directory / paths[index].name, "wt", encoding="utf-8") as file:
            file.write(documents[index])
