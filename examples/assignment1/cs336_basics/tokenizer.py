"""Byte-level BPE training plus a streaming encoder/decoder."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _merge_sequence(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge every non-overlapping occurrence of ``pair`` in one pre-token."""

    merged: list[bytes] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            merged.append(tokens[index] + tokens[index + 1])
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return tuple(merged)


def _pair_occurrences(tokens: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    return Counter(zip(tokens, tokens[1:]))


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE with deterministic tie-breaking and incremental pair counts."""

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size cannot hold the 256 bytes and all special tokens")

    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    for special in special_tokens:
        encoded = special.encode("utf-8")
        if encoded not in vocab.values():
            vocab[len(vocab)] = encoded

    text = Path(input_path).read_text(encoding="utf-8")
    if special_tokens:
        alternatives = sorted((regex.escape(token) for token in special_tokens), key=len, reverse=True)
        ordinary_segments = regex.split("|".join(alternatives), text)
    else:
        ordinary_segments = [text]

    pretoken_counts: Counter[bytes] = Counter()
    for segment in ordinary_segments:
        for match in regex.finditer(PAT, segment):
            pretoken_counts[match.group().encode("utf-8")] += 1

    words: dict[int, tuple[bytes, ...]] = {
        word_id: tuple(bytes([byte]) for byte in pretoken)
        for word_id, pretoken in enumerate(pretoken_counts)
    }
    frequencies = {word_id: pretoken_counts[pretoken] for word_id, pretoken in enumerate(pretoken_counts)}

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, word in words.items():
        for pair, occurrences in _pair_occurrences(word).items():
            pair_counts[pair] += occurrences * frequencies[word_id]
            pair_to_words[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        if pair_counts[best_pair] <= 0:
            break
        affected_word_ids = tuple(pair_to_words[best_pair])
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        for word_id in affected_word_ids:
            old_word = words[word_id]
            frequency = frequencies[word_id]

            for pair, occurrences in _pair_occurrences(old_word).items():
                pair_counts[pair] -= occurrences * frequency
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word_id)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            new_word = _merge_sequence(old_word, best_pair)
            words[word_id] = new_word
            for pair, occurrences in _pair_occurrences(new_word).items():
                pair_counts[pair] += occurrences * frequency
                pair_to_words[pair].add(word_id)

    return vocab, merges


class Tokenizer:
    """Encode Unicode text with learned byte merges and decode IDs back to text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        existing = set(self.vocab.values())
        for special in self.special_tokens:
            encoded = special.encode("utf-8")
            if encoded not in existing:
                self.vocab[len(self.vocab)] = encoded
                existing.add(encoded)

        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_to_id = {
            special: self.bytes_to_id[special.encode("utf-8")] for special in self.special_tokens
        }
        if self.special_tokens:
            alternatives = sorted((regex.escape(token) for token in self.special_tokens), key=len, reverse=True)
            self.special_pattern: regex.Pattern[str] | None = regex.compile(
                f"({'|'.join(alternatives)})"
            )
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load the simple JSON/line-pair format used by this walkthrough."""

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)
        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in raw_vocab.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                left, right = line.rstrip().split("\t")
                merges.append((bytes.fromhex(left), bytes.fromhex(right)))
        return cls(vocab, merges, special_tokens=special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        pieces: list[bytes] = [bytes([byte]) for byte in pretoken.encode("utf-8")]
        while len(pieces) >= 2:
            candidates = [
                (self.merge_rank[pair], index, pair)
                for index, pair in enumerate(zip(pieces, pieces[1:]))
                if pair in self.merge_rank
            ]
            if not candidates:
                break
            _, _, best_pair = min(candidates)
            pieces = list(_merge_sequence(tuple(pieces), best_pair))
        return [self.bytes_to_id[piece] for piece in pieces]

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in regex.finditer(PAT, text):
            yield from self._encode_pretoken(match.group())

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self.special_pattern is None:
            return list(self._encode_ordinary(text))

        ids: list[int] = []
        for piece in self.special_pattern.split(text):
            if not piece:
                continue
            special_id = self.special_to_id.get(piece)
            if special_id is not None:
                ids.append(special_id)
            else:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode lazily; each yielded string is consumed before the next is requested."""

        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
