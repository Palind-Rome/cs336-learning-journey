"""Stream JSONL documents into the uint16 GPT-2 token format used by training."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Iterable

import numpy as np
import tiktoken


_ENCODING = None


def _initialize_worker() -> None:
    global _ENCODING
    _ENCODING = tiktoken.get_encoding("gpt2")


def _encode_document(text: str) -> list[int]:
    if _ENCODING is None:
        _initialize_worker()
    return _ENCODING.encode_ordinary(text) + [_ENCODING.eot_token]


def iter_jsonl_text(paths: Iterable[str | Path]):
    for path in paths:
        with Path(path).open(encoding="utf-8") as file:
            for line in file:
                yield json.loads(line)["text"]


def tokenize_jsonl(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    workers: int | None = None,
    chunksize: int = 100,
) -> int:
    """Write incrementally so billions of token IDs never live in one Python list."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    with output_path.open("wb") as output, multiprocessing.Pool(
        workers or multiprocessing.cpu_count(), initializer=_initialize_worker
    ) as pool:
        for token_ids in pool.imap(
            _encode_document, iter_jsonl_text(input_paths), chunksize=chunksize
        ):
            array = np.asarray(token_ids, dtype=np.uint16)
            array.tofile(output)
            token_count += len(array)
    return token_count
