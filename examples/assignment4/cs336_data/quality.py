"""Rule-based and learned document-quality classifiers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import fasttext
import regex


WORD_PATTERN = regex.compile(r"[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*")


def words(text: str) -> list[str]:
    # Unlike whitespace splitting, this separates the common crawl artifact
    # ``sentence.Next`` at punctuation while retaining apostrophes in words.
    return WORD_PATTERN.findall(text)


def passes_gopher_quality_filter(text: str) -> bool:
    tokens = words(text)
    if not 50 <= len(tokens) <= 100_000:
        return False

    mean_length = sum(len(token) for token in tokens) / len(tokens)
    if not 3 <= mean_length <= 10:
        return False

    nonempty_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        ellipsis_fraction = sum(line.endswith("...") for line in nonempty_lines) / len(nonempty_lines)
        if ellipsis_fraction > 0.30:
            return False

    alphabetic_fraction = sum(any(character.isalpha() for character in token) for token in tokens) / len(tokens)
    if alphabetic_fraction < 0.80:
        return False
    return True


class FastTextQualityClassifier:
    def __init__(self, model_path: str | Path):
        self.model = fasttext.load_model(str(model_path))

    def classify(self, text: str) -> tuple[str, float]:
        labels, probabilities = self.model.predict(text.replace("\n", " "), k=1)
        return labels[0].removeprefix("__label__"), float(probabilities[0])


def train_quality_classifier(
    positive_documents: Iterable[str],
    negative_documents: Iterable[str],
    output_path: str | Path,
    *,
    epoch: int = 10,
) -> Path:
    """Train the Wikipedia-reference-vs-random-CC fastText classifier."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as training_file:
        training_path = Path(training_file.name)
        for text in positive_documents:
            training_file.write("__label__wiki " + " ".join(text.split()) + "\n")
        for text in negative_documents:
            training_file.write("__label__cc " + " ".join(text.split()) + "\n")
    try:
        model = fasttext.train_supervised(
            input=str(training_path),
            epoch=epoch,
            lr=0.2,
            wordNgrams=2,
            dim=100,
            loss="softmax",
            thread=1,
        )
        model.save_model(str(output_path))
    finally:
        training_path.unlink(missing_ok=True)
    return output_path
