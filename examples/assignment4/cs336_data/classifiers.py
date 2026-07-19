"""Thin, label-normalizing wrappers around the assignment's fastText models."""

from __future__ import annotations

from pathlib import Path

import fasttext


class FastTextClassifier:
    def __init__(self, model_path: str | Path):
        self.model = fasttext.load_model(str(model_path))

    def predict(self, text: str) -> tuple[str, float]:
        labels, probabilities = self.model.predict(text.replace("\n", " "), k=1)
        label = labels[0].removeprefix("__label__")
        return label, float(probabilities[0])


class LanguageIdentifier(FastTextClassifier):
    """Return ISO-like language IDs from lid.176.bin."""

    LABEL_ALIASES = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "cmn": "zh",
    }

    def identify(self, text: str) -> tuple[str, float]:
        label, score = self.predict(text)
        return self.LABEL_ALIASES.get(label.lower(), label.lower()), score


class BinaryHarmClassifier(FastTextClassifier):
    """Normalize differing Dolma/Jigsaw label spellings to one public API."""

    def __init__(
        self,
        model_path: str | Path,
        positive_output: str,
        negative_output: str,
        positive_labels: set[str],
        negative_labels: set[str],
    ):
        super().__init__(model_path)
        self.positive_output = positive_output
        self.negative_output = negative_output
        self.positive_labels = {self._normalize(label) for label in positive_labels}
        self.negative_labels = {self._normalize(label) for label in negative_labels}

    @staticmethod
    def _normalize(label: str) -> str:
        return label.removeprefix("__label__").lower().replace("-", "_")

    def classify(self, text: str) -> tuple[str, float]:
        label, score = self.predict(text)
        normalized = self._normalize(label)
        if normalized in self.positive_labels:
            return self.positive_output, score
        if normalized in self.negative_labels:
            return self.negative_output, score
        raise ValueError(
            f"unexpected label {label!r}; expected one of "
            f"{sorted(self.positive_labels | self.negative_labels)}"
        )


def load_nsfw_classifier(model_path: str | Path) -> BinaryHarmClassifier:
    return BinaryHarmClassifier(
        model_path,
        positive_output="nsfw",
        negative_output="non-nsfw",
        positive_labels={"nsfw", "obscene", "porn", "1"},
        negative_labels={"non-nsfw", "non_nsfw", "safe", "0"},
    )


def load_toxicity_classifier(model_path: str | Path) -> BinaryHarmClassifier:
    return BinaryHarmClassifier(
        model_path,
        positive_output="toxic",
        negative_output="non-toxic",
        positive_labels={"toxic", "toxicity", "hate", "hatespeech", "1"},
        negative_labels={"non-toxic", "non_toxic", "non-hate", "non_hate", "0"},
    )
