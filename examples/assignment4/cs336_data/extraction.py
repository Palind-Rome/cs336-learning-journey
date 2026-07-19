"""Robust HTML byte decoding and main-text extraction."""

from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def decode_html(html_bytes: bytes) -> str:
    """Decode UTF-8 quickly, falling back to Resiliparse charset detection."""

    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        if not encoding:
            encoding = "utf-8"
        return html_bytes.decode(encoding, errors="replace")


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    html = decode_html(html_bytes)
    return extract_plain_text(html)
