"""Conservative masking for common email, US phone and IPv4 forms."""

from __future__ import annotations

import re


EMAIL_TOKEN = "|||EMAIL_ADDRESS|||"
PHONE_TOKEN = "|||PHONE_NUMBER|||"
IP_TOKEN = "|||IP_ADDRESS|||"

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
    r"(?![\w-])",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?<![\w\d])"
    r"(?:\+?1[\s.-]?)?"
    r"(?:\(\s*\d{3}\s*\)|\d{3})"
    r"[\s.-]?\d{3}[\s.-]?\d{4}"
    r"(?![\w\d])"
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_PATTERN = re.compile(rf"(?<![\d.])(?:{_OCTET}\.){{3}}{_OCTET}(?!\d|\.\d)")


def _mask(pattern: re.Pattern[str], token: str, text: str) -> tuple[str, int]:
    return pattern.subn(token, text)


def mask_emails(text: str) -> tuple[str, int]:
    return _mask(EMAIL_PATTERN, EMAIL_TOKEN, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _mask(PHONE_PATTERN, PHONE_TOKEN, text)


def mask_ipv4_addresses(text: str) -> tuple[str, int]:
    return _mask(IPV4_PATTERN, IP_TOKEN, text)
