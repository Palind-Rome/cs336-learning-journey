"""Reference data-curation primitives for CS336 Assignment 4."""

from .deduplication import exact_line_deduplicate, minhash_deduplicate
from .extraction import extract_text_from_html_bytes
from .pii import mask_emails, mask_ipv4_addresses, mask_phone_numbers
from .quality import passes_gopher_quality_filter

__all__ = [
    "exact_line_deduplicate",
    "extract_text_from_html_bytes",
    "mask_emails",
    "mask_ipv4_addresses",
    "mask_phone_numbers",
    "minhash_deduplicate",
    "passes_gopher_quality_filter",
]
