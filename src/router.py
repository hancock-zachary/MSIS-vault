"""Signal-based chunking router.

Computes a ChunkingProfile from measurable document signals and routes each
document to the appropriate chunking strategy.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from src.chunk import is_quality_text
from src.config import SLIDE_PAGE_TOKEN_THRESHOLD

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkingProfile:
    strategy: str              # "slides" | "semantic" | "structured" | "window"
    avg_tokens_per_page: float
    has_structure: bool        # headings or PDF outline detected
    garbled_page_ratio: float  # fraction of pages that failed quality check
    file_extension: str
    page_count: int


def _avg_tokens_per_page(pages: list[dict]) -> float:
    if not pages:
        return 0.0
    return sum(len(_enc.encode(p["text"])) for p in pages) / len(pages)


def _has_structure(path: Path, pages: list[dict]) -> bool:
    """True if the document has detectable heading structure."""
    ext = path.suffix.lower()
    if ext in (".docx", ".md"):
        return True
    if ext == ".pdf":
        return any(p.get("slide_title", "") for p in pages)
    return False


def _compute_garbled_ratio(pages: list[dict]) -> float:
    if not pages:
        return 0.0
    garbled = sum(1 for p in pages if not is_quality_text(p["text"]))
    return garbled / len(pages)


def build_profile(path: Path, pages: list[dict]) -> ChunkingProfile:
    """Compute a ChunkingProfile from measurable document signals.

    Routing priority:
    1. .docx or .md → structured
    2. .pdf + avg tokens/page < SLIDE_PAGE_TOKEN_THRESHOLD → slides
    3. .pdf + outline/headings detected → structured
    4. everything else → semantic
    """
    ext = path.suffix.lower()
    avg_tokens = _avg_tokens_per_page(pages)
    has_struct = _has_structure(path, pages)
    garbled_ratio = _compute_garbled_ratio(pages)
    page_count = len(pages)

    if ext in (".docx", ".md"):
        strategy = "structured"
    elif ext == ".pdf" and avg_tokens < SLIDE_PAGE_TOKEN_THRESHOLD:
        strategy = "slides"
    elif ext == ".pdf" and has_struct:
        strategy = "structured"
    else:
        strategy = "semantic"

    return ChunkingProfile(
        strategy=strategy,
        avg_tokens_per_page=round(avg_tokens, 1),
        has_structure=has_struct,
        garbled_page_ratio=round(garbled_ratio, 3),
        file_extension=ext,
        page_count=page_count,
    )
