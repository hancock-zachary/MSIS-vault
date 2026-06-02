import re
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from brain.config import ENTAILMENT_THRESHOLD
from brain.rerank import _get_model


@dataclass
class VerifiedClaim:
    claim_text: str
    filename: str | None
    page: int | None
    chunk_text: str | None
    verified: bool


_CITATION_RE = re.compile(
    r"([^[]+?)\s*\[source:\s*([^,\]]+),\s*page\s*(\d+)\]",
    re.IGNORECASE,
)


def parse_citations(response: str) -> list[VerifiedClaim]:
    """Parse inline citations from response text.

    Extracts claims in the format: "claim text [source: filename, page N]"
    If no citations found, treats entire response as one uncited claim.

    Args:
        response: The response text containing inline citations.

    Returns:
        List of VerifiedClaim objects with filename and page extracted.
    """
    claims = []
    matches = list(_CITATION_RE.finditer(response))
    if not matches:
        # treat entire response as one uncited claim
        return [VerifiedClaim(response.strip(), None, None, None, False)]
    for m in matches:
        claims.append(VerifiedClaim(
            claim_text=m.group(1).strip(),
            filename=m.group(2).strip(),
            page=int(m.group(3)),
            chunk_text=None,
            verified=False,
        ))
    return claims


def verify_claims(claims: list[VerifiedClaim], chunks: list[dict]) -> list[VerifiedClaim]:
    """Verify claims against retrieved chunks using cross-encoder entailment scoring.

    For each claim with a citation, looks up the corresponding chunk and uses
    the cross-encoder model to score whether the claim is entailed by the chunk.
    Claims scoring >= ENTAILMENT_THRESHOLD are marked as verified.

    Args:
        claims: List of VerifiedClaim objects to verify.
        chunks: List of chunk dicts with 'filename', 'page', and 'text' keys.

    Returns:
        List of VerifiedClaim objects with updated verified status and chunk_text.
    """
    page_texts: dict[tuple, list[str]] = defaultdict(list)
    for c in chunks:
        key = (c.get("filename", ""), c.get("page", 0))
        page_texts[key].append(c["text"])
    chunk_map = {k: " ".join(v) for k, v in page_texts.items()}
    model = _get_model()
    for claim in claims:
        if claim.filename is None:
            continue
        chunk_text = chunk_map.get((claim.filename, claim.page))
        if not chunk_text:
            claim.verified = False
            continue
        claim.chunk_text = chunk_text
        raw = model.predict([(claim.claim_text, chunk_text)])
        score = float(np.atleast_1d(raw)[0])
        claim.verified = score >= ENTAILMENT_THRESHOLD
    return claims


def format_verified_response(claims: list[VerifiedClaim]) -> str:
    """Format verified claims with grounding ratio.

    Outputs each claim with its citation status and appends a summary line
    showing the ratio of verified to total claims.

    Unverified claims and uncited claims are marked with ⚠ symbol.

    Args:
        claims: List of VerifiedClaim objects to format.

    Returns:
        Formatted string with claims and grounding ratio.
    """
    lines = []
    verified_count = sum(1 for c in claims if c.verified)
    for claim in claims:
        if claim.filename is None:
            lines.append(f"{claim.claim_text} ⚠ unverified (no citation)")
        elif claim.verified:
            lines.append(f"{claim.claim_text} [source: {claim.filename}, page {claim.page}]")
        else:
            lines.append(f"{claim.claim_text} [source: {claim.filename}, page {claim.page}] ⚠ unverified")
    lines.append(f"Grounding: {verified_count}/{len(claims)} claims verified")
    return "\n".join(lines)
