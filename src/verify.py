import re
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from sentence_transformers import CrossEncoder
from src.config import ENTAILMENT_THRESHOLD, NLI_MODEL


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


@lru_cache(maxsize=1)
def _get_nli_model() -> tuple:
    """Load the NLI model and detect which output index corresponds to entailment.

    NLI models return 3 scores (contradiction, neutral, entailment) but the
    column order varies by model. We read the label mapping from the model's
    own config rather than hardcoding an index.
    """
    model = CrossEncoder(NLI_MODEL)
    id2label = model.model.config.id2label
    entailment_idx = next(
        (i for i, label in id2label.items() if "entail" in label.lower()),
        1,  # fallback
    )
    return model, entailment_idx


def _entailment_score(chunk_text: str, claim_text: str) -> float:
    """Return the probability that chunk_text entails claim_text (0.0–1.0).

    Uses a proper NLI model rather than a retrieval cross-encoder. The chunk
    is the premise; the claim is the hypothesis. apply_softmax=True converts
    raw logits to probabilities that sum to 1 across the three NLI labels.
    """
    model, entailment_idx = _get_nli_model()
    # NLI input order: (premise, hypothesis)
    scores = model.predict([(chunk_text, claim_text)], apply_softmax=True)
    scores = np.atleast_2d(scores)
    return float(scores[0, entailment_idx])


def parse_citations(response: str) -> list[VerifiedClaim]:
    claims = []
    matches = list(_CITATION_RE.finditer(response))
    if not matches:
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
    """Verify each cited claim using NLI entailment scoring.

    Asks: does the cited chunk (premise) logically entail the claim (hypothesis)?
    This is meaningfully different from relevance scoring — a passage can be
    relevant to a topic while not actually supporting a specific factual claim.
    """
    # Slide chunks span a page range (page_start..page_end) but are keyed at
    # the first page; a citation to any page in the range must match. Chunks
    # without range metadata fall back to exact-page matching.
    def _texts_for(filename: str, page: int) -> str:
        texts = []
        for c in chunks:
            if c.get("filename", "") != filename:
                continue
            start = c.get("page_start", c.get("page", 0))
            end = c.get("page_end", c.get("page", 0))
            if start <= page <= end:
                texts.append(c["text"])
        return " ".join(texts)

    for claim in claims:
        if claim.filename is None:
            continue
        chunk_text = _texts_for(claim.filename, claim.page)
        if not chunk_text:
            claim.verified = False
            continue
        claim.chunk_text = chunk_text
        score = _entailment_score(chunk_text, claim.claim_text)
        claim.verified = score >= ENTAILMENT_THRESHOLD
    return claims


def format_verified_response(claims: list[VerifiedClaim]) -> str:
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
