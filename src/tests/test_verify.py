from unittest.mock import patch, MagicMock
import numpy as np
from src.verify import parse_citations, format_verified_response, verify_claims, VerifiedClaim


def test_parse_citations_extracts_inline_sources():
    response = (
        "An ERD models entities and relationships [source: week3.pdf, page 12]. "
        "Normalization reduces redundancy [source: week4.pdf, page 5]."
    )
    claims = parse_citations(response)
    assert len(claims) == 2
    assert claims[0].filename == "week3.pdf"
    assert claims[0].page == 12
    assert "ERD models entities" in claims[0].claim_text
    assert claims[1].filename == "week4.pdf"
    assert claims[1].page == 5
    assert "Normalization reduces redundancy" in claims[1].claim_text


def test_parse_citations_no_sources_returns_uncited():
    response = "This claim has no citation."
    claims = parse_citations(response)
    assert len(claims) == 1
    assert claims[0].filename is None
    assert claims[0].page is None
    assert claims[0].claim_text == "This claim has no citation."


def test_format_verified_response_shows_grounding_ratio():
    claims = [
        VerifiedClaim("Claim A", "a.pdf", 1, "chunk text", True),
        VerifiedClaim("Claim B", "b.pdf", 2, "chunk text", False),
    ]
    output = format_verified_response(claims)
    assert "1/2 claims verified" in output
    assert "⚠ unverified" in output


def test_verify_claims_marks_entailed_claim_verified():
    claims = [VerifiedClaim("The Scrum Master removes impediments", "scrum.pdf", 5, None, False)]
    chunks = [{"filename": "scrum.pdf", "page": 5, "text": "The Scrum Master removes impediments for the team."}]

    mock_model = MagicMock()
    # NLI returns (n_pairs, 3) softmax probabilities — entailment is index 1
    mock_model.predict.return_value = np.array([[0.05, 0.90, 0.05]])
    mock_model.model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    with patch("src.verify._get_nli_model", return_value=(mock_model, 1)):
        result = verify_claims(claims, chunks)

    assert result[0].verified is True
    assert result[0].chunk_text is not None


def test_verify_claims_flags_contradicted_claim_unverified():
    claims = [VerifiedClaim("The Scrum Master manages the team", "scrum.pdf", 5, None, False)]
    chunks = [{"filename": "scrum.pdf", "page": 5, "text": "The Scrum Master has no authority over the team."}]

    mock_model = MagicMock()
    # High contradiction score, low entailment
    mock_model.predict.return_value = np.array([[0.85, 0.05, 0.10]])
    mock_model.model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    with patch("src.verify._get_nli_model", return_value=(mock_model, 1)):
        result = verify_claims(claims, chunks)

    assert result[0].verified is False


def test_verify_claims_skips_uncited_claims():
    claims = [VerifiedClaim("Some uncited claim", None, None, None, False)]
    result = verify_claims(claims, [])
    assert result[0].verified is False
    assert result[0].chunk_text is None


def test_verify_claims_skips_missing_chunk():
    claims = [VerifiedClaim("A claim", "missing.pdf", 99, None, False)]
    chunks = [{"filename": "other.pdf", "page": 1, "text": "irrelevant"}]
    with patch("src.verify._get_nli_model"):
        result = verify_claims(claims, chunks)
    assert result[0].verified is False
