from brain.verify import parse_citations, format_verified_response, VerifiedClaim


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
