import subprocess
from unittest.mock import patch
from src.rewrite import rewrite_query, parse_variants


def test_parse_variants_extracts_numbered_list():
    raw = "1. What is an ERD?\n2. Entity relationship diagram definition\n3. ERD data modeling"
    variants = parse_variants(raw)
    assert len(variants) == 3
    assert "ERD" in variants[0]


def test_parse_variants_handles_bullet_list():
    raw = "- ERD definition\n- entity relationship diagram\n- data modeling entities"
    variants = parse_variants(raw)
    assert len(variants) == 3


def test_rewrite_query_returns_list_of_strings():
    with patch("src.rewrite.subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1. ERD definition\n2. entity relationship diagram\n3. data modeling"
        mock_run.return_value.returncode = 0
        variants = rewrite_query("What is an ERD?")
    assert isinstance(variants, list)
    assert len(variants) >= 1
    assert all(isinstance(v, str) for v in variants)
    # original query always included
    assert "What is an ERD?" in variants


def test_rewrite_query_falls_back_on_timeout():
    with patch("src.rewrite.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
        variants = rewrite_query("What is an ERD?")
    assert variants == ["What is an ERD?"]
