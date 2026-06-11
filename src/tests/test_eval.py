from unittest.mock import MagicMock, patch

from src.eval import (
    _ask_claude_for_question,
    evaluate_ranking,
    make_snippet,
    mean_reciprocal_rank,
    parse_generated_question,
    recall_at_k,
    sample_eval_chunks,
)


def _proc(returncode=0, stdout="", stderr=""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_ask_claude_retries_transient_failures():
    # First call fails (e.g. rate limit), second succeeds — must not give up.
    with patch("src.eval.subprocess.run") as mock_run, \
         patch("src.eval.time.sleep") as mock_sleep:
        mock_run.side_effect = [
            _proc(returncode=1, stderr="rate limit exceeded"),
            _proc(stdout="What is an ERD?\n"),
        ]
        result = _ask_claude_for_question("chunk text")
    assert result == "What is an ERD?"
    assert mock_run.call_count == 2
    mock_sleep.assert_called()


def test_ask_claude_generates_with_haiku_model():
    # Question generation is a simple task — pin the cheap, fast model so
    # 50-call bursts cost less and are less likely to hit rate limits.
    with patch("src.eval.subprocess.run") as mock_run:
        mock_run.return_value = _proc(stdout="What is an ERD?\n")
        _ask_claude_for_question("chunk text")
    cmd = mock_run.call_args.args[0]
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "haiku"


def test_ask_claude_reports_reason_after_all_attempts_fail(capsys):
    with patch("src.eval.subprocess.run") as mock_run, \
         patch("src.eval.time.sleep"):
        mock_run.return_value = _proc(returncode=1, stderr="rate limit exceeded")
        result = _ask_claude_for_question("chunk text")
    assert result is None
    assert mock_run.call_count == 3
    assert "rate limit exceeded" in capsys.readouterr().err


def _chunk(filename="a.pdf", text="some chunk text here", course="IS 6410", is_stub=False):
    return {
        "id": f"{course}_{filename}_p1_c0",
        "filename": filename,
        "course": course,
        "page": 1,
        "text": text,
        "is_stub": is_stub,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_recall_at_k_counts_ranks_within_k():
    ranks = [1, 3, None, 6]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.75


def test_recall_at_k_empty_ranks_is_zero():
    assert recall_at_k([], 5) == 0.0


def test_mean_reciprocal_rank_averages_inverse_ranks():
    # 1/1 + 1/2 + 0 (miss) = 1.5 / 3 = 0.5
    assert mean_reciprocal_rank([1, 2, None]) == 0.5


def test_mean_reciprocal_rank_empty_is_zero():
    assert mean_reciprocal_rank([]) == 0.0


# ---------------------------------------------------------------------------
# Hit detection against a ranked list
# ---------------------------------------------------------------------------

def test_evaluate_ranking_finds_doc_and_passage_ranks():
    gold = {
        "gold_filename": "b.pdf",
        "gold_snippet": "entity relationship diagrams model data",
    }
    ranked = [
        _chunk(filename="b.pdf", text="totally unrelated content about agile sprints"),
        _chunk(filename="c.pdf", text="Entity   Relationship\nDiagrams MODEL data entities"),
    ]
    result = evaluate_ranking(gold, ranked)
    assert result["doc_rank"] == 1
    assert result["passage_rank"] == 2


def test_evaluate_ranking_returns_none_on_miss():
    gold = {"gold_filename": "z.pdf", "gold_snippet": "nothing matches this snippet"}
    ranked = [_chunk(filename="a.pdf", text="unrelated")]
    result = evaluate_ranking(gold, ranked)
    assert result["doc_rank"] is None
    assert result["passage_rank"] is None


def test_evaluate_ranking_passage_match_ignores_case_and_whitespace():
    gold = {"gold_filename": "x.pdf", "gold_snippet": "Supply Chain  Resilience"}
    ranked = [_chunk(filename="x.pdf", text="intro to supply\nchain resilience in 2024")]
    result = evaluate_ranking(gold, ranked)
    assert result["passage_rank"] == 1


# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------

def test_make_snippet_short_text_returns_whole_text():
    assert make_snippet("short text", length=120) == "short text"


def test_make_snippet_long_text_takes_middle_slice():
    text = "a" * 100 + "MIDDLE" + "b" * 100
    snippet = make_snippet(text, length=40)
    assert len(snippet) == 40
    assert "MIDDLE" in snippet
    assert snippet in text


# ---------------------------------------------------------------------------
# Question generation parsing
# ---------------------------------------------------------------------------

def test_parse_generated_question_strips_numbering_and_quotes():
    assert parse_generated_question('1. "What is an ERD?"') == "What is an ERD?"


def test_parse_generated_question_takes_first_nonempty_line():
    raw = "\n\nWhat are the three phases?\nExtra commentary here."
    assert parse_generated_question(raw) == "What are the three phases?"


def test_parse_generated_question_empty_returns_none():
    assert parse_generated_question("   \n  ") is None


# ---------------------------------------------------------------------------
# Chunk sampling for question generation
# ---------------------------------------------------------------------------

def test_sample_eval_chunks_excludes_stubs_and_short_chunks():
    chunks = [
        _chunk(text="x" * 300),
        _chunk(text="too short", filename="b.pdf"),
        _chunk(text="y" * 300, filename="c.pdf", is_stub=True),
    ]
    sampled = sample_eval_chunks(chunks, n=3, min_chars=200)
    assert len(sampled) == 1
    assert sampled[0]["filename"] == "a.pdf"


def test_sample_eval_chunks_is_deterministic():
    chunks = [_chunk(text=f"{'z' * 250} chunk {i}", filename=f"f{i}.pdf") for i in range(20)]
    first = sample_eval_chunks(chunks, n=5, min_chars=200)
    second = sample_eval_chunks(chunks, n=5, min_chars=200)
    assert [c["id"] for c in first] == [c["id"] for c in second]


def test_sample_eval_chunks_stratifies_across_courses():
    chunks = (
        [_chunk(text="a" * 300, filename=f"a{i}.pdf", course="IS 6410") for i in range(10)]
        + [_chunk(text="b" * 300, filename=f"b{i}.pdf", course="OSC 6660") for i in range(10)]
    )
    sampled = sample_eval_chunks(chunks, n=10, min_chars=200)
    courses = [c["course"] for c in sampled]
    assert courses.count("IS 6410") == 5
    assert courses.count("OSC 6660") == 5
