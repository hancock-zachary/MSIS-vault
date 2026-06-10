"""
Batch evaluation harness for the retrieval pipeline.

Usage:
  uv run python src/eval.py generate [--n 50] [--force]
      Sample indexed chunks and auto-generate one question per chunk via
      `claude -p`. Writes eval/questions.json (editable — curate freely).

  uv run python src/eval.py run [--save-baseline]
      Score every question against the live pipeline (hybrid retrieve +
      rerank) and print recall@k / MRR at document and passage level.
      Compares against eval/baseline.json when present and saves a
      timestamped copy to eval/results/.

Design notes:
  - Query rewriting is deliberately skipped: it shells out to `claude -p`
    per question, which makes scores nondeterministic. The harness measures
    hybrid_retrieve + rerank on the raw question so runs are reproducible.
  - Gold answers are matched by filename (document level) and by normalized
    snippet containment (passage level), not chunk id — chunk ids and page
    numbers change whenever chunking parameters change, which is exactly
    when this harness needs to produce comparable numbers.
"""
import argparse
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

from src.config import (
    EVAL_BASELINE_PATH, EVAL_DIR, EVAL_MIN_CHUNK_CHARS, EVAL_QUESTION_COUNT,
    EVAL_QUESTIONS_PATH, EVAL_RESULTS_DIR, EVAL_SAMPLE_SEED, EVAL_SNIPPET_CHARS,
    TOP_K_RERANK, TOP_K_RETRIEVAL,
)

GENERATE_PROMPT = """\
You write exam-style study questions for a graduate business school student.
Below is a passage from course material. Write ONE specific factual question
that this passage directly answers. The question must be answerable from this
passage alone, phrased the way a student would naturally ask it, and must not
refer to "the passage" or "the text".

Output only the question, nothing else.

Passage:
{text}
"""

_RETRIEVAL_KS = (1, 5, 8, 20)
_RERANK_KS = (1, 5, 8)


# ---------------------------------------------------------------------------
# Metrics (pure functions)
# ---------------------------------------------------------------------------

def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose gold answer appeared at rank <= k."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Mean of 1/rank across questions; misses contribute 0."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def evaluate_ranking(gold: dict, ranked: list[dict]) -> dict:
    """Find the rank (1-indexed) of the gold answer in a ranked chunk list.

    Returns doc_rank (first chunk from the gold filename) and passage_rank
    (first chunk whose text contains the gold snippet, whitespace- and
    case-insensitive). None means the gold answer never appeared.
    """
    doc_rank = None
    passage_rank = None
    snippet = _normalize(gold["gold_snippet"])
    for rank, chunk in enumerate(ranked, start=1):
        if doc_rank is None and chunk.get("filename") == gold["gold_filename"]:
            doc_rank = rank
        if passage_rank is None and snippet in _normalize(chunk.get("text", "")):
            passage_rank = rank
        if doc_rank is not None and passage_rank is not None:
            break
    return {"doc_rank": doc_rank, "passage_rank": passage_rank}


# ---------------------------------------------------------------------------
# Question generation helpers (pure functions)
# ---------------------------------------------------------------------------

def make_snippet(text: str, length: int = EVAL_SNIPPET_CHARS) -> str:
    """Take a slice from the middle of the text — chunk edges are the most
    likely parts to move when chunk boundaries change."""
    text = text.strip()
    if len(text) <= length:
        return text
    start = (len(text) - length) // 2
    return text[start:start + length]


def parse_generated_question(raw: str) -> str | None:
    """Extract the question from claude output: first non-empty line,
    stripped of list numbering and surrounding quotes."""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(\d+\.\s*|[-*]\s+|Q:\s*)", "", line).strip()
        line = line.strip('"“”').strip()
        return line or None
    return None


def sample_eval_chunks(
    chunks: list[dict],
    n: int,
    min_chars: int = EVAL_MIN_CHUNK_CHARS,
    seed: int = EVAL_SAMPLE_SEED,
) -> list[dict]:
    """Sample up to n chunks, stratified across courses (round-robin), with a
    fixed seed for reproducibility. Stubs and short chunks are excluded —
    they don't contain enough content to generate a fair question from."""
    eligible = [
        c for c in chunks
        if not c.get("is_stub") and len(c.get("text", "")) >= min_chars
    ]
    by_course: dict[str, list[dict]] = defaultdict(list)
    for c in eligible:
        by_course[c.get("course", "Unknown")].append(c)

    rng = random.Random(seed)
    for course_chunks in by_course.values():
        course_chunks.sort(key=lambda c: c["id"])  # stable base order before shuffle
        rng.shuffle(course_chunks)

    sampled = []
    courses = sorted(by_course)
    while len(sampled) < n and any(by_course[c] for c in courses):
        for course in courses:
            if by_course[course] and len(sampled) < n:
                sampled.append(by_course[course].pop())
    return sampled


# ---------------------------------------------------------------------------
# generate subcommand
# ---------------------------------------------------------------------------

def _ask_claude_for_question(chunk_text: str) -> str | None:
    prompt = GENERATE_PROMPT.format(text=chunk_text)
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return parse_generated_question(result.stdout)


def generate_questions(n: int, force: bool) -> None:
    from src.index import get_collection

    if EVAL_QUESTIONS_PATH.exists() and not force:
        print(f"{EVAL_QUESTIONS_PATH} already exists (it may contain hand-edits).")
        print("Re-run with --force to overwrite.")
        sys.exit(1)

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"id": cid, "text": text, **meta}
        for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    if not chunks:
        print("Index is empty. Run src/ingest.py first.")
        sys.exit(1)

    sampled = sample_eval_chunks(chunks, n)
    print(f"Generating questions for {len(sampled)} chunks "
          f"({len(chunks)} indexed, stratified across courses)...")

    questions = []
    for i, chunk in enumerate(sampled, 1):
        question = _ask_claude_for_question(chunk["text"])
        if not question:
            print(f"  [{i}/{len(sampled)}] generation failed, skipping {chunk['id']}")
            continue
        questions.append({
            "id": f"q{len(questions) + 1:03d}",
            "question": question,
            "gold_filename": chunk.get("filename", ""),
            "gold_page": chunk.get("page", 0),
            "gold_course": chunk.get("course", ""),
            "gold_snippet": make_snippet(chunk["text"]),
        })
        print(f"  [{i}/{len(sampled)}] {question}")

    EVAL_DIR.mkdir(exist_ok=True)
    payload = {
        "created": datetime.now().strftime("%Y-%m-%d"),
        "note": "Auto-generated by src/eval.py — edit freely; gold matching uses filename + snippet.",
        "questions": questions,
    }
    EVAL_QUESTIONS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {len(questions)} questions to {EVAL_QUESTIONS_PATH}")


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

def _aggregate(per_question: list[dict], stage: str, ks: tuple) -> dict:
    metrics = {}
    for level in ("doc", "passage"):
        ranks = [q[stage][f"{level}_rank"] for q in per_question]
        for k in ks:
            metrics[f"{level}_recall@{k}"] = round(recall_at_k(ranks, k), 3)
        metrics[f"{level}_mrr"] = round(mean_reciprocal_rank(ranks), 3)
    return metrics


def _print_metrics(title: str, metrics: dict, baseline: dict | None) -> None:
    print(f"\n{title}")
    for name, value in metrics.items():
        line = f"  {name:<20} {value:.3f}"
        if baseline and name in baseline:
            delta = value - baseline[name]
            line += f"   ({'+' if delta >= 0 else ''}{delta:.3f} vs baseline)"
        print(line)


def run_eval(save_baseline: bool) -> None:
    from src.rerank import rerank_chunks
    from src.retrieve import hybrid_retrieve

    if not EVAL_QUESTIONS_PATH.exists():
        print(f"No question set at {EVAL_QUESTIONS_PATH}. Run: python src/eval.py generate")
        sys.exit(1)

    payload = json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if not questions:
        print("Question set is empty.")
        sys.exit(1)

    print(f"Evaluating {len(questions)} questions "
          f"(retrieval k={TOP_K_RETRIEVAL}, rerank k={TOP_K_RERANK})...")

    per_question = []
    for i, q in enumerate(questions, 1):
        candidates = hybrid_retrieve([q["question"]])
        retrieved = candidates[:TOP_K_RETRIEVAL]
        reranked = rerank_chunks(q["question"], candidates, top_k=TOP_K_RERANK)
        per_question.append({
            "id": q["id"],
            "retrieval": evaluate_ranking(q, retrieved),
            "rerank": evaluate_ranking(q, reranked),
        })
        print(f"  [{i}/{len(questions)}] {q['id']}", file=sys.stderr)

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question_count": len(questions),
        "retrieval": _aggregate(per_question, "retrieval", _RETRIEVAL_KS),
        "rerank": _aggregate(per_question, "rerank", _RERANK_KS),
        "per_question": per_question,
    }

    baseline = None
    if EVAL_BASELINE_PATH.exists():
        baseline = json.loads(EVAL_BASELINE_PATH.read_text(encoding="utf-8"))

    _print_metrics(f"Retrieval stage (top {TOP_K_RETRIEVAL})", results["retrieval"],
                   baseline.get("retrieval") if baseline else None)
    _print_metrics(f"Rerank stage (top {TOP_K_RERANK})", results["rerank"],
                   baseline.get("rerank") if baseline else None)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_path = EVAL_RESULTS_DIR / f"{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    if save_baseline:
        EVAL_BASELINE_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Baseline updated: {EVAL_BASELINE_PATH}")
    elif baseline is None:
        print("No baseline yet — re-run with --save-baseline to set one.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="auto-generate the question set")
    gen.add_argument("--n", type=int, default=EVAL_QUESTION_COUNT)
    gen.add_argument("--force", action="store_true",
                     help="overwrite an existing questions.json")

    run = sub.add_parser("run", help="score the pipeline against the question set")
    run.add_argument("--save-baseline", action="store_true",
                     help="save this run as the comparison baseline")

    args = parser.parse_args()
    if args.command == "generate":
        generate_questions(args.n, args.force)
    else:
        run_eval(args.save_baseline)


if __name__ == "__main__":
    main()
