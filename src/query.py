"""
Usage: python src/query.py "Your question here" [--course "IS 6410"]

Runs the full pipeline: rewrite → hybrid retrieve → rerank → format context.
Output is a context block for Claude to consume. Does NOT call Claude itself.

Retrieval is scoped to one course when --course is given, or when the
question mentions exactly one known course (auto-detected). Questions
mentioning several courses are never scoped — cross-course synthesis
needs candidates from all of them.
"""
import argparse
import sys
from src.rewrite import rewrite_query
from src.retrieve import expand_neighbors, hybrid_retrieve
from src.rerank import rerank_chunks
from src.config import RAW_DIR, TOP_K_RERANK


def detect_course(question: str, known_courses: list[str]) -> str | None:
    """Return the course iff exactly one known course is mentioned."""
    q = question.lower()
    mentioned = [c for c in known_courses if c.lower() in q]
    return mentioned[0] if len(mentioned) == 1 else None


def _known_courses() -> list[str]:
    """Course names are the first-level subfolders of raw/."""
    if not RAW_DIR.exists():
        return []
    return sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())


def format_context(question: str, chunks: list[dict]) -> str:
    """Format reranked chunks as a context block for Claude.

    Slide chunks carry a page range (page_start..page_end); showing the full
    range lets Claude cite any page in it, which verify.py will accept.
    """
    lines = [f"# Retrieved context for: {question}\n"]
    for i, chunk in enumerate(chunks, 1):
        start = chunk.get("page_start", chunk.get("page", "?"))
        end = chunk.get("page_end", chunk.get("page", "?"))
        page_label = f"pages {start}-{end}" if start != end else f"page {start}"
        course_label = chunk.get("course", "?")
        source_type = chunk.get("source_type", "")
        if source_type and source_type != "unknown":
            course_label += f" · {source_type}"
        lines.append(
            f"## [{i}] {chunk.get('filename', 'unknown')}, {page_label} "
            f"(course: {course_label})\n"
            f"{chunk.get('text', '')}\n"
        )
    return "\n".join(lines)


def run_query(question: str, course: str | None = None) -> str:
    if course is None:
        course = detect_course(question, _known_courses())
        if course:
            print(f"[src] Course filter auto-detected: {course}", file=sys.stderr)

    print(f"[src] Rewriting query...", file=sys.stderr)
    variants = rewrite_query(question)
    print(f"[src] {len(variants)} variants generated.", file=sys.stderr)

    print(f"[src] Running hybrid retrieval...", file=sys.stderr)
    candidates = hybrid_retrieve(variants, course=course)
    print(f"[src] {len(candidates)} candidates before reranking.", file=sys.stderr)

    if not candidates:
        return f"# Retrieved context for: {question}\n\n(no results found)\n"

    print(f"[src] Reranking...", file=sys.stderr)
    top_chunks = rerank_chunks(question, candidates, top_k=TOP_K_RERANK)
    print(f"[src] {len(top_chunks)} chunks after reranking.", file=sys.stderr)

    print(f"[src] Expanding neighbor chunks...", file=sys.stderr)
    top_chunks = expand_neighbors(top_chunks)

    return format_context(question, top_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the second brain.")
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument("--course", default=None,
                        help='restrict retrieval to one course, e.g. "IS 6410"')
    args = parser.parse_args()
    question = " ".join(args.question)
    sys.stdout.buffer.write((run_query(question, course=args.course) + "\n").encode("utf-8"))
