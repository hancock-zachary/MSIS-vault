"""
Usage: python src/query.py "Your question here"

Runs the full pipeline: rewrite → hybrid retrieve → rerank → format context.
Output is a context block for Claude to consume. Does NOT call Claude itself.
"""
import sys
from src.rewrite import rewrite_query
from src.retrieve import hybrid_retrieve
from src.rerank import rerank_chunks
from src.config import TOP_K_RERANK


def run_query(question: str) -> str:
    print(f"[src] Rewriting query...", file=sys.stderr)
    variants = rewrite_query(question)
    print(f"[src] {len(variants)} variants generated.", file=sys.stderr)

    print(f"[src] Running hybrid retrieval...", file=sys.stderr)
    candidates = hybrid_retrieve(variants)
    print(f"[src] {len(candidates)} candidates before reranking.", file=sys.stderr)

    if not candidates:
        return f"# Retrieved context for: {question}\n\n(no results found)\n"

    print(f"[src] Reranking...", file=sys.stderr)
    top_chunks = rerank_chunks(question, candidates, top_k=TOP_K_RERANK)
    print(f"[src] {len(top_chunks)} chunks after reranking.", file=sys.stderr)

    lines = [f"# Retrieved context for: {question}\n"]
    for i, chunk in enumerate(top_chunks, 1):
        lines.append(
            f"## [{i}] {chunk.get('filename', 'unknown')}, page {chunk.get('page', '?')} "
            f"(course: {chunk.get('course', '?')})\n"
            f"{chunk.get('text', '')}\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/query.py \"Your question\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    sys.stdout.buffer.write((run_query(question) + "\n").encode("utf-8"))
