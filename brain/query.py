"""
Usage: python brain/query.py "Your question here"

Runs the full pipeline: rewrite → hybrid retrieve → rerank → format context.
Output is a context block for Claude to consume. Does NOT call Claude itself.
"""
import sys
from brain.rewrite import rewrite_query
from brain.retrieve import hybrid_retrieve
from brain.rerank import rerank_chunks
from brain.config import TOP_K_RERANK


def run_query(question: str) -> str:
    print(f"[brain] Rewriting query...", file=sys.stderr)
    variants = rewrite_query(question)
    print(f"[brain] {len(variants)} variants generated.", file=sys.stderr)

    print(f"[brain] Running hybrid retrieval...", file=sys.stderr)
    candidates = hybrid_retrieve(variants)
    print(f"[brain] {len(candidates)} candidates before reranking.", file=sys.stderr)

    if not candidates:
        return f"# Retrieved context for: {question}\n\n(no results found)\n"

    print(f"[brain] Reranking...", file=sys.stderr)
    top_chunks = rerank_chunks(question, candidates, top_k=TOP_K_RERANK)
    print(f"[brain] {len(top_chunks)} chunks after reranking.", file=sys.stderr)

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
        print("Usage: python brain/query.py \"Your question\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(run_query(question))
