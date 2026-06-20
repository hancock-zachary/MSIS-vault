"""Diagnostic: which eval gold snippets no longer exist in any indexed chunk?

Gold snippets are cut from the chunks of the index that existed at generation
time. Re-chunking moves boundaries, and a snippet that straddled an old chunk
seam may exist in no current chunk — those questions become unwinnable at
passage level and drag passage metrics down as a measurement artifact, not a
retrieval regression.

Usage: uv run python scripts/audit_eval_snippets.py
"""
import json

from src.config import EVAL_QUESTIONS_PATH
from src.index import get_collection


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    payload = json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = payload["questions"]
    docs = [norm(d) for d in get_collection().get(include=["documents"])["documents"]]

    orphaned = [
        q for q in questions
        if not any(norm(q["gold_snippet"]) in d for d in docs)
    ]

    print(f"{len(questions)} questions, {len(orphaned)} orphaned gold snippets\n")
    for q in orphaned:
        print(f"  {q['id']}  {q['gold_filename']} p{q['gold_page']}")
        print(f"        {q['question'][:90]}")


if __name__ == "__main__":
    main()
