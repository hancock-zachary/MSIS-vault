import re
import subprocess

REWRITE_PROMPT = """\
You are a query rewriting assistant for a student's academic knowledge base.

Given the user's question, produce exactly 4 rewritten variants as a numbered list.
Apply these strategies:
1. Decompose multi-part questions into the most important sub-question
2. Expand acronyms and abbreviations
3. Add domain synonyms (e.g. "ERD" → "entity relationship diagram")
4. Write a short hypothetical answer fragment (HyDE) to improve retrieval

Output ONLY the 4 numbered variants, nothing else.

User question: {question}
"""


def rewrite_query(question: str) -> list[str]:
    prompt = REWRITE_PROMPT.format(question=question)
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return [question]
    if result.returncode != 0 or not result.stdout.strip():
        return [question]  # fallback: use original
    variants = parse_variants(result.stdout)
    # always include original to guarantee coverage
    if question not in variants:
        variants.insert(0, question)
    return variants


def parse_variants(raw: str) -> list[str]:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    cleaned = []
    for line in lines:
        # strip leading "1." or "-" or "*"
        line = re.sub(r"^(\d+\.\s+|[-*]\s+)", "", line)
        if line:
            cleaned.append(line)
    return cleaned
