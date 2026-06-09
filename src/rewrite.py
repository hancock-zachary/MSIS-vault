import re
import subprocess

REWRITE_PROMPT = """\
You are a query rewriting assistant for a graduate student's academic knowledge base.
The knowledge base contains lecture slides, transcripts, assigned readings, and study
materials across multiple business school courses.

Generate exactly 5 retrieval-optimized variants of the student's question. Each variant
serves a specific purpose in a hybrid dense + BM25 retrieval system.

1. KEYWORDS — Extract key technical terms, acronyms in both forms (e.g. "WBS work breakdown
   structure"), and proper nouns. Write as a compact phrase, not a sentence. Optimized for
   BM25 keyword matching.

2. PARAPHRASE — Restate the question using different vocabulary and synonyms while preserving
   full meaning. Optimized for semantic/dense retrieval.

3. HYDE — Write a 2-3 sentence hypothetical answer as it would appear in a lecture slide or
   textbook. Do not answer the question yourself — write what a correct source passage would
   look like. This is the most important variant for dense retrieval.

4. SPECIFIC — If the question has multiple parts, isolate the single most critical
   sub-question. If already focused, write a narrower, more precise version.

5. BROADER — Rewrite as a search for the general topic this question belongs to. Useful for
   finding definitions, overviews, and introductory passages.

Output ONLY the 5 numbered variants, one per line. No labels, no explanations.

Student question: {question}
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
