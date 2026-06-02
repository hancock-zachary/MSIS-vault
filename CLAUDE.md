# Second Brain — Claude Code Instructions

## How to answer study questions

When the user asks any factual question, you MUST:

1. Run the retrieval pipeline first:
   ```
   python brain/query.py "<user's question>"
   ```
2. Use ONLY the returned context chunks as your factual basis.
3. Tag every factual claim with an inline citation: `[source: filename, page N]`
4. If retrieval returns no results, respond: "I don't have slides covering this topic."
5. Never assert facts not present in the retrieved chunks.

## Citation rules

- Every sentence containing a factual claim needs `[source: filename, page N]`
- Synthesis across multiple chunks is allowed — cite all sources used
- Your own explanations and analogies are fine — label them clearly as "explanation:" not as facts

## Adding new slides

When the user adds new PDFs to the vault:
```
python brain/ingest.py
```

## Running tests

```
pytest brain/tests/ -v
```
