# Second Brain — Claude Code Instructions

This vault uses the **LLM Wiki pattern** (Karpathy, 2026): a persistent, compounding wiki
in `wiki/` sits between you and the raw sources. The wiki accumulates knowledge — summaries,
cross-references, and synthesis — so it gets richer with every source added and every question
asked. You (Claude) own the wiki layer. The human curates sources and asks questions.

---

## Wiki structure

```
raw/          ← source documents (PDFs, docs, txt). Immutable — never modify.
wiki/         ← LLM-maintained wiki pages (one .md per indexed document)
  index.md    ← content catalog of all pages; read this first when navigating
  log.md      ← append-only run log (ingests, queries, lint passes)
  <Course>/   ← pages grouped by course
src/          ← pipeline code (ingest, embed, graph, query)
```

Each wiki page has:
- **YAML frontmatter** — `course`, `chunks`, `updated`, `tags` (queryable via Dataview)
- **Key Excerpts** — actual text from the source's top chunks
- **Related Documents** — `[[wikilinks]]` to semantically related pages (mutual top-K)

---

## Answering study questions

**Always invoke the `query-second-brain` skill before answering any factual question.**
The user will typically prefix their question with `/query-second-brain`. If they do not,
invoke the skill yourself before proceeding.

When the user asks any factual question:

1. Read `wiki/index.md` to identify relevant pages.
2. Read the relevant wiki pages to gather context.
3. If the wiki pages don't fully cover the question, also run:
   ```
   python src/query.py "<question>"
   ```
   to pull additional chunks from the raw index.
4. Answer using only retrieved content. Tag every factual claim:
   `[source: filename, page N]`
5. If no relevant content exists: "I don't have citable information covering this topic."
6. **File valuable answers back into the wiki** — if the answer required non-trivial
   synthesis or revealed a connection not yet on any page, write it as a new wiki page
   or update an existing one. Explorations should compound in the wiki, not disappear
   into chat history.

### Citation rules
- Every factual sentence needs `[source: filename, page N]`
- Synthesis across multiple sources is allowed — cite all sources used
- Your own explanations and analogies are fine — label them `(explanation)`
- Chunks with `is_stub: true` in metadata indicate a topic exists in the source document but the content was unreadable (diagram, scanned image). Treat them as evidence of topic presence only — do not make factual claims from stub chunks and do not cite them as factual sources.

---

## Adding new sources

When the user adds new PDFs or documents to `raw/`:

```
uv run python src/ingest.py   # chunk, embed, and index the new files
uv run python src/graph.py    # rebuild wiki pages, index.md, and log.md
```

`graph.py` also cleans up stale pages (deleted source files) and detects orphans.

---

## Wiki maintenance (lint)

Periodically, or when asked to "lint the wiki", check for:
- **Contradictions** — claims on one page that conflict with another
- **Orphan pages** — pages with no inbound wikilinks (flagged in `wiki/log.md`)
- **Missing cross-references** — concepts mentioned on a page but lacking a `[[link]]`
  to another page that covers that concept
- **Stale claims** — content superseded by newer sources
- **Data gaps** — important topics with thin coverage; suggest new sources to find

After a lint pass, append a summary entry to `wiki/log.md`:
```
## [YYYY-MM-DD] lint | <one-line summary>
```

---

## Filing answers and analyses back into the wiki

When a query produces a useful synthesis (comparison, cross-course connection, worked
explanation), create a new wiki page for it:

1. Write the page to `wiki/<Course>/<Title>.md` (or `wiki/Synthesis/<Title>.md` for
   cross-course pages).
2. Add YAML frontmatter:
   ```yaml
   ---
   course: <course or "Synthesis">
   type: synthesis
   updated: <today>
   tags: [synthesis]
   ---
   ```
3. Add `[[wikilinks]]` to every source page referenced.
4. Update `wiki/index.md` to include the new page.
5. Append an entry to `wiki/log.md`:
   ```
   ## [YYYY-MM-DD] query | <question or topic>
   ```

---

## Running tests

```
uv run pytest src/tests/ -v
```

## Setup (first time)

```
uv sync --extra dev
```
