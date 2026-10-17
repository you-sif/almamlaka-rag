# Almamlaka TV | Digital Expansion Initiative Assistant

A retrieval-augmented chatbot that answers questions **only** from three project PDFs
(Project Overview, Budget & Timeline, Team & Governance). It cites the document, page and
section for every answer, flags conflicts between documents, handles follow-up questions,
replies in Arabic or English, and refuses prompt-injection attempts.

## How it works

```
question ─► injection pre-check ─► language detect ─► follow-up rewrite (small LLM)
        ─► retrieval (OpenAI embeddings + Chroma, per-document coverage)
        ─► grounded answer (GPT-4.1, JSON output) ─► citation validation ─► UI
```

| Piece | Choice | Why |
|---|---|---|
| UI / hosting | Streamlit, Docker Space on Hugging Face | Fast to brand, free hosting. No local model, so the image stays small |
| Extraction | PyMuPDF; tables rendered row-by-row (`Q2 2027 — Development: $500,000; ...`) | Keeps each table row self-describing and retrievable |
| Chunking | Split at numbered headings, then pack to ~1,100 chars (about 250-300 tokens) with 150-char overlap for long sections. Each chunk stores document, section and page range and is embedded with a header (`<document> \| §3 Key Details`). Repeated chunks are dropped. | The documents are short and sectioned. Section-aligned chunks never mix topics, and the header keeps tiny chunks (e.g. "Project Lead: ...") findable |
| Embeddings | OpenAI `text-embedding-3-large` (cosine). A local `multilingual-e5-base` backend is included (`EMBED_PROVIDER=local`) | Strong Arabic and English in one vector space, so an Arabic question retrieves English passages. Cost is negligible for 16 chunks |
| Vector DB | ChromaDB (persistent, cosine HNSW). The index is rebuilt automatically when the PDFs or embedding model change | Simple, embedded, supports metadata filters |
| LLM | OpenAI `gpt-4.1` for answers, `gpt-4.1-mini` for query rewriting (override with `ANSWER_MODEL` / `REWRITE_MODEL`). Claude is also supported via `LLM_PROVIDER=anthropic` | Reliable instruction-following and JSON output; cheap per question |

## Requirement by requirement

- **Answers only from the PDFs.** The answer prompt allows only facts stated in the supplied
  excerpts, forbids extending lists beyond what the text names, and returns a structured
  `status`. `not_found` produces a fixed reply ("I don't have that information...") in the
  user's language; the model never writes its own refusal. An answer with no valid citation
  is downgraded to "not found".
- **Citations.** The model marks claims with `[S1]`-style ids internally. The app removes them from
  the answer text and lists each source (document, page, section, excerpt) in an always-open Sources
  panel built from chunk metadata. Ids that don't exist are dropped, so a citation can never be
  invented, and an answer with no valid source is replaced by "not found".
- **Conflicts.** Retrieval always includes the top chunks from *each* document, so every document
  is represented. The prompt requires listing each value with its own citation and saying the
  documents disagree, including any revision date the sources state. The UI adds a warning banner.
- **Multi-turn.** Before retrieval, the small model rewrites the latest message into a standalone question
  using the last 8 messages ("What about its budget?" becomes a full question).
- **Language.** Detected from the user's message (Arabic vs. Latin letters); the answer prompt and
  fixed messages follow it. Arabic answers render right-to-left. Arabic questions are also searched
  in English.
- **Prompt injection.** Four layers: (1) regex pre-check in English and Arabic (diacritic-normalised)
  refuses clear override / prompt-leak attempts before any model call; (2) user text and PDF text
  go into delimited data blocks with `<` `>` neutralised so they can't close the delimiters;
  (3) the system prompt treats all of it as untrusted data and defines a `refused` status for
  anything that isn't a document question; (4) refusals are fixed strings.
- **Branding.** Put the official logo in `assets/logo.png`. It is shown in the header on a white
  plate (so any logo stays readable), used as the favicon, and its dominant colour becomes the accent
  and tints the dark background (`rag/branding.py`). Without a logo the app falls back to a
  placeholder wordmark and palette from `rag/config.py`.
- **Rendering detail.** Streamlit treats `$...$` as LaTeX, which would swallow dollar signs in
  budget figures, so all text is escaped before display.

## Design notes and limitations

- Embedding similarity scores are compressed (unrelated queries can still score high), so a
  similarity cut-off would be unreliable. Out-of-scope detection is done by the answer
  step instead, backed by the citation check.
- Scanned (image-only) PDFs need OCR first; ingestion stops with a clear error.
- The regex pre-check is a first filter, not the whole defence; paraphrased attacks are caught by
  the prompt-level rules.
- Conflict detection depends on both versions being retrieved. Per-document retrieval makes that
  likely for a three-document corpus.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# copy your three PDFs into data/
export OPENAI_API_KEY=sk-...
python -m rag.ingest --chunks-only     # inspect chunks (optional)
streamlit run app.py
python -m unittest discover -s tests -t .   # offline tests
```

## Deploy (Streamlit Community Cloud, free)

1. Push this repo to GitHub (PDFs included in `data/`; never commit keys).
2. On share.streamlit.io choose **Create app**, pick the repo, branch `main`, main file `app.py`.
3. Advanced settings > Secrets: `OPENAI_API_KEY = "sk-..."`. Python 3.11 or 3.12.
4. Deploy. The index (about 16 chunks) is built on first start, which takes a few seconds.
5. Cost control: the link is public and the key is billed per use. Keep the prepaid balance small,
   turn off auto-recharge, and note the app caps each browser session (`MAX_QUESTIONS_PER_SESSION`,
   default 40). Free apps hibernate after inactivity; opening the link wakes them.

(The included `Dockerfile` also works on any container host.)

`EVAL.md` lists the test questions used for the demo screenshots.
