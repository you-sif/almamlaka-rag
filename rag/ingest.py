"""Extract text from the PDFs in data/, chunk it and build the Chroma index.

    python -m rag.ingest                 # build (or rebuild if the PDFs changed)
    python -m rag.ingest --chunks-only   # just print the chunks (no models needed)
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

from . import config
from .chunking import Chunk, build_chunks, dedupe_chunks

SUPPORTED = {".pdf", ".txt", ".md"}


def list_sources(data_dir=None):
    d = Path(data_dir or config.DATA_DIR)
    return sorted(p for p in d.glob("*") if p.suffix.lower() in SUPPORTED)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "doc"


# ------------------------------------------------------------------ extraction

def _table_to_text(rows) -> str:
    """Turn a table into self-describing lines so each row can be retrieved on its own."""
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return "\n".join(" | ".join(r) for r in rows)
    header = rows[0]
    out = ["Table (columns: " + " | ".join(header) + ")"]
    for r in rows[1:]:
        cells = "; ".join(f"{h}: {v}" for h, v in zip(header[1:], r[1:]) if v)
        out.append(f"{r[0]} \u2014 {cells}" if cells else r[0])
    return "\n".join(out)


def _pdf_page_text(page) -> str:
    import fitz  # PyMuPDF

    items, table_rects = [], []
    try:
        for t in page.find_tables().tables:
            rows = t.extract()
            if rows:
                table_rects.append(fitz.Rect(t.bbox))
                items.append((t.bbox[1], 0.0, _table_to_text(rows)))
    except Exception:  # older PyMuPDF or odd layout: fall back to plain text
        table_rects = []
        items = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, _no, btype = b[:7]
        if btype != 0 or not text.strip():
            continue
        centre = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(r.contains(centre) for r in table_rects):
            continue  # already captured by the table renderer
        items.append((y0, x0, text.strip()))
    items.sort(key=lambda i: (round(i[0]), i[1]))
    return "\n".join(i[2] for i in items)


def extract_pages(path: Path):
    """Return [(page_number, text), ...]."""
    if path.suffix.lower() != ".pdf":
        return [(1, path.read_text(encoding="utf-8"))]
    import fitz

    with fitz.open(path) as doc:
        pages = [(i, _pdf_page_text(p)) for i, p in enumerate(doc, start=1)]
    if not any(t.strip() for _, t in pages):
        raise RuntimeError(f"{path.name}: no extractable text (scanned PDF?). OCR it first.")
    return pages


def load_chunks(data_dir=None) -> list[Chunk]:
    files = list_sources(data_dir)
    if not files:
        raise FileNotFoundError(f"No PDFs found in {data_dir or config.DATA_DIR}. Put the 3 PDFs there.")
    chunks: list[Chunk] = []
    for f in files:
        chunks += build_chunks(slugify(f.stem), f.name, extract_pages(f), fallback_title=f.stem)
    return dedupe_chunks(chunks)


# ------------------------------------------------------------------ indexing

def corpus_signature(data_dir=None) -> str:
    h = hashlib.sha1()
    for f in list_sources(data_dir):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    for part in (config.EMBED_ID, str(config.CHUNK_MAX_CHARS), str(config.CHUNK_OVERLAP_CHARS), "v1"):
        h.update(part.encode())
    return h.hexdigest()


def ensure_index(embedder, client):
    """Return a Chroma collection that matches the current PDFs, rebuilding it if needed."""
    sig_file = config.DB_DIR / "signature.txt"
    sig = corpus_signature()
    col = None
    try:
        col = client.get_collection(config.COLLECTION)
    except Exception:
        pass
    if col is not None and col.count() > 0 and sig_file.exists() and sig_file.read_text().strip() == sig:
        return col
    if col is not None:
        client.delete_collection(config.COLLECTION)

    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    chunks = load_chunks()
    embeddings = embedder.embed_passages([c.embed_text for c in chunks])
    col.add(
        ids=[f"{c.doc_id}-{c.idx:02d}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "doc_short": c.doc_short,
                "file": c.file,
                "section_no": c.section_no,
                "section_title": c.section_title,
                "page_start": c.page_start,
                "page_end": c.page_end,
            }
            for c in chunks
        ],
    )
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    sig_file.write_text(sig)
    return col


def _print_chunks(chunks):
    for c in chunks:
        print(f"[{c.doc_id}-{c.idx:02d}] {c.doc_short} | p.{c.page_start}-{c.page_end} | {c.section_label} | {len(c.text)} chars")
        print("    " + c.text.replace("\n", "\n    "))


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    chunks = load_chunks()
    if "--chunks-only" in argv:
        _print_chunks(chunks)
        print(f"\n{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents")
        return
    import chromadb
    from chromadb.config import Settings
    from .embeddings import get_embedder

    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.DB_DIR), settings=Settings(anonymized_telemetry=False))
    col = ensure_index(get_embedder(), client)
    print(f"Index ready: {col.count()} chunks from {len({c.doc_id for c in chunks})} documents")


if __name__ == "__main__":
    main()
