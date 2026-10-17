"""Structure-aware chunking. Pure Python, no third-party dependencies.

Strategy
1. Split each document at numbered headings ("3. Key Details") so a chunk never mixes topics.
2. Inside a section, pack lines up to CHUNK_MAX_CHARS with a small overlap (long sections only).
3. Every chunk remembers document, section and page range, which become the citations.
4. Every chunk is embedded with a context header ("<document> | §3 Key Details") so that
   short chunks such as "Project Lead: ..." still carry their topic.
5. Identical chunks inside one document (repeated pages) are dropped.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from . import config

HEADING_RE = re.compile(r"^(\d{1,2})[.)]\s+(\S.{1,90})$")
PAGE_NOISE_RE = re.compile(r"^(page\s*\d+(\s*(of|/)\s*\d+)?|\d+\s*/\s*\d+|-\s*\d+\s*-)$", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u061f])\s+")


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    doc_short: str
    file: str
    section_no: int  # 0 = not a numbered section
    section_title: str
    page_start: int
    page_end: int
    idx: int
    text: str

    @property
    def section_label(self) -> str:
        return f"\u00a7{self.section_no} {self.section_title}" if self.section_no else self.section_title

    @property
    def embed_text(self) -> str:
        return f"{self.doc_short} | {self.section_label}\n{self.text}"


# ------------------------------------------------------------------ helpers

def _is_heading(line: str):
    m = HEADING_RE.match(line)
    if not m:
        return None
    title = m.group(2).strip()
    if title.endswith((".", ",", ";", ":")):  # a numbered sentence, not a heading
        return None
    return int(m.group(1)), title


def detect_title(pages, fallback: str):
    """Return (full_title, short_title, lines_used) from the first lines of page 1."""
    if not pages:
        return fallback, fallback, []
    used = []
    for raw in pages[0][1].splitlines():
        s = raw.strip()
        if not s:
            continue
        if _is_heading(s) or len(s) > 100:
            break
        used.append(s)
        if len(used) == 2:
            break
    if not used:
        return fallback, fallback, []
    return " \u2014 ".join(used), used[-1], used


def strip_noise(pages):
    """Remove page-number lines and running headers/footers (lines on >= 60% of pages)."""
    cleaned = [
        (n, [l.strip() for l in t.splitlines() if l.strip() and not PAGE_NOISE_RE.match(l.strip())])
        for n, t in pages
    ]
    if len(cleaned) >= 3:
        counts = Counter(l for _, ls in cleaned for l in set(ls))
        limit = 0.6 * len(cleaned)
        cleaned = [(n, [l for l in ls if counts[l] < limit]) for n, ls in cleaned]
    return [(n, "\n".join(ls)) for n, ls in cleaned]


def _drop_leading(text: str, title_lines):
    """Remove title lines repeated at the very top of a later page (running header / pasted copy)."""
    if not title_lines:
        return text
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].strip() in title_lines:
        i += 1
    return "\n".join(lines[i:])


def split_sections(pages):
    sections = []
    cur = {"no": 0, "title": "Overview", "page": pages[0][0] if pages else 1, "lines": []}
    for page_no, text in pages:
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            h = _is_heading(s)
            if h:
                if cur["lines"]:
                    sections.append(cur)
                cur = {"no": h[0], "title": h[1], "page": page_no, "lines": []}
            else:
                cur["lines"].append((page_no, s))
    if cur["lines"]:
        sections.append(cur)
    return sections


def _split_long(line: str, max_chars: int):
    parts, buf = [], ""
    for sent in SENTENCE_SPLIT_RE.split(line):
        while len(sent) > max_chars:  # no sentence boundary: hard cut
            if buf:
                parts.append(buf)
                buf = ""
            parts.append(sent[:max_chars])
            sent = sent[max_chars:]
        if buf and len(buf) + len(sent) + 1 > max_chars:
            parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def pack_lines(lines, max_chars: int, overlap: int):
    """Greedy-pack (page, line) tuples into chunks with a trailing-line overlap."""
    items = []
    for page, line in lines:
        if len(line) <= max_chars:
            items.append((page, line))
        else:
            items.extend((page, p) for p in _split_long(line, max_chars))
    chunks, cur, size = [], [], 0
    for page, line in items:
        add = len(line) + 1
        if cur and size + add > max_chars:
            chunks.append(cur)
            carry, csize = [], 0
            for p, l in reversed(cur):
                if csize + len(l) + 1 > overlap:
                    break
                carry.insert(0, (p, l))
                csize += len(l) + 1
            cur, size = carry, csize
        cur.append((page, line))
        size += add
    if cur:
        chunks.append(cur)
    return [("\n".join(l for _, l in c), c[0][0], c[-1][0]) for c in chunks]


# ------------------------------------------------------------------ public API

def build_chunks(doc_id: str, file: str, pages, fallback_title: str | None = None):
    """pages: list of (page_number, text). Returns a list of Chunk."""
    title, short, title_lines = detect_title(pages, fallback_title or doc_id)
    pages = strip_noise(pages)
    pages = [pages[0]] + [(n, _drop_leading(t, title_lines)) for n, t in pages[1:]] if pages else pages
    sections = split_sections(pages)
    out = []
    for sec in sections:
        lines = sec["lines"]
        if sec["no"] == 0:  # drop the title lines from the preamble
            lines = [(p, l) for p, l in lines if l not in title_lines]
        if not lines:
            continue
        packed = pack_lines(lines, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS)
        for i, (text, p0, p1) in enumerate(packed):
            if i == 0:
                p0 = min(p0, sec["page"])
            out.append(
                Chunk(doc_id, title, short, file, sec["no"], sec["title"], p0, max(p0, p1), len(out), text)
            )
    return out


def dedupe_chunks(chunks):
    """Drop chunks whose text already appeared in the same document; re-index per document."""
    seen, out, counters = set(), [], Counter()
    for c in chunks:
        norm = re.sub(r"\s+", " ", c.text).strip().lower()
        key = (c.doc_id, hashlib.sha1(norm.encode("utf-8")).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        c.idx = counters[c.doc_id]
        counters[c.doc_id] += 1
        out.append(c)
    return out
