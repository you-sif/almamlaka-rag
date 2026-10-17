"""End-to-end question answering with guardrails.

Flow: injection pre-check -> language -> follow-up rewrite -> retrieval -> grounded answer
-> citation validation. Citations shown to the user are built from chunk metadata, never from
text the model typed, and an answer without a valid citation is downgraded to "not found".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import config
from .textutils import detect_language, injection_level

log = logging.getLogger(__name__)

_RAW_MARK = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]")
_MARK = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")


@dataclass
class Answer:
    status: str  # answered | not_found | refused | error
    text: str
    lang: str
    conflict: bool = False
    sources: list = field(default_factory=list)  # [{"label", "file", "excerpt"}]
    standalone: str = ""


def source_label(meta: dict) -> str:
    p0, p1 = meta["page_start"], meta["page_end"]
    pages = f"p. {p0}" if p0 == p1 else f"pp. {p0}\u2013{p1}"
    sec = meta.get("section_no") or 0
    section = f"\u00a7{sec} {meta['section_title']}" if sec else meta["section_title"]
    return f"{meta['doc_short']}, {pages}, {section}"


def render_citations(answer: str, labels: dict):
    """Remove [S#] markers from the answer text (citations live in the Sources panel only).

    Returns (clean_text, used_ids). Ids the model invented are ignored, so they can never be shown.
    """
    used: list[str] = []

    def repl(m):
        for sid in (x.strip() for x in re.split(r"[,;]", m.group(1))):
            if sid in labels and sid not in used:
                used.append(sid)
        return ""

    text = _MARK.sub(repl, answer)
    text = re.sub(r"[ \t]+([.,;:!?\u061f\u060c])", r"\1", text)  # no gap left before punctuation
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), used


def _answer_lang(text: str) -> str:
    return detect_language(_RAW_MARK.sub("", text))


def _fixed(status: str, lang: str) -> Answer:
    return Answer(status, config.MESSAGES[status][lang], lang)


def answer_question(question: str, history: list, retriever, llm) -> Answer:
    question = (question or "").strip()[: config.MAX_QUESTION_CHARS]
    lang = detect_language(question)
    level = injection_level(question)
    if level == 2:
        return _fixed("refused", lang)
    try:
        standalone = english = question
        if history or lang == "ar":
            r = llm.rewrite(history[-config.HISTORY_MESSAGES :], question)
            standalone = r.get("standalone") or question
            english = r.get("english") or standalone
        queries = list(dict.fromkeys(q for q in (standalone, english) if q))
        hits = retriever.retrieve(queries)
        if not hits:
            return _fixed("not_found", lang)
        sources = {f"S{i}": h for i, h in enumerate(hits, 1)}
        # Ask in the user's own language: the English query when they wrote English, so an Arabic
        # history can never pull the answer into Arabic (and vice versa).
        q_for_answer = english if lang == "en" else standalone
        data = llm.answer(q_for_answer, lang, sources, suspicious=(level == 1))
        for _ in range(2):  # enforce the reply language: retry (strictly) if it came back wrong
            raw0 = str((data or {}).get("answer") or "").strip()
            if (data or {}).get("status") != "answered" or not raw0 or _answer_lang(raw0) == lang:
                break
            retry = llm.answer(q_for_answer, lang, sources, suspicious=(level == 1), strict_lang=True)
            if retry:
                data = retry
    except Exception:
        log.exception("answer pipeline failed")
        return _fixed("error", lang)

    if not data:
        return _fixed("error", lang)
    status = data.get("status")
    if status == "refused":
        return _fixed("refused", lang)
    raw = str(data.get("answer") or "").strip()
    if status != "answered" or not raw:
        return _fixed("not_found", lang)

    labels = {sid: source_label(h.meta) for sid, h in sources.items()}
    text, used = render_citations(raw, labels)
    if not used:  # the model cited nothing valid inline; fall back to its declared sources
        used = [s for s in data.get("sources_used", []) if s in labels]
    if not used:  # still nothing verifiable, so treat it as ungrounded
        return _fixed("not_found", lang)

    cites = [
        {"label": labels[s], "file": sources[s].meta["file"], "excerpt": sources[s].text} for s in used
    ]
    return Answer("answered", text, lang, bool(data.get("conflict")), cites, standalone)
