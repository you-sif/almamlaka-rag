"""Dependency-free helpers: language detection, injection heuristics, JSON parsing."""
from __future__ import annotations

import json
import re

# ---------------------------------------------------------------- language

_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LAT_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Return 'ar' if Arabic letters dominate, else 'en'."""
    ar = len(_AR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    return "ar" if ar > 0 and ar >= lat else "en"


_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def normalize_ar(text: str) -> str:
    """Strip diacritics/tatweel and unify common letter variants for pattern matching."""
    text = _DIACRITICS_RE.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


# ---------------------------------------------------------------- injection

# STRONG: an attempt to override rules, reveal the prompt, or break out of our delimiters.
_STRONG = [
    r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the|these|those|system)\b.{0,30}\b(instructions?|prompts?|rules|guidelines|directions|constraints|restrictions)\b",
    r"\b(reveal|show|print|repeat|display|leak|output|tell me)\b.{0,30}\b(system prompt|your prompt|your instructions|initial instructions|hidden instructions|developer message)\b",
    r"\bsystem prompt\b",
    r"\b(jailbreak|dan mode|developer mode|do anything now)\b",
    r"\bnew instructions?\s*:",
    r"\byou are no longer\b",
    r"</?\s*(system|sources?|question|instructions?|assistant)\b",
]
# WEAK: phrases often used for role-hijacking; legitimate questions can contain them,
# so they only raise a flag that tightens the model prompt instead of blocking outright.
_WEAK = [
    r"\byou are now\b",
    r"\bfrom now on\b",
    r"\bpretend (to be|you are|you're)\b",
    r"\bact as (a|an|if)\b",
    r"\brole-?play\b",
    r"\bnew persona\b",
]
_STRONG_AR = [
    r"(تجاهل|تجاهلي|تجاهلوا|انس|انسي|اهمل|تخط)\w*.{0,30}(تعليمات|اوامر|قواعد|توجيهات|ارشادات|قيود)",
    r"(اكشف|اظهر|اعرض|اطبع|كرر|اخبرني)\w*.{0,30}(التعليمات|البرومبت|الموجه|رساله النظام|تعليمات النظام)",
    r"(برومبت|موجه)\s*(النظام)",
    r"(وضع المطور|كسر الحمايه)",
]
_WEAK_AR = [
    r"انت الان",
    r"تصرف ك",
    r"تظاهر",
    r"من الان فصاعدا",
    r"العب دور",
]

_STRONG_RE = [re.compile(p, re.I | re.S) for p in _STRONG]
_WEAK_RE = [re.compile(p, re.I | re.S) for p in _WEAK]
_STRONG_AR_RE = [re.compile(p, re.S) for p in _STRONG_AR]
_WEAK_AR_RE = [re.compile(p, re.S) for p in _WEAK_AR]


def injection_level(text: str) -> int:
    """0 = clean, 1 = suspicious (weak pattern), 2 = clear injection attempt."""
    plain = re.sub(r"\s+", " ", text)
    ar = normalize_ar(plain)
    if any(r.search(plain) for r in _STRONG_RE) or any(r.search(ar) for r in _STRONG_AR_RE):
        return 2
    if any(r.search(plain) for r in _WEAK_RE) or any(r.search(ar) for r in _WEAK_AR_RE):
        return 1
    return 0


def neutralize(text: str) -> str:
    """Stop user/document text from closing or forging our XML-style prompt delimiters."""
    return text.replace("<", "\uff1c").replace(">", "\uff1e")


# ---------------------------------------------------------------- JSON

def extract_json(text: str):
    """Parse the first JSON object in a model reply (tolerates code fences). None on failure."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def escape_md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX, which swallows currency signs: escape them."""
    return text.replace("$", "\\$")
