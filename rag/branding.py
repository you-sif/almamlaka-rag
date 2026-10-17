"""Brand helpers: find the logo and derive the colour palette from it.

Drop the official logo into assets/ as logo.png (or .jpg/.jpeg/.webp). The accent colour is taken
from the logo's dominant saturated colour, and the dark background is tinted with the same hue,
so the interface matches the brand without hard-coding hex values.
"""
from __future__ import annotations

import base64
import colorsys
from pathlib import Path

from . import config

ASSETS = Path(__file__).resolve().parent.parent / "assets"
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def find_logo():
    for ext in _MIME:
        p = ASSETS / f"logo{ext}"
        if p.exists():
            return p
    return None


def logo_data_uri(path: Path) -> str:
    return f"data:{_MIME[path.suffix.lower()]};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


def accent_from_logo(path: Path):
    """Dominant saturated colour of the logo as (r, g, b), or None (e.g. black/white/grey logos)."""
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    im.thumbnail((96, 96))
    buckets: dict[int, list] = {}
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r, g, b, a = px[x, y]
            if a < 200:
                continue
            h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            if s < 0.35 or l < 0.18 or l > 0.85:
                continue
            acc = buckets.setdefault(round(h * 36) % 36, [0, 0, 0, 0])
            acc[0] += 1
            acc[1] += r
            acc[2] += g
            acc[3] += b
    if not buckets:
        return None
    n, r, g, b = max(buckets.values(), key=lambda v: v[0])
    return r // n, g // n, b // n


def get_palette(logo: Path | None = None) -> dict:
    """config.BRAND, with colours re-derived from the logo when one is available."""
    pal = dict(config.BRAND)
    if logo is None:
        return pal
    try:
        rgb = accent_from_logo(logo)
    except Exception:
        rgb = None
    if not rgb:
        return pal
    h, l, s = colorsys.rgb_to_hls(*(c / 255 for c in rgb))
    pal["gold"] = _hex(h, min(max(l, 0.5), 0.68), max(s, 0.55))  # readable on a dark background
    pal["bg"] = _hex(h, 0.08, 0.45)
    pal["panel"] = _hex(h, 0.14, 0.45)
    if h < 0.04 or h > 0.96:  # accent is red: keep the conflict colour distinguishable
        pal["alert"] = "#F0A030"
    return pal
