"""Embedding backends. Default: OpenAI (no local model download, fast to deploy)."""
from __future__ import annotations

from . import config


class OpenAIEmbedder:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()

    def _embed(self, texts):
        out = []
        for i in range(0, len(texts), 96):
            res = self.client.embeddings.create(model=config.EMBED_MODEL, input=texts[i : i + 96])
            out += [d.embedding for d in sorted(res.data, key=lambda d: d.index)]
        return out

    def embed_passages(self, texts):
        return self._embed(texts)

    def embed_queries(self, texts):
        return self._embed(texts)


class LocalEmbedder:
    """sentence-transformers (e.g. multilingual-e5). Needs requirements-local.txt."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(config.EMBED_MODEL)

    def _enc(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_passages(self, texts):
        return self._enc(["passage: " + t for t in texts])

    def embed_queries(self, texts):
        return self._enc(["query: " + t for t in texts])


def get_embedder():
    return OpenAIEmbedder() if config.EMBED_PROVIDER == "openai" else LocalEmbedder()
