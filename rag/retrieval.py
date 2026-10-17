"""Vector retrieval over Chroma.

For every query we take (a) the global top-K chunks and (b) the top PER_DOC_K chunks of EACH
document. (b) guarantees that every document gets a voice, which is what lets the answer step
notice when two documents disagree instead of seeing only the best-matching one.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, ingest


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float  # cosine similarity


class Retriever:
    def __init__(self):
        import chromadb
        from chromadb.config import Settings
        from .embeddings import get_embedder

        config.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.embedder = get_embedder()
        self.client = chromadb.PersistentClient(
            path=str(config.DB_DIR), settings=Settings(anonymized_telemetry=False)
        )
        self.col = ingest.ensure_index(self.embedder, self.client)
        metas = self.col.get(include=["metadatas"])["metadatas"]
        self.doc_ids = sorted({m["doc_id"] for m in metas})
        self.count = self.col.count()

    def _embed(self, queries):
        return self.embedder.embed_queries(queries)

    def retrieve(self, queries: list[str]) -> list[Hit]:
        hits: dict[str, Hit] = {}

        def collect(res):
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                score = 1.0 - float(dist)
                if id_ not in hits or score > hits[id_].score:
                    hits[id_] = Hit(id_, doc, meta, score)

        for emb in self._embed(queries):
            collect(self.col.query(query_embeddings=[emb], n_results=min(config.TOP_K, self.count)))
            for doc_id in self.doc_ids:
                try:
                    collect(
                        self.col.query(
                            query_embeddings=[emb],
                            n_results=config.PER_DOC_K,
                            where={"doc_id": doc_id},
                        )
                    )
                except Exception:
                    continue
        ranked = sorted(hits.values(), key=lambda h: h.score, reverse=True)
        return ranked[: config.MAX_CONTEXT_CHUNKS]
