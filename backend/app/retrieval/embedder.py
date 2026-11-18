"""
Embedding Engine — supports two backends (ADR-4, INT-05):

  EMBEDDING_BACKEND=local    Fast, deterministic n-gram hashing; fully offline.
  EMBEDDING_BACKEND=gateway  Calls GATEWAY_BASE_URL/v1/embeddings with
                             EMBEDDING_API_MODEL (OpenAI-compatible endpoint).
                             Falls back to local hashing on any error when
                             EMBEDDING_FALLBACK_TO_LOCAL=true.

ADR-4 stability guarantee
─────────────────────────
Every embedding batch is tagged with the model identifier used.  If the model
recorded in a paper's existing chunk file differs from the currently configured
model, a warning is raised and the caller should trigger re-embedding.  This
prevents silent index corruption when provider routing changes the upstream
model.
"""

from __future__ import annotations

import asyncio
import hashlib
import warnings
from typing import List, Optional

import aiohttp
import numpy as np

from backend.app.core.config import settings
from backend.app.core.logging import logger


# ── Stability sentinel ──────────────────────────────────────────────────────


class EmbeddingModelChangedWarning(UserWarning):
    """Raised when the active embedding model differs from the one recorded in
    the vector store for a paper (ADR-4).  Caller should re-embed."""


# ── Gateway Embedder ────────────────────────────────────────────────────────


class Embedder:
    """
    Unified async embedding engine.

    Usage
    -----
    embedder = Embedder()

    # batch — returns List[List[float]]
    vecs = await embedder.embed_texts(["sentence one", "sentence two"])

    # single query — returns List[float]
    vec = await embedder.embed_query("what is attention?")

    # sync wrapper (blocks; only use from sync contexts / tests)
    vecs = embedder.embed_texts_sync(["sentence one"])
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        api_model: Optional[str] = None,
        dimension: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None,
        fallback_to_local: Optional[bool] = None,
    ):
        self.backend = (backend or settings.EMBEDDING_BACKEND).lower()
        self.api_model = api_model or settings.EMBEDDING_API_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.base_url = (base_url or settings.GATEWAY_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.GATEWAY_API_KEY
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.fallback = (
            fallback_to_local
            if fallback_to_local is not None
            else settings.EMBEDDING_FALLBACK_TO_LOCAL
        )

    # ── Public async interface ──────────────────────────────────────────────

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings.  Batches automatically to stay within gateway
        limits.  Falls back to deterministic hashing on failure (if configured).
        """
        if not texts:
            return []

        if self.backend == "gateway":
            return await self._embed_gateway(texts)
        return self._embed_local(texts)

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string and return its vector."""
        results = await self.embed_texts([query])
        return results[0]

    # ── Sync shim (for legacy callers & tests) ─────────────────────────────

    def embed_texts_sync(self, texts: List[str]) -> List[List[float]]:
        """Blocking wrapper around embed_texts for sync contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an event loop (e.g. called from a coroutine
                # via run_in_executor); create a new loop in a thread instead.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.embed_texts(texts))
                    return future.result()
            return loop.run_until_complete(self.embed_texts(texts))
        except RuntimeError:
            return asyncio.run(self.embed_texts(texts))

    def embed_query_sync(self, query: str) -> List[float]:
        return self.embed_texts_sync([query])[0]

    # ── ADR-4 stability check ───────────────────────────────────────────────

    def check_model_stability(self, recorded_model_id: str) -> bool:
        """
        Returns True if the currently configured model matches the one used to
        build an existing index.  Emits EmbeddingModelChangedWarning otherwise.
        """
        active = self._active_model_id()
        if recorded_model_id and recorded_model_id != active:
            warnings.warn(
                f"[ADR-4] Embedding model mismatch: index was built with "
                f"'{recorded_model_id}' but active model is '{active}'. "
                f"Re-embedding is required to avoid silent retrieval errors.",
                EmbeddingModelChangedWarning,
                stacklevel=2,
            )
            logger.warning(
                f"ADR-4 model mismatch: stored='{recorded_model_id}' active='{active}'"
            )
            return False
        return True

    def active_model_id(self) -> str:
        """Return the identifier that should be persisted alongside vectors."""
        return self._active_model_id()

    # ── Gateway implementation ──────────────────────────────────────────────

    async def _embed_gateway(self, texts: List[str]) -> List[List[float]]:
        """
        Calls POST /v1/embeddings in batches of EMBEDDING_BATCH_SIZE.
        On network / server errors:
          - If EMBEDDING_FALLBACK_TO_LOCAL is true  → logs warning, falls back.
          - Otherwise raises.
        """
        all_vectors: List[List[float]] = []
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            payload = {"model": self.api_model, "input": batch}

            try:
                timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT_SECONDS)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # OpenAI format: {"data": [{"embedding": [...], "index": N}]}
                            items = sorted(
                                data["data"], key=lambda x: x.get("index", 0)
                            )
                            batch_vecs = [item["embedding"] for item in items]

                            # Validate dimension on the first successful call
                            if batch_vecs and self.dimension != len(batch_vecs[0]):
                                actual_dim = len(batch_vecs[0])
                                logger.warning(
                                    f"[ADR-4] Gateway returned {actual_dim}-d vectors but "
                                    f"EMBEDDING_DIMENSION={self.dimension}. "
                                    f"Update EMBEDDING_DIMENSION in .env to avoid index errors."
                                )
                                self.dimension = actual_dim

                            all_vectors.extend(batch_vecs)
                            logger.info(
                                f"Embedded {len(batch)} texts via gateway "
                                f"({self.api_model}, {len(batch_vecs[0])}-d)"
                            )
                        else:
                            error_text = await resp.text()
                            raise RuntimeError(
                                f"Gateway /embeddings returned HTTP {resp.status}: "
                                f"{error_text[:200]}"
                            )

            except Exception as exc:
                logger.error(
                    f"Gateway embedding error (model={self.api_model}, "
                    f"batch={batch_start}–{batch_start + len(batch)}): {exc}"
                )
                if self.fallback:
                    logger.warning(
                        "EMBEDDING_FALLBACK_TO_LOCAL=true — falling back to "
                        "deterministic hashing for this batch."
                    )
                    all_vectors.extend(self._embed_local(batch))
                else:
                    raise

        return all_vectors

    # ── Local hashing fallback ──────────────────────────────────────────────

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """
        Deterministic n-gram hashing embeddings (ADR-4 fallback).
        Output is L2-normalised so cosine similarity is well-defined.
        These vectors are NOT semantically equivalent to gateway embeddings;
        they exist solely as an offline / test fallback.
        """
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> List[float]:
        dim = self.dimension
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return [0.0] * dim

        for i, word in enumerate(words):
            h1 = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
            vec[h1] += 1.0
            if i > 0:
                bigram = f"{words[i - 1]}_{word}"
                h2 = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16) % dim
                vec[h2] += 1.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _active_model_id(self) -> str:
        if self.backend == "gateway":
            return f"gateway:{self.api_model}"
        return "local:hash-ngram"


# Module-level singleton — updated whenever settings change.
embedder = Embedder()
# Keep old name for backward compatibility with any import of `local_embedder`
local_embedder = embedder
# Alias for backward compatibility with imports of `LocalEmbedder`
LocalEmbedder = Embedder
