"""
Hybrid Retrieval Engine (Dense + BM25) compliant with FR-QA-02 and FR-FIG-04.
Indexes both text chunks and figure summaries with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.schemas.domain import Chunk, PaperDoc
from backend.app.retrieval.embedder import LocalEmbedder, local_embedder


class SearchResult:
    def __init__(
        self,
        item_id: str,
        item_type: str,  # 'chunk' or 'figure'
        score: float,
        text: str,
        metadata: Dict[str, Any],
    ):
        self.item_id = item_id
        self.item_type = item_type
        self.score = score
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "score": round(self.score, 4),
            "text": self.text,
            "metadata": self.metadata,
        }


class HybridSearchIndex:
    """
    Maintains dense vector index and BM25 index for a single paper.
    Supports hybrid queries across sections and figure captions.
    """

    def __init__(
        self,
        paper: PaperDoc,
        chunks: List[Chunk],
        embedder: Optional[LocalEmbedder] = None,
    ):
        self.paper = paper
        self.chunks = chunks
        self.embedder = embedder or local_embedder

        # Build corpus
        self.items: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.dense_matrix: Optional[np.ndarray] = None

        self._build_index()

    def _build_index(self):
        vectors = []

        # Index text chunks
        for chunk in self.chunks:
            tokens = chunk.text.lower().split()
            self.tokenized_corpus.append(tokens)
            self.items.append(
                {
                    "id": chunk.chunk_id,
                    "type": "text",
                    "text": chunk.text,
                    "block_ids": chunk.block_ids,
                    "section": " > ".join(chunk.section_path),
                    "page": chunk.page_start,
                    "raw_chunk": chunk,
                    "grobid_paper_id": chunk.grobid_paper_id,
                    "paper_metadata": chunk.paper_metadata,
                    "linked_references": chunk.linked_references,
                }
            )
            vec = chunk.embedding or self.embedder.embed_query_sync(chunk.text)
            vectors.append(vec)

        # Index figures (FR-FIG-04)
        for fig in self.paper.figures:
            fig_text = f"Figure {fig.figure_id}: {fig.caption}"
            if fig.summary and isinstance(fig.summary, dict):
                fig_text += " " + json.dumps(fig.summary)
            tokens = fig_text.lower().split()
            self.tokenized_corpus.append(tokens)
            self.items.append(
                {
                    "id": fig.figure_id,
                    "type": "figure",
                    "text": fig_text,
                    "figure_id": fig.figure_id,
                    "page": fig.page,
                    "bbox": fig.bbox,
                    "crop_path": fig.crop_path,
                    "raw_figure": fig,
                }
            )
            vec = fig.embedding or self.embedder.embed_query_sync(fig_text)
            vectors.append(vec)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

        if vectors:
            self.dense_matrix = np.array(vectors, dtype=np.float32)
        else:
            self.dense_matrix = np.zeros((0, self.embedder.dimension), dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> List[SearchResult]:
        if not self.items:
            return []

        n_items = len(self.items)

        # 1. BM25 Scores
        query_tokens = query.lower().split()
        if self.bm25 and query_tokens:
            bm25_scores = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)
            # Min-max normalize BM25 scores
            if bm25_scores.max() > bm25_scores.min():
                bm25_norm = (bm25_scores - bm25_scores.min()) / (
                    bm25_scores.max() - bm25_scores.min()
                )
            else:
                bm25_norm = np.zeros(n_items, dtype=np.float32)
        else:
            bm25_norm = np.zeros(n_items, dtype=np.float32)

        # 2. Dense Cosine Similarity Scores
        query_vec = np.array(self.embedder.embed_query_sync(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        matrix_norms = np.linalg.norm(self.dense_matrix, axis=1, keepdims=True)
        matrix_norms[matrix_norms == 0] = 1.0
        normalized_matrix = self.dense_matrix / matrix_norms

        dense_scores = np.dot(normalized_matrix, query_vec)
        # Shift cosine from [-1, 1] to [0, 1]
        dense_norm = (dense_scores + 1.0) / 2.0

        # 3. Hybrid Combination
        combined_scores = (dense_weight * dense_norm) + (bm25_weight * bm25_norm)

        # Top-K ranking
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(combined_scores[idx])
            item = self.items[idx]
            results.append(
                SearchResult(
                    item_id=item["id"],
                    item_type=item["type"],
                    score=score,
                    text=item["text"],
                    metadata=item,
                )
            )

        return results
