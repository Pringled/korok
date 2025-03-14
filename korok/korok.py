from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any, DefaultDict

import bm25s
import numpy as np
from vicinity import Backend, Metric, Vicinity

from korok.datatypes import DenseResult, Document, HybridResult, QueryResult, SparseResult
from korok.rerankers import CrossEncoderReranker
from korok.utils import Encoder


class Pipeline:
    def __init__(
        self,
        encoder: Encoder | None = None,
        dense_index: Vicinity | None = None,
        sparse_index: bm25s.BM25 | None = None,
        reranker: CrossEncoderReranker | None = None,
        alpha: float = 0.5,
        corpus: list[str] | None = None,
    ) -> None:
        """
        Initialize a Pipeline instance.

        :param encoder: An encoder for dense vector search.
        :param dense_index: A dense vector index using the provided encoder.
        :param sparse_index: A sparse vector index using BM25.
        :param reranker: A cross-encoder reranker.
        :param alpha: The alpha value for hybrid search.
        :param corpus: List of documents used for BM25.
        """
        self.encoder = encoder
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.reranker = reranker
        self.alpha = alpha
        self.corpus = corpus

    @classmethod
    def fit(
        cls,
        texts: list[str],
        encoder: Encoder | None = None,
        bm25: bool = False,
        reranker: CrossEncoderReranker | None = None,
        backend_type: Backend = Backend.BASIC,
        distance_metric: Metric = Metric.COSINE,
        alpha: float = 0.5,
        stopwords: str | list[str] = "en",
        **kwargs: Any,
    ) -> Pipeline:
        """
        Fit a pipeline on a corpus of documents.

        - If an encoder is provided, build a dense vector index using the encoder.
        - If bm25 is True, build a sparse vector index using BM25.
        - If both are provided, build a hybrid index.
        - If a reranker is provided, rerank the results for each query.

        :param texts: The corpus of documents to index.
        :param encoder: An encoder for dense vector search.
        :param bm25: A bool indicating whether to build a BM25 index for sparse vector search.
        :param reranker: A cross-encoder reranker.
        :param backend_type: The backend type for the dense vector index.
        :param distance_metric: The distance metric for the dense vector index.
        :param alpha: The alpha value for hybrid search.
            Lower values give more weight to sparse search, higher values to dense search.
        :param stopwords: Stopwords for BM25 tokenization. Defaults to "en" (English stopwords).
        :param **kwargs: Additional args passed to Vicinity.from_vectors_and_items.
        :return: A Pipeline instance.
        :raises ValueError: If neither encoder nor bm25 is provided.
        :raises ValueError: If alpha is not between 0 and 1.
        """
        if encoder is None and bm25 is False:
            raise ValueError("At least one of encoder or bm25 must be provided.")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1")

        # Build a dense vector index using the encoder
        dense_index = None
        if encoder is not None:
            vectors = encoder.encode(texts, show_progressbar=True)
            dense_index = Vicinity.from_vectors_and_items(
                vectors=vectors,
                items=texts,
                backend_type=backend_type,
                metric=distance_metric,
                **kwargs,
            )

        # Build a sparse vector index using BM25
        sparse_index = None
        if bm25:
            sparse_index = bm25s.BM25()
            tokens = bm25s.tokenize(texts, stopwords=stopwords)
            sparse_index.index(tokens)

        return cls(
            encoder=encoder,
            dense_index=dense_index,
            sparse_index=sparse_index,
            reranker=reranker,
            alpha=alpha,
            corpus=texts,
        )

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize row-wise."""
        min_vals = np.min(scores, axis=-1, keepdims=True)
        max_vals = np.max(scores, axis=-1, keepdims=True)
        denom = max_vals - min_vals
        denom[denom == 0] = 1e-9
        return (scores - min_vals) / denom

    def _split_dense_results(self, results: DenseResult) -> tuple[list[list[str]], np.ndarray]:
        """Split the results from the dense index into docs and scores."""
        docs = [[doc for doc, _ in row] for row in results]
        scores = np.array([[float(score) for _, score in row] for row in results])
        return (docs, scores)

    def _merge_results(
        self,
        dense_results: DenseResult,
        sparse_results: SparseResult,
        k: int,
    ) -> HybridResult:
        """Merge dense and sparse results."""
        # Unpack
        dense_docs, dense_scores = self._split_dense_results(dense_results)
        sparse_docs, sparse_scores = sparse_results

        # Ensure docs are strings
        sparse_docs = [[str(doc) for doc in doclist] for doclist in sparse_docs]

        # Normalize scores
        dense_scores = self._normalize_scores(dense_scores)
        sparse_scores = self._normalize_scores(sparse_scores)

        results_out: HybridResult = []
        for i in range(len(dense_results)):
            doc_map: DefaultDict[str, Document] = defaultdict(Document)

            # Assign scores for each document
            for doc, score in zip(dense_docs[i], dense_scores[i]):
                doc_map[doc].text = doc
                doc_map[doc].dense_score = float(score)

            for doc, score in zip(sparse_docs[i], sparse_scores[i]):
                doc_map[doc].text = doc
                doc_map[doc].sparse_score = float(score)

            # Combine + partial sort
            combined = [(doc_id, d_obj.combine_scores(self.alpha)) for doc_id, d_obj in doc_map.items()]
            top_k = heapq.nlargest(k, combined, key=lambda x: x[1])
            results_out.append(top_k)

        return results_out

    def query(self, texts: list[str], k: int = 10) -> QueryResult:
        """
        Query the pipeline.

        This does the following:
          - Hybrid search if we have both dense & sparse indexes.
          - Dense search if we have only have a dense index.
          - Sparse search if we have only a sparse index.
          - Reranking if a reranker is provided.

        :param texts: The list of texts to query.
        :param k: The number of results to return.
        :return: The search results.
        """
        # Compute dense results if both dense index and encoder are available
        dense_results = None
        if self.dense_index is not None and self.encoder is not None:
            vectors = self.encoder.encode(texts, show_progressbar=True)
            dense_results = self.dense_index.query(vectors, k)

        # Compute sparse results if sparse index is available
        sparse_results = None
        if self.sparse_index is not None:
            tokens = bm25s.tokenize(texts, stopwords="en")
            sparse_results = self.sparse_index.retrieve(tokens, k=k, corpus=self.corpus, return_as="tuple")

        # Hybrid: merge dense and sparse results
        if dense_results is not None and sparse_results is not None:
            # Hybrid: merge dense and sparse results.
            results = self._merge_results(dense_results, sparse_results, k)
        # Dense only
        elif dense_results is not None:
            results = dense_results
        # Sparse only
        elif sparse_results is not None:
            # Normalize and partial sort sparse results
            docs, scores = sparse_results
            scores = self._normalize_scores(scores)
            results = [
                heapq.nlargest(k, zip(row_docs, row_scores), key=lambda x: x[1])
                for row_docs, row_scores in zip(docs, scores)
            ]

        # Apply the reranker if provided.
        if self.reranker is not None:
            results = self.reranker(texts, results)

        return results
