from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any, DefaultDict

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder
from vicinity import Backend, Metric, Vicinity

from korok.datatypes import DenseResult, Document, HybridResult, QueryResult, SparseResult
from korok.utils import Encoder, convert_distances_to_similarities, normalize_scores


class Pipeline:
    def __init__(
        self,
        encoder: Encoder | None = None,
        dense_index: Vicinity | None = None,
        sparse_index: bm25s.BM25 | None = None,
        reranker: CrossEncoder | None = None,
        distance_metric: Metric = Metric.COSINE,
        alpha: float = 0.5,
        corpus: list[str] | None = None,
        stopwords: list[str] | None = None,
    ) -> None:
        """
        Initialize a Pipeline instance.

        :param encoder: An encoder for dense vector search.
        :param dense_index: A dense vector index using the provided encoder.
        :param sparse_index: A sparse vector index using BM25.
        :param reranker: A cross-encoder reranker.
        :param distance_metric: The distance metric for the dense vector index.
        :param alpha: The alpha value for hybrid search.
        :param corpus: List of documents used for BM25.
        :param stopwords: Stopwords for BM25 tokenization.
        """
        self.encoder = encoder
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.reranker = reranker
        self.distance_metric = distance_metric
        self.alpha = alpha
        self.corpus = corpus
        self.stopwords = stopwords

    @classmethod
    def fit(
        cls,
        texts: list[str],
        encoder: Encoder | None = None,
        use_bm25: bool = False,
        reranker: CrossEncoder | None = None,
        backend_type: Backend = Backend.BASIC,
        distance_metric: Metric = Metric.COSINE,
        alpha: float = 0.5,
        stopwords: str | list[str] = "en",
        **kwargs: Any,
    ) -> Pipeline:
        """
        Fit a pipeline on a corpus of documents.

        - If an encoder is provided, build a dense vector index using the encoder.
        - If use_bm25 is True, build a sparse vector index using BM25.
        - If both are provided, build a hybrid index.
        - If a reranker is provided, rerank the results for each query.

        :param texts: The corpus of documents to index.
        :param encoder: An encoder for dense vector search.
        :param use_bm25: A bool indicating whether to build a BM25 index for sparse vector search.
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
        :raises ValueError: If the distance metric is not supported for hybrid search.
        """
        if encoder is None and use_bm25 is False:
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
        if use_bm25:
            sparse_index = bm25s.BM25()
            tokens = bm25s.tokenize(texts, stopwords=stopwords)
            sparse_index.index(tokens)

        # Check if the distance metric is supported for hybrid search
        if dense_index and sparse_index:
            if distance_metric not in [Metric.COSINE, Metric.INNER_PRODUCT, Metric.EUCLIDEAN]:
                raise ValueError(
                    "Unsupported metric for hybrid search. Use Metric.COSINE, "
                    "Metric.INNER_PRODUCT, or Metric.EUCLIDEAN."
                )

        return cls(
            encoder=encoder,
            dense_index=dense_index,
            sparse_index=sparse_index,
            reranker=reranker,
            distance_metric=distance_metric,
            alpha=alpha,
            corpus=texts,
        )

    def _merge_results(
        self,
        dense_results: DenseResult,
        sparse_results: SparseResult,
        k: int,
    ) -> HybridResult:
        """Merge dense and sparse results."""
        # Unpack dense results
        dense_docs, dense_scores = (
            [[doc for doc, _ in row] for row in dense_results],
            np.array([[float(score) for _, score in row] for row in dense_results]),
        )
        # Unpack sparse results
        sparse_docs, sparse_scores = sparse_results

        # Convert sparse docs to strings
        sparse_docs = [[str(doc) for doc in doclist] for doclist in sparse_docs]

        # Normalize scores
        dense_scores = normalize_scores(dense_scores)
        sparse_scores = normalize_scores(sparse_scores)

        results: HybridResult = []
        for i in range(len(dense_results)):
            doc_map: DefaultDict[str, Document] = defaultdict(Document)

            # Assign scores for each document
            for doc, score in zip(dense_docs[i], dense_scores[i]):
                doc_map[doc].text = doc
                doc_map[doc].dense_score = float(score)

            for doc, score in zip(sparse_docs[i], sparse_scores[i]):
                doc_map[doc].text = doc
                doc_map[doc].sparse_score = float(score)

            # Combine scores and sort
            combined = [(doc_id, d_obj.combine_scores(self.alpha)) for doc_id, d_obj in doc_map.items()]
            top_k = heapq.nlargest(k, combined, key=lambda x: x[1])
            results.append(top_k)

        return results

    def query(self, texts: list[str], k: int = 10, k_reranker: int = 30, instruction: str | None = None) -> QueryResult:
        """
        Query the pipeline.

        This does the following:
          - Hybrid search if we have both dense and sparse indexes.
          - Dense search if we have only have a dense index.
          - Sparse search if we have only a sparse index.
          - Reranking if a reranker is provided.

        :param texts: The list of texts to query.
        :param k: The number of results to return.
        :param k_reranker: The number of results to consider for reranking.
        :param instruction: An optional instruction to add to the query (for dense retrieval).
        :return: The search results.
        """
        # Compute dense results if both dense index and encoder are available
        dense_results = None
        if self.dense_index and self.encoder:
            # If an instruction is provided, combine it with each query for dense retrieval.
            if instruction:
                vectors = self.encoder.encode([f"{instruction} {text}" for text in texts], show_progressbar=True)
            else:
                vectors = self.encoder.encode(texts, show_progressbar=True)
            dense_results = self.dense_index.query(vectors, k_reranker)
            # Convert distances to similarities
            dense_results = convert_distances_to_similarities(dense_results, self.distance_metric)

        # Compute sparse results if sparse index is available
        sparse_results = None
        if self.sparse_index and self.corpus:
            tokens = bm25s.tokenize(texts, stopwords=self.stopwords)
            if k_reranker > len(self.corpus):
                # If k_reranker is greater than the number of documents in the corpus, set it to the corpus size
                k_reranker = len(self.corpus)
            sparse_results = self.sparse_index.retrieve(tokens, k=k_reranker, corpus=self.corpus, return_as="tuple")

        # Hybrid search
        if dense_results and sparse_results:
            results = self._merge_results(dense_results, sparse_results, k_reranker)
        # Dense search
        elif dense_results:
            results = dense_results
        # Sparse search
        elif sparse_results:
            # Normalize and sort sparse results
            docs, scores = sparse_results
            scores = normalize_scores(scores)
            results = [
                heapq.nlargest(k_reranker, zip(row_docs, row_scores), key=lambda x: x[1])
                for row_docs, row_scores in zip(docs, scores)
            ]

        # Rerank the results
        if self.reranker:
            reranked_results = []
            for query, candidates in zip(texts, results):
                # Extract the candidate documents
                documents = [document for document, _ in candidates]
                # Rerank the documents
                reranked_documents = self.reranker.rank(query, documents, top_k=k_reranker, return_documents=True)
                # Convert ranked output into a list of (document, score) tuples.
                reranked_result = [(str(item["text"]), float(item["score"])) for item in reranked_documents]
                reranked_results.append(reranked_result)
            results = reranked_results

        # Convert outputs and return the top k results
        results = [[(str(doc), float(score)) for doc, score in row][:k] for row in results]
        return results
