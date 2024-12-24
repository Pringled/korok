from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Tuple

import bm25s
import numpy as np
from model2vec import StaticModel
from vicinity import Backend, Metric, Vicinity

from korok.rerankers import CrossEncoderReranker


@dataclass
class Document:
    text: str = None
    vicinity_score: float = 0.0
    bm25_score: float = 0.0

    def combine_scores(self, alpha: float) -> float:
        """Combine the vicinity and bm25 scores."""
        return self.vicinity_score * alpha + self.bm25_score * (1 - alpha)

class Pipeline:
    def __init__(
        self,
        encoder: StaticModel,
        vicinity: Vicinity,
        reranker: CrossEncoderReranker | None = None,
        bm25: bm25s.BM25 | None = None,
        alpha: float = 0.5,
        corpus: list[str] | None = None,
    ) -> None:
        """
        Initialize a Pipeline instance.

        :param encoder: The encoder used to encode the items.
        :param vicinity: The vicinity object used to find nearest neighbors.
        :param reranker: The reranker used to rerank the results (optional).
        :param bm25: The bm25 index used for hybrid search (optional).
        :param alpha: The alpha value for the hybrid search (optional).
        :param corpus: The corpus used for bm25 search (optional).

        """
        self.encoder = encoder
        self.vicinity = vicinity
        self.reranker = reranker
        self.bm25 = bm25
        self.alpha = alpha
        self.corpus = corpus

        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("Alpha must be between 0 and 1")

    @classmethod
    def fit(
        cls,
        texts: list[str],
        encoder: StaticModel,
        reranker: CrossEncoderReranker | None = None,
        hybrid: bool = False,
        alpha: float = 1.0,
        stopwords: str | List[str] = "en",
        **kwargs: Any,
    ) -> Pipeline:
        """
        Fit the encoder to the texts.

        :param texts: The texts to fit the encoder to.
        :param encoder: The encoder to use.
        :param reranker: The reranker to use (optional).
        :param hybrid: Whether to use hybrid search (optional).
        :param alpha: The alpha value for the hybrid search (optional).
        :param stopwords: The stopwords to use for bm25 search (optional).
        :param **kwargs: Additional keyword arguments.
        :return: A Pipeline instance.
        """
        vectors = encoder.encode(texts, show_progressbar=True)
        vicinity = Vicinity.from_vectors_and_items(
            vectors=vectors,
            items=texts,
            backend_type=Backend.BASIC,
            metric=Metric.COSINE,
            **kwargs,
        )
        
        # Add bm25s if hybrid search is enabled and alpha < 1.0
        if hybrid and alpha < 1.0:
            bm25 = bm25s.BM25()
            tokens = bm25s.tokenize(texts, stopwords=stopwords)
            bm25.index(tokens)
        else:
            bm25 = None

        return cls(encoder=encoder, vicinity=vicinity, reranker=reranker, bm25=bm25, alpha=alpha, corpus=texts)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize the scores to be between 0 and 1."""
        # Get the min and max scores for each document
        min_score = np.min(scores, axis=-1).reshape(-1, 1)
        max_score = np.max(scores, axis=-1).reshape(-1, 1)

        # Normalize the scores
        denom = max_score - min_score
        numerator = scores - min_score

        # Handle division by zero, zero if max == min, i.e. all scores are the same
        result = np.divide(numerator, denom, out=scores, where=denom != 0)
        return result

    def _split_vicinity_results(self, results: List[List[Tuple[str, float]]]) -> Tuple[List[str], np.ndarray]:
        """Split the results from vector search into two lists."""
        vicinity_docs = [[doc for doc, _ in result] for result in results]
        vicinity_scores = np.array([[score for _, score in result] for result in results])
        return (vicinity_docs, vicinity_scores)

    def _merge_results(
        self,
        vicinity_results: List[List[Tuple[str, np.float64]]],
        bm25_results: Tuple[List[List[str]], np.ndarray],
        k: int = 10,
    ) -> List[List[tuple[str, float]]]:
        """Merge the results from vector search and bm25 search."""
        # Initialize the scores list
        scores_list = []

        # Split the results into docs and scores
        vicinity_docs, vicinity_scores = self._split_vicinity_results(vicinity_results)
        bm25_docs, bm25_scores = bm25_results

        # convert bm25 docs to strings
        bm25_docs = [[str(doc) for doc in doclist] for doclist in bm25_docs]

        # Normalize the scores
        vicinity_scores = self._normalize_scores(vicinity_scores)
        bm25_scores = self._normalize_scores(bm25_scores)

        # Combine the docs and scores into a dictionary (to account for mismatched docs)
        for i in range(len(vicinity_results)):
            scores_list.append(defaultdict(Document))
            for doc, score in zip(vicinity_docs[i], vicinity_scores[i]):
                scores_list[i][doc].text = doc
                scores_list[i][doc].vicinity_score = score

            for doc, score in zip(bm25_docs[i], bm25_scores[i]):
                scores_list[i][doc].text = doc
                scores_list[i][doc].bm25_score = score

        # Combine the docs and scores into a list of tuples
        results = []
        for i in range(len(vicinity_results)):
            # Get the combined scores
            result = [(key, value.combine_scores(self.alpha)) for key, value in scores_list[i].items()]
            
            # Sort the scores
            result.sort(key=lambda x: x[1], reverse=True)

            # Take the top k results
            results.append(result[:k])

        return results

    def query(
        self,
        texts: list[str],
        k: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """
        Find the nearest neighbors for a list of texts.

        :param texts: Texts to query for
        :param k: The number of most similar items to retrieve.
        :return: For each item in the input, the num most similar items are returned in the form of
            (NAME, SIMILARITY) tuples.
        """
        # Do vector search
        vectors = self.encoder.encode(texts, show_progressbar=True)
        results = self.vicinity.query(vectors, k)

        # Do bm25 search if alpha < 1 and merge results
        if self.bm25 is not None and self.alpha < 1.0:
            query_tokens = bm25s.tokenize(texts, stopwords="en")
            bm25_results = self.bm25.retrieve(query_tokens, k=k, corpus=self.corpus, return_as="tuple")
            results = self._merge_results(results, bm25_results, k=k)

        # Apply reranker if available
        if self.reranker is not None:
            results = self.reranker(texts, results)

        return results
