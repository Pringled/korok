from typing import Any, Protocol, Sequence

import numpy as np
from vicinity import Metric

from korok.datatypes import DenseResult


class Encoder(Protocol):
    """An encoder protocol."""

    def encode(
        self,
        sentences: list[str] | str | Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of sentences into embeddings.

        :param sentences: A list of sentences to encode.
        :param **kwargs: Additional keyword arguments.
        :return: The embeddings of the sentences.
        """
        ...  # pragma: no cover


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize row-wise."""
    min_vals = np.min(scores, axis=-1, keepdims=True)
    max_vals = np.max(scores, axis=-1, keepdims=True)
    denom = max_vals - min_vals
    denom[denom == 0] = 1e-9
    return (scores - min_vals) / denom


def convert_distances_to_similarities(dense_results: DenseResult, metric: Metric) -> DenseResult:
    """Convert dense results from distances to similarity scores based on the metric."""

    def convert_score(score: float) -> float:
        if metric == Metric.COSINE:
            return 1.0 - score
        elif metric == Metric.INNER_PRODUCT:
            return score
        elif metric == Metric.EUCLIDEAN:
            return 1.0 / (1.0 + score)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    return [[(doc, convert_score(score)) for doc, score in row] for row in dense_results]
