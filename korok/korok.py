from __future__ import annotations

from typing import Any

from model2vec import StaticModel
from vicinity import Backend, Metric, Vicinity

from korok.rerankers import CrossEncoderReranker


class Pipeline:
    def __init__(
        self,
        encoder: StaticModel,
        vicinity: Vicinity,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        """
        Initialize a Pipeline instance.

        :param encoder: The encoder used to encode the items.
        :param vicinity: The vicinity object used to find nearest neighbors.
        :param reranker: The reranker used to rerank the results (optional).
        """
        self.encoder = encoder
        self.vicinity = vicinity
        self.reranker = reranker

    @classmethod
    def fit(
        cls,
        texts: list[str],
        encoder: StaticModel,
        reranker: CrossEncoderReranker | None = None,
        **kwargs: Any,
    ) -> Pipeline:
        """
        Fit the encoder to the texts.

        :param texts: The texts to fit the encoder to.
        :param encoder: The encoder to use.
        :param reranker: The reranker to use (optional).
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
        return cls(encoder=encoder, vicinity=vicinity, reranker=reranker)

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
        vectors = self.encoder.encode(texts, show_progressbar=True)
        results = self.vicinity.query(vectors, k)

        # Apply reranker if available
        if self.reranker is not None:
            results = self.reranker(texts, results)
        return results
