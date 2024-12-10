from __future__ import annotations

from typing import Any

from model2vec import StaticModel
from vicinity import Backend, Metric, Vicinity


class Pipeline:
    def __init__(
        self,
        encoder: StaticModel,
        vicinity: Vicinity,
    ) -> None:
        """
        Initialize a Pipeline instance.

        :param encoder: The encoder used to encode the items.
        :param vicinity: The vicinity object used to find nearest neighbors.
        """
        self.encoder = encoder
        self.vicinity = vicinity

    @classmethod
    def fit(
        cls,
        texts: list[str],
        encoder: StaticModel,
        **kwargs: Any,
    ) -> Pipeline:
        """
        Fit the encoder to the texts.

        :param texts: The texts to fit the encoder to.
        :param encoder: The encoder to use.
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
        return cls(encoder=encoder, vicinity=vicinity)

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
        return results
