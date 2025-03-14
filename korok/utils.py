from typing import Any, Protocol, Sequence

import numpy as np


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
