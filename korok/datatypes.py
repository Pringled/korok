from dataclasses import dataclass

import numpy as np

BM25Result = tuple[list[list[str]], np.ndarray]


@dataclass
class Document:
    text: str | None = None
    vicinity_score: float = 0.0
    bm25_score: float = 0.0

    def combine_scores(self, alpha: float) -> float:
        """Combine the vicinity and bm25 scores."""
        return self.vicinity_score * alpha + self.bm25_score * (1 - alpha)
