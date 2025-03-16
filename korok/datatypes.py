from dataclasses import dataclass

import numpy as np
from vicinity.datatypes import SimilarityResult

SparseResult = tuple[list[list[str]], np.ndarray]
DenseResult = SimilarityResult
HybridResult = SimilarityResult
QueryResult = SimilarityResult


@dataclass
class Document:
    text: str | None = None
    dense_score: float = 0.0
    sparse_score: float = 0.0

    def combine_scores(self, alpha: float) -> float:
        """Combine the dense and sparse scores, weighted by alpha."""
        return self.dense_score * alpha + self.sparse_score * (1 - alpha)
