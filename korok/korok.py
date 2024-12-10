from __future__ import annotations

from typing import Any, TYPE_CHECKING

import importlib

from model2vec import StaticModel
from vicinity import Backend, Metric, Vicinity

# We need to import these classes to check if they are installed
if TYPE_CHECKING:
    import bm25s
    from bm25s import BM25S
    import ranx
    
from korok.rerankers import CrossEncoderReranker


class Pipeline:
    def __init__(
        self,
        encoder: StaticModel,
        vicinity: Vicinity,
        bm25s: "BM25S" | None = None
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        """
        Initialize a Pipeline instance.

        :param encoder: The encoder used to encode the items.
        :param vicinity: The vicinity object used to find nearest neighbors.
        :param bm25s: The BM25S object used for keyword search.
        :param reranker: The reranker used to rerank the results (optional).
        """
        self.encoder = encoder
        self.vicinity = vicinity
        self.reranker = reranker

        # Keeping the hybrid components separate from the main pipeline
        # and importing them only when needed
        if bm25s is not None:
            self._check_and_import_hybrid_components()
            self.bm25 = bm25s
            self.rrf = ranx.RRF()
        

    # To check if the hybrid components are installed
    def _check_and_import_hybrid_components(self) -> bool:
        try:
            import bm25s
            import ranx
            return True
        except ImportError:
            raise ImportError("hybrid is not installed. Please install it using `pip install korok[hybrid]`.")

    @classmethod
    def fit(
        cls,
        texts: list[str],
        encoder: StaticModel,
        hybrid: bool = False,
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

        # If hybrid is enabled, we need to fit the BM25S model
        if hybrid:
            bm25 = bm25s.BM25S()
            corpus_tokens = bm25s.tokenize(texts, stopwords="en")
            bm25.index(corpus_tokens)
            return cls(encoder=encoder, vicinity=vicinity, bm25s=bm25, reranker=reranker)
        
        return cls(encoder=encoder, vicinity=vicinity, reranker=reranker)
    
    def _fuse_results(self, 
                      texts: list[str],
                      results: list[list[tuple[str, float]]],
                      bm25_results: list[list[tuple[str, float]]]) -> list[list[tuple[str, float]]]:
        """Fuse the results from the BM25S model and the vicinity model."""
        # Convert the results to a ranx Run objects
        bm25_run_dict = {
            "query_id": [i for i in range(len(texts))],
            "doc_id": [doc_id for doc_id in bm25_results.keys()],
            "score": [score for score in bm25_results.values()],
        }
        bm25_run = ranx.Run(bm25_run_dict, name="bm25")

        vicinity_run_dict = {
            "query_id": [i for i in range(len(texts))],
            "doc_id": [doc_id for doc_id in results.keys()],
            "score": [score for score in results.values()],
        }
        vicinity_run = ranx.Run(vicinity_run_dict, name="vicinity")

        # Fuse the results
        results = ranx.fuse(runs=[bm25_run, vicinity_run],
                            method="rrf",
                            norm="rank"
                            ).to_dict()
        results = [
            [(doc_id, score) for doc_id, score in run.items()]
            for run in results.values()
        ]
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
        vectors = self.encoder.encode(texts, show_progressbar=True)
        results = self.vicinity.query(vectors, k)

        # If the BM25S model is enabled, we need to fuse the results
        if self.bm25 is not None:
            query_tokens = self.bm25s.tokenize(texts)
            bm25_results = self.bm25.retrieve(query_tokens, k)
            results = self._fuse_results(texts, results, bm25_results)

        # Apply reranker if available
        if self.reranker is not None:
            results = self.reranker(texts, results)
        return results


