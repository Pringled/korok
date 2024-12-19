import importlib
from typing import List, Sequence, Tuple
    

class CrossEncoderReranker:
    """A reranker class that uses a cross-encoder to rerank the results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512) -> None:
        """
        Initialize the CrossEncoder reranker.

        :param model_name: The name of the pre-trained cross-encoder model.
        :param max_length: The maximum sequence length for the cross-encoder.
        """
        if not self.is_available():
            raise ImportError("sentence_transformers is not installed. Please install it using `pip install korok[rerankers]`.")
        else:
            global CrossEncoder
            from sentence_transformers import CrossEncoder

        self.cross_encoder = CrossEncoder(model_name)
        self.cross_encoder.max_length = max_length
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if the cross-encoder rerankeris available."""
        return importlib.util.find_spec("sentence_transformers") is not None

    def __call__(
        self,
        query_texts: Sequence[str],
        results_list: List[List[Tuple[str, float]]],
    ) -> List[List[Tuple[str, float]]]:
        """
        Rerank the results using the cross-encoder.

        :param query_texts: The list of query texts.
        :param results_list: The initial results from the vector search.
        :return: The reranked results.
        """
        reranked_results = []
        for query_text, results in zip(query_texts, results_list):
            # Prepare pairs for cross-encoder
            cross_inp = [(query_text, item_id) for item_id, _ in results]
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(cross_inp)
            # Combine items with cross-encoder scores
            reranked = [(item_id, score) for (item_id, _), score in zip(results, cross_scores)]
            # Sort by cross-encoder scores
            reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
            reranked_results.append(reranked)
        return reranked_results
