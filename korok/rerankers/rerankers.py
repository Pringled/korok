from typing import Any, Sequence

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int | None = None,
        trust_remote_code: bool = False,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CrossEncoder reranker.

        :param model_name: The name of the pre-trained cross-encoder model.
        :param max_length: The maximum sequence length for the cross-encoder.
        :param trust_remote_code: A boolean indicating whether to trust remote code.
        :param device: The device to use for inference.
        :param **kwargs: Additional keyword arguments.
        """
        self.cross_encoder = CrossEncoder(
            model_name,
            max_length=max_length,  # type: ignore[arg-type]
            trust_remote_code=trust_remote_code,
            device=device,
            **kwargs,
        )

    def __call__(
        self,
        query_texts: Sequence[str],
        results_list: list[list[tuple[str, float]]],
    ) -> list[list[tuple[str, float]]]:
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
