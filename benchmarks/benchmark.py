import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from bm25s.utils.beir import evaluate, postprocess_results_for_eval
from datasets import load_dataset
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer

from korok import Pipeline
from korok.rerankers import CrossEncoderReranker
from korok.utils import Encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    hf_dataset_id: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]], list[str], dict[str, str]]:
    """
    Load a NanoBEIR dataset and prepare an ordered corpus and a lookup mapping.

    :param hf_dataset_id: The Hugging Face dataset identifier.
    :return: A tuple containing:
             - corpus: Mapping of document ID to text.
             - queries: Mapping of query ID to text.
             - qrels: Mapping of query ID to document scores.
             - ordered_corpus_texts: List of texts in corpus order.
             - doc_text_to_id: Mapping from document text to document ID.
    """
    corpus_ds = load_dataset(hf_dataset_id, "corpus", split="train")
    queries_ds = load_dataset(hf_dataset_id, "queries", split="train")
    qrels_ds = load_dataset(hf_dataset_id, "qrels", split="train")

    corpus = {sample["_id"]: sample["text"] for sample in corpus_ds if sample["text"].strip()}
    queries = {sample["_id"]: sample["text"] for sample in queries_ds if sample["text"].strip()}
    qrels: dict[str, dict[str, int]] = {}
    for sample in qrels_ds:
        qid = sample["query-id"]
        cid = sample["corpus-id"]
        score = int(sample.get("score", 1))
        qrels.setdefault(qid, {})[cid] = score

    ordered_doc_ids = list(corpus.keys())
    ordered_corpus_texts = [corpus[doc_id] for doc_id in ordered_doc_ids]
    doc_text_to_id = {text: doc_id for doc_id, text in zip(ordered_doc_ids, ordered_corpus_texts)}
    return corpus, queries, qrels, ordered_corpus_texts, doc_text_to_id


def build_hybrid_pipeline(
    ordered_corpus_texts: list[str],
    encoder: Encoder | None,
    reranker: CrossEncoderReranker | None,
    alpha_value: float,
    use_bm25: bool,
) -> Pipeline:
    """
    Build and return a retrieval pipeline.

    :param ordered_corpus_texts: List of corpus texts.
    :param encoder: Encoder model.
    :param reranker: Optional cross-encoder reranker.
    :param alpha_value: Alpha value for the pipeline.
    :param use_bm25: Whether to use BM25 for sparse retrieval.
    :return: A Pipeline instance configured for hybrid retrieval.
    """
    return Pipeline.fit(
        texts=ordered_corpus_texts,
        encoder=encoder,
        bm25=use_bm25,
        reranker=reranker,
        alpha=alpha_value,
    )


def retrieve_query_results(
    pipeline: Pipeline,
    queries: dict[str, str],
    doc_text_to_id: dict[str, str],
    k: int,
    k_reranker: int,
) -> tuple[list[list[str]], list[list[float]], list[str]]:
    """
    Retrieve ranked document IDs and scores for each query.

    :param pipeline: The Pipeline instance to use for querying.
    :param queries: Mapping of query IDs to query texts.
    :param doc_text_to_id: Mapping from document text to document ID.
    :param k: Number of documents to retrieve (full corpus size).
    :param k_reranker: Number of top documents to consider for reranking.
    :return: A tuple containing:
             - all_ranked_results: List of ranked document IDs per query.
             - all_scores: List of score lists corresponding to the documents.
             - query_ids: List of query IDs.
    """
    all_ranked_results: list[list[str]] = []
    all_scores: list[list[float]] = []
    query_ids: list[str] = []

    for qid, qtext in queries.items():
        # Query returns a list per query; take the first result.
        hybrid_results = pipeline.query([qtext], k=k, k_reranker=k_reranker)[0]
        ranked_doc_ids: list[str] = []
        scores: list[float] = []
        for doc_text, score in hybrid_results:
            doc_id = doc_text_to_id.get(doc_text)
            if doc_id:
                ranked_doc_ids.append(doc_id)
                scores.append(score)
        if ranked_doc_ids:
            all_ranked_results.append(ranked_doc_ids)
            all_scores.append(scores)
            query_ids.append(qid)
        else:
            logger.warning(f"No results retrieved for query {qid}")
    return all_ranked_results, all_scores, query_ids


def evaluate_results(
    qrels: dict[str, dict[str, int]],
    all_ranked_results: list[list[str]],
    all_scores: list[list[float]],
    query_ids: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    """
    Evaluate retrieval results using BEIR evaluation helpers.

    :param qrels: Query relevance judgments.
    :param all_ranked_results: Ranked results for each query.
    :param all_scores: Scores corresponding to each ranked result.
    :param query_ids: List of query IDs.
    :param k_values: List of k values for evaluation.
    :return: Dictionary containing evaluation metrics.
    """
    max_len = max(len(scores) for scores in all_scores)
    # Pad each score list with 0.0 to make them all the same length.
    padded_scores = [scores + [0.0] * (max_len - len(scores)) for scores in all_scores]

    results_for_eval = postprocess_results_for_eval(all_ranked_results, np.array(padded_scores), query_ids)
    ndcg, mean_avg_precision, recall, precision = evaluate(qrels, results_for_eval, k_values)
    return {"ndcg": ndcg, "mean_avg_precision": mean_avg_precision, "recall": recall, "precision": precision}


def initialize_models(
    encoder_model: str | None,
    reranker_model: str | None,
    device: str | None,
) -> tuple[Encoder | None, CrossEncoderReranker | None]:
    """
    Initialize and return the encoder and reranker models.

    :param encoder_model: Pretrained encoder model name or path.
    :param reranker_model: Optional reranker model name or path.
    :param device: Device to use for inference.
    :return: Tuple of (encoder, reranker).
    """
    if not encoder_model:
        encoder = None
    elif encoder_model == "minishlab/potion-retrieval-32M":
        encoder = StaticModel.from_pretrained(encoder_model)
    else:
        encoder = SentenceTransformer(encoder_model, trust_remote_code=True, device=device)

    reranker = CrossEncoderReranker(reranker_model, trust_remote_code=True, device=device) if reranker_model else None
    return encoder, reranker


def build_save_folder_name(
    encoder_model: str | None,
    use_bm25: bool,
    reranker_model: str | None,
    alpha_value: float,
    k_reranker: int,
) -> str:
    """Build the folder name based on the model names, BM25 flag, alpha value, and k_reranker."""
    encoder_part = encoder_model.split("/")[-1].replace("_", "-") if encoder_model else ""
    bm25_part = "bm25" if use_bm25 else ""
    reranker_part = reranker_model.split("/")[-1].replace("_", "-") if reranker_model else ""
    parts = [part for part in (encoder_part, bm25_part, reranker_part) if part]
    base_name = "_".join(parts)
    return f"{base_name}_alpha{alpha_value}_kr{k_reranker}"


def save_json(data: Any, path: Path) -> None:
    """
    Save data as a JSON file to the specified path.

    :param data: Data to save.
    :param path: Path to the output file.
    """
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def process_dataset(
    ds_name: str,
    hf_id: str,
    encoder: Encoder | None,
    reranker: CrossEncoderReranker | None,
    alpha_value: float,
    use_bm25: bool,
    k_reranker: int,
    output_dir: Path,
    k_values: list[int],
) -> tuple[dict[str, Any] | None, float, float, int]:
    """Process a single dataset: load data, build pipeline, query and evaluate results, and save the results to a file."""
    try:
        logger.info(f"=== Processing dataset: {ds_name} ===")
        corpus, queries, qrels, ordered_texts, doc_text_to_id = load_and_prepare_dataset(hf_id)
        logger.info(f"Loaded corpus with {len(corpus)} documents and {len(queries)} queries.")
        k = len(ordered_texts)  # Use full corpus size

        fit_start = time.perf_counter()
        pipeline = build_hybrid_pipeline(ordered_texts, encoder, reranker, alpha_value, use_bm25)
        fit_time = time.perf_counter() - fit_start
        logger.info(f"Pipeline fitted in {fit_time:.4f} seconds.")

        query_start = time.perf_counter()
        all_ranked_results, all_scores, query_ids = retrieve_query_results(
            pipeline, queries, doc_text_to_id, k, k_reranker
        )
        query_time = time.perf_counter() - query_start
        logger.info(f"Retrieved results for {len(query_ids)} queries in {query_time:.4f} seconds.")

        qps = len(query_ids) / query_time if query_time > 0 else 0.0

        metrics = evaluate_results(qrels, all_ranked_results, all_scores, query_ids, k_values)
        metrics["qps"] = qps
        metrics["pipeline_fit_time"] = fit_time

        results_path = output_dir / f"{ds_name}_results.json"
        save_json(metrics, results_path)
        logger.info(f"Saved results for {ds_name} to {results_path}")

        return metrics, fit_time, query_time, len(query_ids)
    except Exception as e:
        logger.exception(f"Error processing {ds_name}: {e}")
        return None, 0.0, 0.0, 0


def main(
    encoder_model: str | None,
    reranker_model: str | None,
    alpha_value: float,
    k_reranker: int,
    save_path: str,
    use_bm25: bool,
    overwrite_results: bool,
    device: str | None,
) -> None:
    """
    Evaluate a retrieval pipeline on multiple NanoBEIR datasets.

    :param encoder_model: Pretrained encoder model name or path.
    :param reranker_model: Optional reranker model name or path.
    :param alpha_value: Alpha value for the pipeline (0 <= alpha <= 1).
    :param k_reranker: Number of top documents to re-rank.
    :param save_path: Directory to save results.
    :param use_bm25: Whether BM25 (sparse retrieval) is used.
    :param overwrite_results: If False and the save folder exists, skip evaluation.
    :param device: Device to use for inference.
    """
    dataset_name_to_id: dict[str, str] = {
        "climatefever": "zeta-alpha-ai/NanoClimateFEVER",
        "dbpedia": "zeta-alpha-ai/NanoDBPedia",
        "fever": "zeta-alpha-ai/NanoFEVER",
        "fiqa2018": "zeta-alpha-ai/NanoFiQA2018",
        "hotpotqa": "zeta-alpha-ai/NanoHotpotQA",
        "msmarco": "zeta-alpha-ai/NanoMSMARCO",
        "nfcorpus": "zeta-alpha-ai/NanoNFCorpus",
        "nq": "zeta-alpha-ai/NanoNQ",
        "quoraretrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
        "scidocs": "zeta-alpha-ai/NanoSCIDOCS",
        "arguana": "zeta-alpha-ai/NanoArguAna",
        "scifact": "zeta-alpha-ai/NanoSciFact",
        "touche2020": "zeta-alpha-ai/NanoTouche2020",
    }

    # Initialize models.
    encoder, reranker = initialize_models(encoder_model, reranker_model, device)

    # Build output directory based on models and parameters.
    save_folder = build_save_folder_name(encoder_model, use_bm25, reranker_model, alpha_value, k_reranker)
    output_dir = Path(save_path) / save_folder

    if output_dir.exists() and not overwrite_results:
        logger.info(f"Output folder '{output_dir}' already exists and overwrite_results is False. Skipping evaluation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to folder: {output_dir}")

    # Save configuration, now including the 'device'
    config = {
        "encoder_model": encoder_model,
        "reranker_model": reranker_model,
        "alpha_value": alpha_value,
        "k_reranker": k_reranker,
        "bm25": use_bm25,
        "device": device,
        "output_dir": str(output_dir),
        "dataset_name_to_id": dataset_name_to_id,
    }
    config_path = output_dir / "config.json"
    save_json(config, config_path)
    logger.info(f"Saved configuration to {config_path}")

    k_values: list[int] = [1, 3, 5, 10, 100]
    all_metrics: dict[str, dict[str, Any]] = {}
    total_queries = 0
    total_query_time = 0.0
    total_fit_time = 0.0
    dataset_count = 0

    # Process each dataset.
    for ds_name, hf_id in dataset_name_to_id.items():
        metrics, fit_time, query_time, num_queries = process_dataset(
            ds_name, hf_id, encoder, reranker, alpha_value, use_bm25, k_reranker, output_dir, k_values
        )
        if metrics is not None:
            all_metrics[ds_name] = metrics
            total_fit_time += fit_time
            total_query_time += query_time
            total_queries += num_queries
            dataset_count += 1

    overall_qps = total_queries / total_query_time if total_query_time > 0 else 0.0
    avg_fit_time = total_fit_time / dataset_count if dataset_count > 0 else 0.0

    # Aggregate scores across datasets.
    aggregated_scores: dict[str, dict[str, float]] = {}
    for mtype in ["ndcg", "mean_avg_precision", "recall", "precision"]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for ds_metrics in all_metrics.values():
            if mtype in ds_metrics:
                for key, val in ds_metrics[mtype].items():
                    sums[key] = sums.get(key, 0.0) + val
                    counts[key] = counts.get(key, 0) + 1
        if counts:
            aggregated_scores[mtype] = {key: sums[key] / counts[key] for key in sums}

    logger.info(f"Overall QPS: {overall_qps:.4f}")
    logger.info("Aggregated Scores:")
    for mtype, scores in aggregated_scores.items():
        logger.info(f"  {mtype.upper()}: {scores}")

    overall_eval = {
        "dataset_scores": all_metrics,
        "aggregated_scores": aggregated_scores,
        "throughput": {
            "qps": overall_qps,
            "pipeline_fit_time": avg_fit_time,
            "total_query_time": total_query_time,
            "total_queries": total_queries,
        },
    }
    overall_save_path = output_dir / "overall_results.json"
    save_json(overall_eval, overall_save_path)
    logger.info(f"Saved overall results to {overall_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a retrieval pipeline on multiple NanoBEIR datasets.")
    parser.add_argument(
        "--encoder_model",
        type=str,
        default=None,
        help="Pretrained encoder model name or path (e.g., 'minishlab/potion-retrieval-32M').",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default=None,
        help="Optional reranker model name or path (e.g., 'BAAI/bge-reranker-v2-m3').",
    )
    parser.add_argument(
        "--alpha_value",
        type=float,
        default=0.5,
        help="Alpha value for the pipeline (0 <= alpha <= 1).",
    )
    parser.add_argument(
        "--k_reranker",
        type=int,
        default=30,
        help="Number of top documents to re-rank.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save results.",
    )
    parser.add_argument(
        "--bm25",
        action="store_true",
        help="If set, BM25 (sparse retrieval) is used for hybrid search.",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        default=False,
        help="If set, overwrite results even if the save folder already exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference.",
    )
    args = parser.parse_args()

    main(
        encoder_model=args.encoder_model,
        reranker_model=args.reranker_model,
        alpha_value=args.alpha_value,
        k_reranker=args.k_reranker,
        save_path=args.save_path,
        use_bm25=args.bm25,
        overwrite_results=args.overwrite_results,
        device=args.device,
    )
