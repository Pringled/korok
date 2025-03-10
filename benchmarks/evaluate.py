import argparse
import json
import logging
import os
import time

import numpy as np
from bm25s.utils.beir import evaluate, postprocess_results_for_eval
from datasets import load_dataset
from model2vec import StaticModel

from korok import Pipeline
from korok.rerankers import CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    hf_dataset_id: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]], list[str], dict[str, str]]:
    """
    Load a NanoBEIR dataset and prepare an ordered corpus and a lookup mapping.

    :param hf_dataset_id: Hugging Face dataset identifier.
    :return: Tuple containing:
             - corpus: mapping of document ID to text,
             - queries: mapping of query ID to text,
             - qrels: mapping of query ID to document scores,
             - ordered_corpus_texts: list of texts in corpus order,
             - doc_text_to_id: mapping from text to document ID.
    """
    corpus_ds = load_dataset(hf_dataset_id, "corpus", split="train")
    queries_ds = load_dataset(hf_dataset_id, "queries", split="train")
    qrels_ds = load_dataset(hf_dataset_id, "qrels", split="train")

    corpus = {sample["_id"]: sample["text"] for sample in corpus_ds if sample["text"].strip() != ""}
    queries = {sample["_id"]: sample["text"] for sample in queries_ds if sample["text"].strip() != ""}
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
    ordered_corpus_texts: list[str], encoder: StaticModel, reranker: CrossEncoderReranker | None, alpha_value: float
) -> Pipeline:
    """
    Build and return a hybrid retrieval pipeline. If alpha is 1.0, the hybrid flag is set to False.

    :param ordered_corpus_texts: List of corpus texts.
    :param encoder: Encoder model.
    :param reranker: Optional cross-encoder reranker.
    :param alpha_value: Alpha value for the pipeline.
    :return: A Pipeline instance.
    """
    hybrid_flag = False if alpha_value == 1.0 else True
    return Pipeline.fit(
        texts=ordered_corpus_texts, encoder=encoder, reranker=reranker, hybrid=hybrid_flag, alpha=alpha_value
    )


def retrieve_query_results(
    pipeline: Pipeline, queries: dict[str, str], doc_text_to_id: dict[str, str], k: int, k_reranker: int
) -> tuple[list[list[str]], list[list[float]], list[str]]:
    """
    Retrieve ranked document IDs and scores for each query.

    :param pipeline: The retrieval pipeline.
    :param queries: Mapping from query ID to query text.
    :param doc_text_to_id: Mapping from document text to document ID.
    :param k: Number of top results to retrieve (set to full corpus size).
    :param k_reranker: Number of top documents to re-rank.
    :return: Tuple containing:
             - all_ranked_results: List of ranked document IDs per query.
             - all_scores: List of score lists corresponding to the documents.
             - query_ids: List of query IDs.
    """
    all_ranked_results, all_scores, query_ids = [], [], []
    for qid, qtext in queries.items():
        hybrid_results = pipeline.query([qtext], k=k, k_reranker=k_reranker)[0]
        ranked_doc_ids = []
        scores = []
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
) -> dict[str, any]:
    """
    Evaluate retrieval results using BEIR evaluation helpers.

    :param qrels: Ground truth relevance judgments.
    :param all_ranked_results: List of ranked document IDs for each query.
    :param all_scores: List of scores corresponding to the documents.
    :param query_ids: List of query IDs.
    :param k_values: List of cutoff values.
    :return: Dictionary of evaluation metrics.
    """
    results_for_eval = postprocess_results_for_eval(all_ranked_results, np.array(all_scores), query_ids)
    ndcg, _map, recall, precision = evaluate(qrels, results_for_eval, k_values)
    return {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}


def main(encoder_model: str, reranker_model: str | None, alpha_value: float, k_reranker: int, save_path: str) -> None:  # noqa C901
    """Main function to evaluate a hybrid retrieval pipeline on multiple NanoBEIR datasets."""
    # Dataset mappings.
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

    # Clean up model names.
    encoder_name_clean = encoder_model.split("/")[-1].replace("_", "-")
    reranker_name_clean = reranker_model.split("/")[-1].replace("_", "-") if reranker_model else "no-reranker"

    # Determine base folder name.
    if alpha_value == 0:
        base_name = "bm25"
    elif alpha_value == 1:
        base_name = encoder_name_clean
    else:
        base_name = f"{encoder_name_clean}_bm25"
    save_folder = f"{base_name}_{reranker_name_clean}_alpha{alpha_value}_kr{k_reranker}"
    output_dir = os.path.join(save_path, save_folder)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving results to folder: {output_dir}")

    # Save configuration.
    config = {
        "encoder_model": encoder_model,
        "reranker_model": reranker_model,
        "alpha_value": alpha_value,
        "k_reranker": k_reranker,
        "output_dir": output_dir,
        "dataset_name_to_id": dataset_name_to_id,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved configuration to {os.path.join(output_dir, 'config.json')}")

    # Initialize models.
    encoder = StaticModel.from_pretrained(encoder_model)
    reranker = CrossEncoderReranker(reranker_model) if reranker_model else None

    k_values: list[int] = [1, 3, 5, 10, 100]
    all_metrics: dict[str, dict[str, any]] = {}
    total_queries = total_time = total_fit_time = 0.0
    dataset_count = 0

    for ds_name, hf_id in dataset_name_to_id.items():
        try:
            logger.info(f"=== Processing dataset: {ds_name} ===")
            corpus, queries, qrels, ordered_texts, doc_text_to_id = load_and_prepare_dataset(hf_id)
            logger.info(f"Loaded corpus with {len(corpus)} documents and {len(queries)} queries.")
            k = len(ordered_texts)  # use full corpus size

            fit_start = time.time()
            pipeline = build_hybrid_pipeline(ordered_texts, encoder, reranker, alpha_value)
            fit_time = time.time() - fit_start
            logger.info(f"Pipeline fitted in {fit_time:.4f} seconds.")
            total_fit_time += fit_time

            start_time = time.time()
            all_ranked_results, all_scores, query_ids = retrieve_query_results(
                pipeline, queries, doc_text_to_id, k, k_reranker, logger
            )
            elapsed = time.time() - start_time
            logger.info(f"Retrieved results for {len(query_ids)} queries in {elapsed:.4f} seconds.")
            qps = len(query_ids) / elapsed if elapsed > 0 else 0.0

            total_queries += len(query_ids)
            total_time += elapsed
            dataset_count += 1

            metrics = evaluate_results(qrels, all_ranked_results, all_scores, query_ids, k_values)
            metrics["qps"] = qps
            metrics["pipeline_fit_time"] = fit_time

            all_metrics[ds_name] = metrics
            logger.info(f"Results for {ds_name}: {metrics}")

            with open(os.path.join(output_dir, f"{ds_name}_results.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved results for {ds_name}.")

        except Exception as e:
            logger.error(f"Error processing {ds_name}: {e}")

    overall_qps = total_queries / total_time if total_time > 0 else 0.0
    avg_fit_time = total_fit_time / dataset_count if dataset_count > 0 else 0.0

    # Compute aggregated scores.
    aggregated_scores: dict[str, dict[str, float]] = {}
    for mtype in ["ndcg", "map", "recall", "precision"]:
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
            "total_time": total_time,
            "total_queries": total_queries,
        },
    }
    overall_save_path = os.path.join(output_dir, "overall_results.json")
    with open(overall_save_path, "w") as f:
        json.dump(overall_eval, f, indent=4)
    logger.info(f"Saved overall results to {overall_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a hybrid retrieval pipeline on multiple NanoBEIR datasets.")
    parser.add_argument(
        "--encoder_model",
        type=str,
        required=True,
        help="Pretrained encoder model name or path (e.g., 'minishlab/potion_retrieval_32M').",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default=None,
        help="Optional reranker model name or path (e.g., 'BAAI/bge_reranker_v2_m3').",
    )
    parser.add_argument(
        "--alpha_value",
        type=float,
        required=True,
        help="Alpha value for the pipeline (0 <= alpha <= 1). If 1.0, hybrid search is disabled.",
    )
    parser.add_argument("--k_reranker", type=int, required=True, help="Number of top documents to re-rank.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save results.")
    args = parser.parse_args()

    main(
        encoder_model=args.encoder_model,
        reranker_model=args.reranker_model,
        alpha_value=args.alpha_value,
        k_reranker=args.k_reranker,
        save_path=args.save_path,
    )
