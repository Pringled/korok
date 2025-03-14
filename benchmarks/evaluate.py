import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]], List[str], Dict[str, str]]:
    """
    Load a NanoBEIR dataset and prepare an ordered corpus and a lookup mapping.

    Returns
    -------
        corpus: Mapping of document ID to text.
        queries: Mapping of query ID to text.
        qrels: Mapping of query ID to document scores.
        ordered_corpus_texts: List of texts in corpus order.
        doc_text_to_id: Mapping from document text to document ID.

    """
    corpus_ds = load_dataset(hf_dataset_id, "corpus", split="train")
    queries_ds = load_dataset(hf_dataset_id, "queries", split="train")
    qrels_ds = load_dataset(hf_dataset_id, "qrels", split="train")

    corpus = {sample["_id"]: sample["text"] for sample in corpus_ds if sample["text"].strip()}
    queries = {sample["_id"]: sample["text"] for sample in queries_ds if sample["text"].strip()}
    qrels: Dict[str, Dict[str, int]] = {}
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
    ordered_corpus_texts: List[str],
    encoder: StaticModel | None,
    reranker: CrossEncoderReranker | None,
    alpha_value: float,
    use_bm25: bool,
) -> Pipeline:
    """
    Build and return a retrieval pipeline.

    Args:
    ----
        ordered_corpus_texts: List of corpus texts.
        encoder: Encoder model.
        reranker: Optional cross-encoder reranker.
        alpha_value: Alpha value for the pipeline.
        use_bm25: Whether to use BM25 for sparse retrieval.

    Returns:
    -------
        A Pipeline instance configured to use BM25 if use_bm25 is True.

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
    queries: Dict[str, str],
    doc_text_to_id: Dict[str, str],
    k: int,
    k_reranker: int,
) -> Tuple[List[List[str]], List[List[float]], List[str]]:
    """
    Retrieve ranked document IDs and scores for each query.

    Returns
    -------
        all_ranked_results: List of ranked document IDs per query.
        all_scores: List of score lists corresponding to the documents.
        query_ids: List of query IDs.

    """
    all_ranked_results: List[List[str]] = []
    all_scores: List[List[float]] = []
    query_ids: List[str] = []

    for qid, qtext in queries.items():
        # Query returns a list per query; take the first result.
        hybrid_results = pipeline.query([qtext], k=k, k_reranker=k_reranker)[0]
        ranked_doc_ids: List[str] = []
        scores: List[float] = []
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
    qrels: Dict[str, Dict[str, int]],
    all_ranked_results: List[List[str]],
    all_scores: List[List[float]],
    query_ids: List[str],
    k_values: List[int],
) -> Dict[str, Any]:
    """
    Evaluate retrieval results using BEIR evaluation helpers.

    Returns
    -------
        Dictionary of evaluation metrics.

    """
    results_for_eval = postprocess_results_for_eval(all_ranked_results, np.array(all_scores), query_ids)
    ndcg, _map, recall, precision = evaluate(qrels, results_for_eval, k_values)
    return {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}


def main(
    encoder_model: str | None,
    reranker_model: str | None,
    alpha_value: float,
    k_reranker: int,
    save_path: str,
    use_bm25: bool,
) -> None:
    """Evaluate a retrieval pipeline on multiple NanoBEIR datasets."""
    dataset_name_to_id: Dict[str, str] = {
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

    # Build parts of the folder name.
    encoder_part = encoder_model.split("/")[-1].replace("_", "-") if encoder_model else ""
    bm25_part = "bm25" if use_bm25 else ""
    reranker_part = reranker_model.split("/")[-1].replace("_", "-") if reranker_model else ""

    # Only include non-empty parts.
    parts = [part for part in (encoder_part, bm25_part, reranker_part) if part]
    base_name = "_".join(parts)

    # Append the common suffix.
    save_folder = f"{base_name}_alpha{alpha_value}_kr{k_reranker}"
    output_dir = Path(save_path) / save_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to folder: {output_dir}")

    # Save configuration.
    config = {
        "encoder_model": encoder_model,
        "reranker_model": reranker_model,
        "alpha_value": alpha_value,
        "k_reranker": k_reranker,
        "bm25": use_bm25,
        "output_dir": str(output_dir),
        "dataset_name_to_id": dataset_name_to_id,
    }
    config_path = output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved configuration to {config_path}")

    # Initialize models.
    encoder = StaticModel.from_pretrained(encoder_model) if encoder_model else None
    reranker = CrossEncoderReranker(reranker_model) if reranker_model else None

    k_values: List[int] = [1, 3, 5, 10, 100]
    all_metrics: Dict[str, Dict[str, Any]] = {}
    total_queries = total_time = total_fit_time = 0.0
    dataset_count = 0

    for ds_name, hf_id in dataset_name_to_id.items():
        try:
            logger.info(f"=== Processing dataset: {ds_name} ===")
            corpus, queries, qrels, ordered_texts, doc_text_to_id = load_and_prepare_dataset(hf_id)
            logger.info(f"Loaded corpus with {len(corpus)} documents and {len(queries)} queries.")
            k = len(ordered_texts)  # Use full corpus size

            fit_start = time.time()
            pipeline = build_hybrid_pipeline(ordered_texts, encoder, reranker, alpha_value, use_bm25)
            fit_time = time.time() - fit_start
            logger.info(f"Pipeline fitted in {fit_time:.4f} seconds.")
            total_fit_time += fit_time

            start_time = time.time()
            all_ranked_results, all_scores, query_ids = retrieve_query_results(
                pipeline, queries, doc_text_to_id, k, k_reranker
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

            results_path = output_dir / f"{ds_name}_results.json"
            with results_path.open("w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved results for {ds_name}.")

        except Exception as e:
            logger.error(f"Error processing {ds_name}: {e}")

    overall_qps = total_queries / total_time if total_time > 0 else 0.0
    avg_fit_time = total_fit_time / dataset_count if dataset_count > 0 else 0.0

    # Compute aggregated scores.
    aggregated_scores: Dict[str, Dict[str, float]] = {}
    for mtype in ["ndcg", "map", "recall", "precision"]:
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
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
    overall_save_path = output_dir / "overall_results.json"
    with overall_save_path.open("w") as f:
        json.dump(overall_eval, f, indent=4)
    logger.info(f"Saved overall results to {overall_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a retrieval pipeline on multiple NanoBEIR datasets.")
    parser.add_argument(
        "--encoder_model",
        type=str,
        default=None,
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
        help="Alpha value for the pipeline (0 <= alpha <= 1).",
    )
    parser.add_argument(
        "--k_reranker",
        type=int,
        required=True,
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
    args = parser.parse_args()

    main(
        encoder_model=args.encoder_model,
        reranker_model=args.reranker_model,
        alpha_value=args.alpha_value,
        k_reranker=args.k_reranker,
        save_path=args.save_path,
        use_bm25=args.bm25,
    )
