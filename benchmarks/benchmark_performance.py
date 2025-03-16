import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from bm25s.utils.beir import evaluate, postprocess_results_for_eval
from datasets import load_dataset
from sentence_transformers import CrossEncoder

from benchmarks.utils import build_save_folder_name, initialize_models, save_json
from korok import Pipeline
from korok.utils import Encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    hf_dataset_id: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]], list[str], dict[str, str]]:
    """Load a NanoBEIR dataset and prepare an ordered corpus and a lookup mapping."""
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
    reranker: CrossEncoder | None,
    alpha_value: float,
    use_bm25: bool,
) -> Pipeline:
    """Build and return a retrieval pipeline."""
    return Pipeline.fit(
        texts=ordered_corpus_texts, encoder=encoder, use_bm25=use_bm25, reranker=reranker, alpha=alpha_value
    )


def retrieve_query_results(
    pipeline: Pipeline, queries: dict[str, str], doc_text_to_id: dict[str, str], k: int, k_reranker: int
) -> tuple[list[list[str]], list[list[float]], list[str]]:
    """Retrieve ranked document IDs and scores for each query."""
    all_ranked_results: list[list[str]] = []
    all_scores: list[list[float]] = []
    query_ids: list[str] = []
    for qid, qtext in queries.items():
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
    """Evaluate retrieval results using BEIR evaluation helpers."""
    max_len = max(len(scores) for scores in all_scores)
    padded_scores = [scores + [0.0] * (max_len - len(scores)) for scores in all_scores]
    results_for_eval = postprocess_results_for_eval(all_ranked_results, np.array(padded_scores), query_ids)
    ndcg, mean_avg_precision, recall, precision = evaluate(qrels, results_for_eval, k_values)
    return {"ndcg": ndcg, "mean_avg_precision": mean_avg_precision, "recall": recall, "precision": precision}


def process_dataset(
    ds_name: str,
    hf_id: str,
    encoder: Encoder | None,
    reranker: CrossEncoder | None,
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
        k = len(ordered_texts)
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
    """Evaluate a retrieval pipeline on multiple NanoBEIR datasets."""
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
    encoder, reranker = initialize_models(encoder_model, reranker_model, device)
    save_folder = build_save_folder_name(encoder_model, use_bm25, reranker_model, alpha_value, k_reranker)
    output_dir = Path(save_path) / save_folder
    if output_dir.exists() and not overwrite_results:
        logger.info(f"Output folder '{output_dir}' already exists and overwrite_results is False. Skipping evaluation.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to folder: {output_dir}")
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
        "--encoder-model",
        type=str,
        default=None,
        help="Pretrained encoder model name or path (e.g., 'minishlab/potion-retrieval-32M').",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Optional reranker model name or path (e.g., 'BAAI/bge-reranker-v2-m3').",
    )
    parser.add_argument(
        "--alpha-value", type=float, default=0.5, help="Alpha value for the pipeline (0 <= alpha <= 1)."
    )
    parser.add_argument("--k-reranker", type=int, default=30, help="Number of top documents to re-rank.")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save results.")
    parser.add_argument(
        "--bm25", action="store_true", help="If set, BM25 (sparse retrieval) is used for hybrid search."
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        default=False,
        help="If set, overwrite results even if the save folder already exists.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference.")
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
