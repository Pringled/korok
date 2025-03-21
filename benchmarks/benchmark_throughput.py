import argparse
import logging
import random
import time
from pathlib import Path

from datasets import load_dataset

from benchmarks.utils import build_save_folder_name, initialize_models, save_json
from korok import Pipeline

random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_corpus(dataset_path: str, dataset_name: str, dataset_split: str, dataset_column: str) -> list[str]:
    """Load a dataset using Hugging Face datasets and return a list of texts from the specified column."""
    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
    corpus_texts = [
        sample[dataset_column] for sample in dataset if sample[dataset_column] and sample[dataset_column].strip()
    ]
    return corpus_texts


def main(
    encoder_model: str | None,
    reranker_model: str | None,
    alpha_value: float,
    k_reranker: int,
    save_path: str,
    use_bm25: bool,
    instruction: str | None,
    overwrite_results: bool,
    device: str | None,
    num_queries: int,
    max_documents: int | None,
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
    dataset_column: str,
) -> None:
    """
    Throughput benchmark using a configurable dataset.

    Loads the specified corpus (optionally capped), fits a retrieval pipeline,
    runs a set number of queries to measure throughput, and saves the results.
    """
    save_folder = build_save_folder_name(encoder_model, use_bm25, reranker_model, alpha_value, k_reranker, instruction)
    output_dir = Path(save_path) / save_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using throughput results folder: {output_dir}")

    # Check if throughput_results.json already exists
    results_path = output_dir / "throughput_results.json"
    if results_path.exists() and not overwrite_results:
        logger.info(f"Throughput results already exist at {results_path}. Skipping benchmark.")
        return

    corpus_texts = load_corpus(dataset_path, dataset_name, dataset_split, dataset_column)
    logger.info(
        f"Loaded corpus '{dataset_name}' from '{dataset_path}' split '{dataset_split}' "
        f"with {len(corpus_texts)} texts."
    )
    if max_documents is not None:
        corpus_texts = corpus_texts[:max_documents]
        logger.info(f"Capped corpus to {len(corpus_texts)} documents due to max_documents parameter.")

    encoder, reranker = initialize_models(encoder_model, reranker_model, device)
    logger.info("Fitting pipeline...")
    fit_start = time.perf_counter()
    pipeline = Pipeline.fit(
        texts=corpus_texts, encoder=encoder, use_bm25=use_bm25, reranker=reranker, alpha=alpha_value
    )
    fit_time = time.perf_counter() - fit_start
    logger.info(f"Pipeline fitted in {fit_time:.4f} seconds.")

    num_queries = min(num_queries, len(corpus_texts))
    queries = random.sample(corpus_texts, num_queries)
    logger.info(f"Selected {len(queries)} queries for throughput testing.")

    logger.info("Running queries for throughput measurement...")
    query_start = time.perf_counter()
    for q in queries:
        _ = pipeline.query([q], k=len(corpus_texts), k_reranker=k_reranker, instruction=instruction)
    total_query_time = time.perf_counter() - query_start
    logger.info(f"Processed {len(queries)} queries in {total_query_time:.4f} seconds.")

    qps = len(queries) / total_query_time if total_query_time > 0 else 0.0
    throughput_results = {
        "pipeline_fit_time": fit_time,
        "total_query_time": total_query_time,
        "num_queries": len(queries),
        "qps": qps,
    }
    save_json(throughput_results, results_path)
    logger.info(f"Saved throughput results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Throughput benchmark using a configurable dataset.")
    parser.add_argument(
        "--encoder-model",
        type=str,
        default=None,
        help="Encoder model name or path (e.g., 'minishlab/potion-retrieval-32M').",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Reranker model name or path (e.g., 'cross-encoder/ms-marco-MiniLM-L6-v2').",
    )
    parser.add_argument("--alpha-value", type=float, default=0.5, help="Alpha value (0 <= alpha <= 1).")
    parser.add_argument("--k-reranker", type=int, default=30, help="Number of top documents to re-rank.")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--use-bm25", action="store_true", help="Use BM25 for hybrid search.")
    parser.add_argument("--instruction", type=str, default=None, help="Optional instruction for the queries.")
    parser.add_argument(
        "--overwrite-results", action="store_true", default=False, help="Overwrite results if folder exists."
    )
    parser.add_argument("--device", type=str, default=None, help="Device for inference (e.g., 'cpu', 'cuda', 'mps').")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries for throughput test.")
    parser.add_argument("--max-documents", type=int, default=None, help="Cap dataset to max number of documents.")
    parser.add_argument("--dataset-path", type=str, default="wikitext", help="Dataset path (default: 'wikitext').")
    parser.add_argument(
        "--dataset-name", type=str, default="wikitext-103-raw-v1", help="Dataset name (default: 'wikitext-103-raw-v1')."
    )
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split (default: 'train').")
    parser.add_argument("--dataset-column", type=str, default="text", help="Column name for text (default: 'text').")
    args = parser.parse_args()

    main(
        encoder_model=args.encoder_model,
        reranker_model=args.reranker_model,
        alpha_value=args.alpha_value,
        k_reranker=args.k_reranker,
        save_path=args.save_path,
        use_bm25=args.use_bm25,
        instruction=args.instruction,
        overwrite_results=args.overwrite_results,
        device=args.device,
        num_queries=args.num_queries,
        max_documents=args.max_documents,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_column=args.dataset_column,
    )
