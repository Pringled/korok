import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(folder: Path) -> tuple[dict, dict]:
    """
    Load the configuration and overall results from the given folder.

    :param folder: Path to the folder containing config.json and overall_results.json.
    :return: A tuple containing the configuration and overall results as dictionaries.
    """
    config_file = folder / "config.json"
    overall_file = folder / "overall_results.json"

    with config_file.open("r") as f:
        config = json.load(f)
    with overall_file.open("r") as f:
        overall = json.load(f)
    return config, overall


def generate_markdown_row(config: dict, overall: dict) -> str:  # noqa: C901
    """
    Generate a markdown table row from the config and overall results.

    The row includes:
      - Encoder Model
      - Reranker Model
      - BM25 flag
      - NDCG@10
      - MAP@10
      - Recall@10
      - Precision@10
      - QPS

    :param config: The configuration dictionary loaded from config.json.
    :param overall: The overall results dictionary loaded from overall_results.json.
    :return: A string containing a single markdown table row.
    """
    encoder_model = config.get("encoder_model", "N/A")
    reranker_model = config.get("reranker_model", "N/A")
    bm25 = config.get("bm25", False)

    aggregated_scores = overall.get("aggregated_scores", {})

    # Convert score values to percentages inline.
    ndcg_raw = aggregated_scores.get("ndcg", {}).get("NDCG@10", None)
    if ndcg_raw is not None:
        try:
            ndcg_at_10 = f"{float(ndcg_raw)*100:.2f}"
        except (TypeError, ValueError):
            ndcg_at_10 = "N/A"
    else:
        ndcg_at_10 = "N/A"

    map_raw = aggregated_scores.get("map", {}).get("MAP@10", None)
    if map_raw is not None:
        try:
            map_at_10 = f"{float(map_raw)*100:.2f}"
        except (TypeError, ValueError):
            map_at_10 = "N/A"
    else:
        map_at_10 = "N/A"

    recall_raw = aggregated_scores.get("recall", {}).get("Recall@10", None)
    if recall_raw is not None:
        try:
            recall_at_10 = f"{float(recall_raw)*100:.2f}"
        except (TypeError, ValueError):
            recall_at_10 = "N/A"
    else:
        recall_at_10 = "N/A"

    precision_raw = aggregated_scores.get("precision", {}).get("P@10", None)
    if precision_raw is not None:
        try:
            precision_at_10 = f"{float(precision_raw)*100:.2f}"
        except (TypeError, ValueError):
            precision_at_10 = "N/A"
    else:
        precision_at_10 = "N/A"

    throughput = overall.get("throughput", {})
    qps_raw = throughput.get("qps", None)
    if qps_raw is not None:
        try:
            qps = f"{float(qps_raw):.2f}"
        except (TypeError, ValueError):
            qps = "N/A"
    else:
        qps = "N/A"

    row = (
        f"| {encoder_model} | {reranker_model} | {bm25} | {ndcg_at_10} | {map_at_10} | "
        f"{recall_at_10} | {precision_at_10} | {qps} |\n"
    )
    return row


def main() -> None:
    """Print a markdown table of benchmark results from a results folder."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown table from subfolders of a results folder. "
        "Each subfolder must contain config.json and overall_results.json."
    )
    parser.add_argument(
        "--results-folder", type=str, required=True, help="Path to the folder containing subfolders with results."
    )
    args = parser.parse_args()

    results_folder = Path(args.results_folder)
    subfolders = [
        d
        for d in results_folder.iterdir()
        if d.is_dir() and (d / "config.json").exists() and (d / "overall_results.json").exists()
    ]

    if not subfolders:
        logger.warning("No valid result subfolders found in the specified folder.")
        return

    rows_with_ndcg: list[tuple[float, str]] = []
    for subfolder in subfolders:
        try:
            config, overall = load_results(subfolder)
            # Get NDCG@10 and ensure it's a float for proper sorting.
            ndcg_value = overall.get("aggregated_scores", {}).get("ndcg", {}).get("NDCG@10", 0)
            try:
                ndcg_value = float(ndcg_value)
            except (TypeError, ValueError):
                ndcg_value = 0.0

            row = generate_markdown_row(config, overall)
            rows_with_ndcg.append((ndcg_value, row))
        except Exception as e:
            logger.error(f"Error processing folder {subfolder.name}: {e}")

    # Sort rows by NDCG@10 in descending order.
    rows_with_ndcg.sort(key=lambda x: x[0], reverse=True)

    header = "| Encoder Model | Reranker Model | BM25 | NDCG@10 | MAP@10 | Recall@10 | Precision@10 | QPS |\n"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    table = header + separator

    for _, row in rows_with_ndcg:
        table += row

    print(table)  # noqa: T201


if __name__ == "__main__":
    main()
