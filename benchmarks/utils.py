import json
from pathlib import Path
from typing import Any

from model2vec import StaticModel
from sentence_transformers import CrossEncoder, SentenceTransformer

from korok.utils import Encoder


def initialize_models(
    encoder_model: str | None, reranker_model: str | None, device: str | None
) -> tuple[Encoder | None, CrossEncoder | None]:
    """Initialize and return the encoder and reranker models."""
    if not encoder_model:
        encoder = None
    elif encoder_model == "minishlab/potion-retrieval-32M":
        encoder = StaticModel.from_pretrained(encoder_model)
    else:
        encoder = SentenceTransformer(encoder_model, trust_remote_code=True, device=device)
    reranker = CrossEncoder(reranker_model, trust_remote_code=True, device=device) if reranker_model else None
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
