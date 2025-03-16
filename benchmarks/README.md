# Benchmarks

The benchmarks in this directory evaluate the performance and throughput of different encoder and reranker models for dense,  sparse, and hybrid vector search. The benchmarks are all run on the [NanoBEIR datasets](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6b1b3b3b3) Throughput is measured on a part of the [wikitext dataset](https://huggingface.co/datasets/Salesforce/wikitext). The benchmarks were run with the following setup:
- The throughput benchmarks were all run on CPU
- The K value for the reranker is set to 30
- The alpha value for hybrid search is set to 0.5

## Results

| Encoder Model | Reranker Model | BM25 | NDCG@10 | MAP@10 | Recall@10 | Precision@10 |
| --- | --- | --- | --- | --- | --- | --- |
| sentence-transformers/all-MiniLM-L6-v2 | BAAI/bge-reranker-v2-m3 | True | 68.36 | 56.77 | 68.31 | 20.69 |
| sentence-transformers/all-MiniLM-L6-v2 | jinaai/jina-reranker-v2-base-multilingual | True | 68.03 | 55.78 | 68.77 | 21.03 |
| ibm-granite/granite-embedding-30m-english | BAAI/bge-reranker-v2-m3 | True | 67.94 | 56.49 | 67.80 | 20.31 |
| minishlab/potion-retrieval-32M | BAAI/bge-reranker-v2-m3 | True | 67.89 | 56.62 | 67.32 | 20.27 |
| ibm-granite/granite-embedding-30m-english | jinaai/jina-reranker-v2-base-multilingual | True | 67.76 | 55.53 | 68.96 | 20.75 |
| minishlab/potion-retrieval-32M | jinaai/jina-reranker-v2-base-multilingual | True | 67.44 | 55.35 | 67.95 | 20.75 |
| ibm-granite/granite-embedding-30m-english | BAAI/bge-reranker-v2-m3 | False | 67.30 | 56.05 | 67.72 | 19.82 |
| ibm-granite/granite-embedding-30m-english | jinaai/jina-reranker-v2-base-multilingual | False | 66.69 | 54.93 | 67.89 | 19.99 |
| minishlab/potion-retrieval-32M | BAAI/bge-reranker-v2-m3 | False | 66.27 | 55.28 | 64.72 | 19.72 |
| sentence-transformers/all-MiniLM-L6-v2 | BAAI/bge-reranker-v2-m3 | False | 66.26 | 54.93 | 65.72 | 19.62 |
| None | BAAI/bge-reranker-v2-m3 | True | 65.88 | 55.04 | 64.09 | 19.58 |
| None | jinaai/jina-reranker-v2-base-multilingual | True | 65.51 | 53.89 | 64.54 | 19.96 |
| sentence-transformers/all-MiniLM-L6-v2 | jinaai/jina-reranker-v2-base-multilingual | False | 65.47 | 53.63 | 66.03 | 19.82 |
| minishlab/potion-retrieval-32M | jinaai/jina-reranker-v2-base-multilingual | False | 65.29 | 53.76 | 64.93 | 19.81 |
| ibm-granite/granite-embedding-30m-english | None | True | 64.16 | 51.73 | 66.89 | 20.32 |
| sentence-transformers/all-MiniLM-L6-v2 | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 63.33 | 50.91 | 65.68 | 20.07 |
| ibm-granite/granite-embedding-30m-english | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 63.06 | 50.75 | 65.28 | 19.90 |
| minishlab/potion-retrieval-32M | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 62.96 | 50.64 | 64.93 | 19.84 |
| ibm-granite/granite-embedding-30m-english | cross-encoder/ms-marco-MiniLM-L6-v2 | False | 62.44 | 50.43 | 64.12 | 19.18 |
| sentence-transformers/all-MiniLM-L6-v2 | None | True | 62.28 | 49.77 | 65.00 | 20.01 |
| sentence-transformers/all-MiniLM-L6-v2 | cross-encoder/ms-marco-MiniLM-L6-v2 | False | 61.58 | 49.47 | 63.46 | 19.19 |
| minishlab/potion-retrieval-32M | cross-encoder/ms-marco-MiniLM-L6-v2 | False | 61.55 | 49.77 | 62.79 | 18.99 |
| None | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 61.20 | 49.26 | 62.17 | 19.30 |
| ibm-granite/granite-embedding-30m-english | None | False | 60.47 | 49.33 | 63.21 | 17.63 |
| minishlab/potion-retrieval-32M | None | True | 57.37 | 44.50 | 61.21 | 19.31 |
| sentence-transformers/all-MiniLM-L6-v2 | None | False | 56.55 | 44.55 | 61.09 | 17.70 |
| None | None | True | 55.96 | 43.40 | 59.42 | 18.90 |
| minishlab/potion-retrieval-32M | None | False | 50.90 | 38.44 | 56.57 | 17.09 |

## Reproducibility

To reproduce the results, run the `benchmark_performance` and `benchmark_throughput` scripts with the config you want to reproduce. For example, to reproduce the results for the `minishlab/potion-retrieval-32M` encoder with the `cross-encoder/ms-marco-MiniLM-L6-v2` reranker and BM25 enabled, run the following commands:

```bash
python -m benchmarks.benchmark_performance --encoder-model "minishlab/potion-retrieval-32M" --bm25 --reranker-model "cross-encoder/ms-marco-MiniLM-L6-v2" --save-path "./results"
```

```bash
python -m benchmarks.benchmark_throughput --encoder-model "minishlab/potion-retrieval-32M" --bm25 --reranker-model "cross-encoder/ms-marco-MiniLM-L6-v2" --save-path "./results" --device "cpu" --max-samples 10000 --num-queries 100
```

Then, the results table can be printed by running the following command:

```bash
python -m benchmarks.print_benchmarks_table --results-path "./results"
```
