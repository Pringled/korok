# Benchmark results

| Encoder Model | Reranker Model | BM25 | NDCG@10 | MAP@10 | Recall@10 | Precision@10 | QPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentence-transformers/all-MiniLM-L6-v2 | BAAI/bge-reranker-v2-m3 | True | 68.36 | 56.77 | 68.31 | 20.69 | 0.21 |
| ibm-granite/granite-embedding-30m-english | BAAI/bge-reranker-v2-m3 | True | 67.94 | 56.49 | 67.80 | 20.31 | 0.17 |
| minishlab/potion-retrieval-32M | BAAI/bge-reranker-v2-m3 | True | 67.89 | 56.62 | 67.32 | 20.27 | 0.22 |
| None | BAAI/bge-reranker-v2-m3 | True | 65.88 | 55.04 | 64.09 | 19.58 | 0.23 |
| ibm-granite/granite-embedding-30m-english | None | True | 64.16 | 51.73 | 66.89 | 20.32 | 110.64 |
| sentence-transformers/all-MiniLM-L6-v2 | None | True | 62.28 | 49.77 | 65.00 | 20.01 | 115.82 |
| minishlab/potion-retrieval-32M | None | True | 57.37 | 44.50 | 61.21 | 19.31 | 1546.47 |
| None | None | True | 55.96 | 43.40 | 59.42 | 18.90 | 2774.99 |
| sentence-transformers/all-MiniLM-L6-v2 | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 18.99 | 9.00 | 28.31 | 10.51 | 2.85 |
| minishlab/potion-retrieval-32M | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 18.53 | 8.77 | 27.39 | 10.34 | 3.14 |
| ibm-granite/granite-embedding-30m-english | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 18.51 | 8.65 | 27.70 | 10.23 | 3.21 |
| None | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 17.83 | 8.62 | 24.93 | 10.02 | 3.15 |
