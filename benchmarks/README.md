# Benchmark results

| Encoder Model | Reranker Model | BM25 | NDCG@10 | MAP@10 | Recall@10 | Precision@10 | QPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minishlab/potion-retrieval-32M | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 62.96 | 50.64 | 64.93 | 19.84 | 8.95 |
| minishlab/potion-retrieval-32M | cross-encoder/ms-marco-MiniLM-L6-v2 | False | 61.55 | 49.77 | 62.79 | 18.99 | 9.37 |
| None | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 61.20 | 49.26 | 62.17 | 19.30 | 9.05 |
| None | None | True | 55.96 | 43.40 | 59.42 | 18.90 | 2766.57 |
| minishlab/potion-retrieval-32M | None | False | 50.90 | 38.44 | 56.57 | 17.09 | 4923.90 |
