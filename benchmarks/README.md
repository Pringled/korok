# Benchmark results

| Encoder Model | Reranker Model | BM25 | NDCG@10 | MAP@10 | Recall@10 | Precision@10 | QPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minishlab/potion-retrieval-32M | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 0.6295530769230768 | 0.5063669230769231 | 0.6493146153846154 | 0.19836384615384617 | 8.950420752782575 |
| minishlab/potion-retrieval-32M | cross-encoder/ms-marco-MiniLM-L6-v2 | False | 0.6155253846153846 | 0.4976630769230769 | 0.6279138461538462 | 0.18985230769230774 | 9.367961902984023 |
| None | cross-encoder/ms-marco-MiniLM-L6-v2 | True | 0.6119523076923078 | 0.492556923076923 | 0.6216807692307692 | 0.1929892307692308 | 9.050031038400487 |
| None | None | True | 0.5595930769230769 | 0.4340146153846154 | 0.59422 | 0.1889923076923077 | 2766.5703817458 |
| minishlab/potion-retrieval-32M | None | False | 0.5089761538461539 | 0.38437538461538456 | 0.5656638461538462 | 0.1708946153846154 | 4923.8982625919125 |
