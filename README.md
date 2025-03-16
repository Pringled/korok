# Korok

Korok is a lightweight vector search and reranking package.

Korok supports three different types of search:
- Dense vector search
- Sparse vector search
- Hybrid search

## Installation

To install the package, run the following command:

```bash
make install
```


## Quickstart


### Dense Vector Search
The following code snippet shows how to use Korok for dense vector search:

```python
from model2vec import StaticModel
from korok import Pipeline

# Create texts to encode
texts = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]

# Initialize the encoder and pipeline
encoder = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
pipeline = Pipeline.fit(texts=texts, encoder=encoder)

# Query for nearest neighbors
query_text = "sword"
results = pipeline.query([query_text], k=3)
```

### Sparse Vector Search
The following code snippet shows how to use Korok for sparse vector search:

```python
from korok import Pipeline

# Create texts to encode
texts = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]

# Initialize the encoder and pipeline
pipeline = Pipeline.fit(texts=texts, use_bm25=True)

# Query for nearest neighbors
query_text = "sword"
results = pipeline.query([query_text], k=3)
```

### Hybrid Vector Search
The following code snippet shows how to use Korok for hybrid search:

```python
from model2vec import StaticModel
from korok import Pipeline

# Create texts to encode
texts = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot", "spear"]

# Initialize the encoder and pipeline
encoder = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
pipeline = Pipeline.fit(texts=texts, encoder=encoder, use_bm25=True)

# Query for nearest neighbors
query_text = "sword"
results = pipeline.query([query_text], k=3)
```

### Rerankers
To use a reranker, simply pass the reranker to the pipeline:

```python
from model2vec import StaticModel
from korok import Pipeline
from sentence_transformers import CrossEncoder

# Create texts to encode
texts = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot", "spear"]

# Initialize the encoder and pipeline
encoder = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
pipeline = Pipeline.fit(texts=texts, encoder=encoder, use_bm25=True, reranker=reranker)

# Query for nearest neighbors
query_text = "sword"
results = pipeline.query([query_text], k=3)
```
