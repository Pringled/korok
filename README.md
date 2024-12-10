# Korok

Korok is a lightweight hybrid search and reranking package.

## Installation

To install the package, run the following command:

```bash
make install
```


## Quickstart

The following code snippet demonstrates how to use Korok for nearest neighbor search with a Model2Vec encoder:

```python
from model2vec import StaticModel
from korok import Pipeline

# Texts to encode
texts = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]
# Initialize the encoder and pipeline
encoder = StaticModel.from_pretrained("minishlab/potion-base-8M")
pipeline = Pipeline.fit(texts=texts, encoder=encoder)

# Query for nearest neighbors
query_text = "sword"
results = pipeline.query([query_text], k=3)
```


## Usage

### Rerankers

The following code snippet demonstrates how to use Korok with a reranker:

```python
from model2vec import StaticModel
from korok import Pipeline
from korok.rerankers import CrossEncoderReranker

texts = ["triforce", "master sword", "hylian shield", "boomerang", "hookshot"]
encoder = StaticModel.from_pretrained("minishlab/potion-base-8M")
reranker = CrossEncoderReranker()
pipeline = Pipeline.fit(texts=texts, encoder=encoder, reranker=reranker)

query_text = "sword"
results = pipeline.query([query_text], k=3)
```
