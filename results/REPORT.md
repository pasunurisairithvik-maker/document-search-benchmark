# Document retrieval experiment

Question: does an 8-dimensional latent index improve retrieval over TF-IDF on this tiny authored corpus?

| Method | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| keyword | 0.650 | 0.900 | 0.768 |
| latent | 0.600 | 0.850 | 0.739 |

The experiment reports both methods, including weaker results; no model is selected using these labels. Corpus TF-IDF and SVD are fitted without queries. This is exploratory demonstration data, not a held-out industrial benchmark.

## Limitations
20 authored demo documents and 20 authored queries; not independent or representative real-world evaluation.
One relevant document per query; labels are illustrative.
Latent semantic analysis is not a pretrained language model; it cannot understand arbitrary unseen vocabulary.
No claim of statistical significance, novelty, or superiority over production search.

## Student analysis
Inspect query_results.json and explain two cases where rankings differ. Create an independent query set before tuning retrieval settings.

## Method reference
Latent semantic analysis reduces a TF-IDF document matrix with TruncatedSVD; vectors are normalized before similarity search. https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis
