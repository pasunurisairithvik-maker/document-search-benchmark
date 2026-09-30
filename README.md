# Document Search Benchmark

Release 1.0 within its documented scope. See [release and operating notes](RELEASE.md).
A small Python document-search application comparing lexical TF-IDF with latent semantic analysis (LSA). The browser shows both rankings side by side. An evaluation script reports Recall@1, Recall@3, mean reciprocal rank, and per-query results, including failures.

## Run: Python 3.11+
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8003
```
Open http://127.0.0.1:8003 or /docs. Two examples: "worker crashed while processing task" and "charged twice for an order". The results refer to fictional documents, not actual Amazon policies or services.

## What semantic search means here
LSA learns compact vectors from word co-occurrence using an 8-dimensional truncated SVD of the corpus TF-IDF matrix. This is a small classical semantic method, NOT a pretrained embedding model or LLM. Unseen vocabulary cannot be understood. Dimensionality reduction can make results worse; compare the measured results rather than assume improvement.

## Evidence
Run `python -m unittest discover -s tests -v`. Read results/REPORT.md, metrics.json and query_results.json. Twenty authored documents and twenty authored queries are bundled; the examples are not independent real-world evidence. Corpus fitting does not use query labels, and settings are fixed rather than tuned against the published results. Similarity scores are not confidence probabilities. No generative answers, external API key, GPU, model download or AWS account is required.

## Limits and attribution
Local learning demo; no authentication, persistence for uploaded documents, multilingual evaluation, or production-scale search. Sample corpus and labels were authored with AI assistance for this project. They are fictional and should not be treated as operational, financial, privacy or security advice. No research novelty is claimed.

Implementation references: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction and https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis
Read STUDENT_GUIDE.md and contribute an independently understood improvement before describing your own project work.
