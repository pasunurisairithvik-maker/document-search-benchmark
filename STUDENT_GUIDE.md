# Learn the retrieval pipeline
Read data/documents.json, app/search.py, evaluate.py and app/main.py in order. Explain TF-IDF, a vector, cosine similarity, truncated SVD, Recall@k and MRR. Inspect the side-by-side rankings and results/query_results.json.

## Two independent extensions
- Add BM25 as a third lexical baseline, document tokenization and compare it on a NEW independent query set.
- Replace/add LSA with a pretrained sentence encoder, report model name/license, CPU latency and retrieval metrics. Keep the lexical baseline and disclose the model's source.

The current implementation does NOT contain a pretrained sentence encoder. Do not claim one on a resume.

## Interview questions
Why can keyword search beat semantic search? Why doesn't a vector similarity equal confidence? What changes if a query contains no corpus vocabulary? What is the difference between retrieval and answer generation? Why can hand-written queries overestimate practical quality? How would you prevent sensitive documents leaking through search?

For an experimental report: define a new question, use independent labels, record methods before inspecting results, analyze errors and report weaknesses. A repository report is not a peer-reviewed publication.
