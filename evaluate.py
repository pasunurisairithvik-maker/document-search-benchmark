import json
from pathlib import Path
from app.search import SearchIndex

def evaluate(corpus='data/documents.json',queries_path='data/queries.json',output='results'):
    index=SearchIndex(corpus);queries=json.loads(Path(queries_path).read_text());details=[];metrics={}
    for method in ['keyword','latent']:
        hits1=hits3=0;reciprocal=0
        for item in queries:
            ids=[r['id'] for r in index.search(item['query'],method,limit=len(index.docs))]
            rank=ids.index(item['relevant_id'])+1 if item['relevant_id'] in ids else None
            hits1+=rank==1;hits3+=rank is not None and rank<=3;reciprocal+=1/rank if rank else 0
            details.append({'method':method,'query':item['query'],'relevant_id':item['relevant_id'],'rank':rank,'top3':ids[:3]})
        count=len(queries)
        metrics[method]={'queries':count,'recall_at_1':hits1/count,'recall_at_3':hits3/count,'mean_reciprocal_rank':reciprocal/count}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report={'metrics':metrics,'documents':len(index.docs),'latent_dimensions':index.svd.n_components,'seed':42,'limitations':['20 authored demo documents and 20 authored queries; not independent or representative real-world evaluation.','One relevant document per query; labels are illustrative.','Latent semantic analysis is not a pretrained language model; it cannot understand arbitrary unseen vocabulary.','No claim of statistical significance, novelty, or superiority over production search.']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'query_results.json').write_text(json.dumps(details,indent=2)+'\n')
    lines=['# Document retrieval experiment','', 'Question: does an 8-dimensional latent index improve retrieval over TF-IDF on this tiny authored corpus?','', '| Method | Recall@1 | Recall@3 | MRR |','|---|---|---|---|']
    for method,m in metrics.items():lines.append(f"| {method} | {m['recall_at_1']:.3f} | {m['recall_at_3']:.3f} | {m['mean_reciprocal_rank']:.3f} |")
    lines.extend(['','The experiment reports both methods, including weaker results; no model is selected using these labels. Corpus TF-IDF and SVD are fitted without queries. This is exploratory demonstration data, not a held-out industrial benchmark.','','## Limitations']+report['limitations']+['','## Student analysis','Inspect query_results.json and explain two cases where rankings differ. Create an independent query set before tuning retrieval settings.','','## Method reference','Latent semantic analysis reduces a TF-IDF document matrix with TruncatedSVD; vectors are normalized before similarity search. https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return report
if __name__=='__main__':print(json.dumps(evaluate(),indent=2))
