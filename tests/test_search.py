import json,tempfile,unittest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import create_app
from app.search import SearchIndex
from evaluate import evaluate
CORPUS=Path(__file__).parent.parent/'data/documents.json'
QUERIES=Path(__file__).parent.parent/'data/queries.json'
class SearchTests(unittest.TestCase):
    def setUp(self):self.index=SearchIndex(CORPUS)
    def test_keyword_exact_topic(self):self.assertEqual(self.index.search('HTTP 429 API rate limits')[0]['id'],'api-limits')
    def test_unknown_vocabulary(self):self.assertEqual(self.index.search('zzzzxxxxqqqq'),[])
    def test_latent_is_deterministic(self):
        self.assertEqual(self.index.search('password account',method='latent'),SearchIndex(CORPUS).search('password account',method='latent'))
    def test_invalid_queries(self):
        for q,method in [(' ','keyword'),('hello','wrong')]:
            with self.assertRaises(ValueError):self.index.search(q,method)
    def test_api(self):
        client=TestClient(create_app(CORPUS));self.assertEqual(client.get('/').status_code,200)
        for method in ['keyword','latent']:
            result=client.get('/search',params={'q':'password account','method':method})
            self.assertEqual(result.status_code,200);self.assertLessEqual(len(result.json()['results']),3)
        self.assertEqual(client.get('/search',params={'q':'hi','method':'bad'}).status_code,400)
        self.assertEqual(client.get('/search',params={'q':'hi','limit':0}).status_code,422)
    def test_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=evaluate(CORPUS,QUERIES,tmp)
            for method in ['keyword','latent']:
                for metric in ['recall_at_1','recall_at_3','mean_reciprocal_rank']:
                    self.assertTrue(0<=report['metrics'][method][metric]<=1)
            self.assertEqual(len(json.loads((Path(tmp)/'query_results.json').read_text())),40)
if __name__=='__main__':unittest.main()
