from pathlib import Path
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import HTMLResponse
from app.search import SearchIndex

def create_app(corpus=None):
    api=FastAPI(title='Document Search Benchmark')
    path=corpus or Path(__file__).parent.parent/'data/documents.json'
    index=SearchIndex(path)
    @api.get('/',response_class=HTMLResponse)
    def home():return (Path(__file__).parent/'index.html').read_text()
    @api.get('/search')
    def search(q:str=Query(min_length=1,max_length=1000),method:str='keyword',limit:int=Query(3,ge=1,le=20)):
        try:return {'method':method,'results':index.search(q,method,limit)}
        except ValueError as exc:raise HTTPException(400,str(exc)) from exc
    return api
app=create_app()
