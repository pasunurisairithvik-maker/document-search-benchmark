"""Compare lexical TF-IDF retrieval with a small latent semantic index."""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class SearchIndex:
    def __init__(self,path):
        self.docs=json.loads(Path(path).read_text())
        if len(self.docs)<3:raise ValueError('Need at least three documents')
        ids=[d['id'] for d in self.docs]
        if len(ids)!=len(set(ids)):raise ValueError('Document IDs must be unique')
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
        self.lexical=self.vectorizer.fit_transform([d['title']+' '+d['text'] for d in self.docs])
        # Latent vectors are learned from the corpus alone, without query labels.
        self.svd=TruncatedSVD(n_components=min(8,len(self.docs)-1,self.lexical.shape[1]-1),random_state=42)
        self.latent=normalize(self.svd.fit_transform(self.lexical))
    def search(self,query,method='keyword',limit=3):
        if not isinstance(query,str) or not query.strip() or len(query)>1000:raise ValueError('Query needs 1-1000 nonblank characters')
        if method not in {'keyword','latent'}:raise ValueError('Use keyword or latent')
        if not 1<=limit<=len(self.docs):raise ValueError('Invalid result limit')
        vector=self.vectorizer.transform([query])
        if vector.nnz==0:return []
        if method=='keyword':scores=(self.lexical@vector.T).toarray().ravel()
        else:scores=(self.latent@normalize(self.svd.transform(vector)).T).ravel()
        ranked=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self.docs[i]['id']))
        return [{'id':self.docs[i]['id'],'title':self.docs[i]['title'],'text':self.docs[i]['text'],'score':round(float(scores[i]),6)} for i in ranked[:limit] if scores[i]>0]
