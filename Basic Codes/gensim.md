# gensim
```python
import gensim
```
## gensim.corpora
### gensim.corpora.Dictionary()
```python
id2word = gensim.corpora.Dictionary(docs_tkn)
```
#### id2word.id2token
- dict(id2word)와 dict(id2word.id2token)은 서로 동일.
#### id2word.token2id
- dict(id2word.token2id)는 key와 value가 서로 반대.
#### id2word.doc2bow()
```python
dtm = [id2word.doc2bow(doc) for doc in docs_tkn]
```
#### gensim.corpora.Dictionary.load()
```python
id2word = gensim.corpora.Dictionary.load("kakaotalk id2word")
```
### gensim.corpora.BleiCorpus
#### gensim.corpora.BleiCorpus.serizalize()
```python
gensim.corpora.BleiCorpus.serialize("kakotalk dtm", dtm)
```
### gensim.corpora.bleicorpus
#### gensim.corpora.bleicorpus.BleiCorpus()
```python
dtm = gensim.corpora.bleicorpus.BleiCorpus("kakaotalk dtm")
```
## gensim.models
### gensim.models.TfidfModel()
```python
tfidf = gensim.models.TfidfModel(dtm)[dtm]
```
### gensim.models.AuthorTopicModel()
```python
model = gensim.models.AuthorTopicModel(corpus=dtm, id2word=id2word, num_topics=n_topics, author2doc=aut2doc, passes=1000)
```
### gensim.models.Word2Vec()
```python
model = gensim.models.Word2Vec(sentences, min_count=5, size=300, sg=1, iter=10, workers=4, ns_exponent=0.75, window=7)
```
### gensim.models.FastText()
```python
model = gensim.models.FastText(sentences, min_count=5, sg=1, size=300, workers=4, min_n=2, max_n=7, alpha=0.05, iter=10, window=7)
```
#### model.save()
```python
model.save("kakaotalk model")
```
#### model.show_topic()
```python
model.show_topic(1, topn=20)
```
#### model.wv.most_similar()
```python
model.wv.most_similar("good)
```
#### gensim.models.AuthorTopicModel.load()
```python
model = gensim.models.AuthorTopicModel.load("kakaotalk model")
```
## gensim.models.ldamodel.Ldamodel()
```python
model = gensim.models.ldamodel.LdaModel(dtm, num_topics=n_topics, id2word=id2word, alpha="auto", eta="auto")
```
### model.show_topic()
```python
model.show_topic(2, 10)
```
- arguments : topic의 index, 출력할 word 개수
