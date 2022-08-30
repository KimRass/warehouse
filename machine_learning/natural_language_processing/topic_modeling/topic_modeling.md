# Topic Modeling
- Reference: https://wikidocs.net/30707
- 토픽(Topic)은 한국어로는 주제라고 합니다. 토픽 모델링(Topic Modeling)이란 기계 학습 및 자연어 처리 분야에서 토픽이라는 문서 집합의 추상적인 주제를 발견하기 위한 통계적 모델 중 하나로, 텍스트 본문의 숨겨진 의미 구조를 발견하기 위해 사용되는 텍스트 마이닝 기법입니다.
## LSA (Latent Semantic Analysis)
- Reference: https://wikidocs.net/24949
- LSA는 정확히는 토픽 모델링을 위해 최적화 된 알고리즘은 아니지만, 토픽 모델링이라는 분야에 아이디어를 제공한 알고리즘이라고 볼 수 있습니다.
- LSA는 기본적으로 DTM이나 TF-IDF 행렬에 절단된 truncated SVD를 사용하여 차원을 축소시키고, 단어들의 잠재적인 의미를 끌어낸다는 아이디어를 갖고 있습니다.
## LDA (Latent Dirichlet Allocation)
```python
model = gensim.models.ldamodel.LdaModel(dtm, num_topics=n_topics, id2word=id2word, alpha="auto", eta="auto")
```
- Visualization
	```python
	import pyLDAvis
	pyLDAvis.enable_notebook()`
	
	pyldavis = pyLDAvis.gensim.prepare(model, dtm, id2word)
	```
## Contextualized Topic Models
### Combined Topic Modeling (CTM)
- Reference: https://wikidocs.net/161310