# Bag-of-Words Model
- Reference: https://en.wikipedia.org/wiki/Bag-of-words_model
- The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, ***a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.***
- *The bag-of-words model is commonly used in methods of document classification where the (frequency of) occurrence of each word is used as a feature for training a classifier.*
```python
corpus = ["먹고 싶은 사과", "먹고 싶은 바나나", "길고 노란 바나나 바나나", "저는 과일이 좋아요"]
```
- Implementation
	```python
	token2idx = {}
	bow = []
	i = 0
	for sent in corpus:
		for token in sent.split(" "):
			if token not in token2idx:
				token2idx[token] = i
				i += 1
				bow.append(1)
			else:
				bow[token2idx[token]] += 1
	```
## Document-Term Matrix (DTM)
- *A document-term matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents. In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms.* This matrix is a specific instance of a document-feature matrix where "features" may refer to other properties of a document besides terms. It is also common to encounter the transpose, or term-document matrix where documents are the columns and terms are the rows. They are useful in the field of natural language processing and computational text analysis.
- While the value of the cells is commonly the raw count of a given term, there are various schemes for weighting the raw counts such as, row normalizing (i.e. relative frequency/proportions) and tf-idf.
- Terms are commonly single words separated by whitespace or punctuation on either side (a.k.a. unigrams). In such a case, this is also referred to as "bag of words" representation because the counts of individual words is retained, but not the order of the words in the document.
- Using `sklearn.feature_extraction.text.CountVectorizer`
	```python
	from sklearn.feature_extraction.text import CountVectorizer

	# Ignore if frequency of the token is greater than `max_df` or lower than `min_df`.
	vect = CountVectorizer(min_df, max_df, max_features)
	```
	```python
	vect.fit(corpus)
	dtm = vect.transform(corpus).toarray()
	```
	```python
	dtm = vect.fit_transform(corpus).toarray()
	```
	```python
	token2id = vect.vocabulary_
	```
- Using `tf.keras.preprocessing.text.Tokenizer().texts_to_matrix(mode="count")`
	```python
	from tensorflow.keras.preprocessing.text import Tokenizer
	
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(corpus)
	
	# token2idx = tokenizer.word_index
	dtm = tokenizer.texts_to_matrix(corpus, mode="count").round(3)
	```
## TF-IDF(Term Frequency-Inverse Document Frequency)
- Reference: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- ***TF-IDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.***
### Term Frequency
- Suppose we have a set of English text documents and wish to rank them by which document is more relevant to the query, "the brown cow". A simple way to start out is by eliminating documents that do not contain all three words "the", "brown", and "cow", but this still leaves many documents. To further distinguish them, we might count the number of times each term occurs in each document; *the number of times a term occurs in a document is called its term frequency. However, in the case where the length of documents varies greatly, adjustments are often made (see definition below
### Inverse Document Frequency
- ***Because the term "the" is so common, term frequency will tend to incorrectly emphasize documents which happen to use the word "the" more frequently, without giving enough weight to the more meaningful terms "brown" and "cow". The term "the" is not a good keyword to distinguish relevant and non-relevant documents and terms, unlike the less-common words "brown" and "cow". Hence, an inverse document frequency factor is incorporated which diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely.***
- Implementation
	```python
	tf = doc.count(term)
	df = len([True for doc in corpus if term in doc])
	idf = log(len(corpus)/(1 + df))
	
	tfidf = tf*idf
	```
- Using `sklearn.feature_extraction.text.TfidfVectorizer()`
	```python
	from sklearn.feature_extraction.text import TfidfVectorizer
	```
- Using `tf.keras.preprocessing.text.Tokenizer().texts_to_matrix(mode="tfidf")`
	```python
	from tensorflow.keras.preprocessing.text import Tokenizer
	
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(corpus)
	
	# token2idx = tokenizer.word_index
	tfidf = tokenizer.texts_to_matrix(corpus, mode="tfidf").round(3)
	```
- Using `gensim.models.TfidfModel()`
	```python
	tfidf = gensim.models.TfidfModel(dtm)[dtm]
	```

# Word Embedding
- In natural language processing (NLP), word embedding is a term used for the representation of words for text analysis, ***typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning.*** Word embeddings can be obtained using a set of language modeling and feature learning techniques where words or phrases from the vocabulary are mapped to vectors of real numbers. ***Conceptually it involves the mathematical embedding from space with many dimensions per word to a continuous vector space with a much lower dimension.***
- Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing and sentiment analysis.
## Word2Vec
- The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, *such a model can detect synonymous words* or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. *The vectors are chosen carefully such that a simple mathematical function (the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors.*
- Word2vec is a group of related models that are used to produce word embeddings. *These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.*
- Word2vec can utilize either of two model architectures to produce a distributed representation of words: continuous bag-of-words (CBOW) or continuous skip-gram.
- Sub-sampling
	- High-frequency words often provide little information. Words with a frequency above a certain threshold may be subsampled to speed up training.
- Dimensionality
	- *Quality of word embedding increases with higher dimensionality. But after reaching some point, marginal gain diminishes. Typically, the dimensionality of the vectors is set to be between 100 and 1,000.*
- Context window
	- *The size of the context window determines how many words before and after a given word are included as context words of the given word. According to the authors' note, the recommended value is 10 for skip-gram and 5 for CBOW.*
## CBOW (Continuous Bag-Of-Words)
- In the continuous bag-of-words architecture, ***the model predicts the current word from a window of surrounding context words. The order of context words does not influence prediction (bag-of-words assumption).***
## (Continuous) Skip-Gram
- In the continuous skip-gram architecture, ***the model uses the current word to predict the surrounding window of context words. The skip-gram architecture weighs nearby context words more heavily than more distant context words. According to the authors' note, CBOW is faster while skip-gram does a better job for infrequent words.***
- Using `gensim.models.Word2Vec()`
	- Reference: https://radimrehurek.com/gensim/models/word2vec.html
	```python
	import os
	import gensim
	
	if os.path.exists(filename):
		model = gensim.models.KeyedVectors.load_word2vec_format(filename, [binary])
	else:
		# `vector_size`: Dimensionality of the word vectors.
		# `window`: Maximum distance between the current and predicted word within a sentence.
		# `min_count`: Ignores all words with total frequency lower than this.
		# `workers`: Use these many worker threads to train the model (= faster training with multicore machines).
		# `sg=0`: CBOW, `sg=1`: Skip-gram.
		model = gensim.models.Word2Vec(corpus, vector_size=100, window=5, min_count=5, workers=4, sg=0)
		model.wv.save_word2vec_format(filename)
		
	emb_mat = model.wv.vectors
	model.wv.most_similar(token)
	```
- Pre-trained Word Embedding (GoogleNews)
	```python
	file_name = "GoogleNews-vectors-negative300.bin.gz"
	if not os.path.exists(file_name):
		urllib.request.urlretrieve(f"https://s3.amazonaws.com/\
		dl4j-distribution/{file_name}")

	model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
	```
## FastText
- Using `gensim.models.FastText()`
	- Reference: https://radimrehurek.com/gensim/models/fasttext.html
	```python
	import os
	import gensim
	
	if os.path.exists(filename):
		pass
		model = gensim.models.FastText.load(filename)
	else:
		model = gensim.models.FastText(corpus, vector_size=100, window=5, min_count=5, workers=4, sg=1)
		model.save(filename)
		
	emb_mat = model.wv.vectors
	model.wv.most_similar(token)
	```
## GloVe (Global Vectors for Word Representation)
- Reference: https://wikidocs.net/22885
- Window based Co-occurrence Matrix
	- 단어의 동시 등장 행렬은 행과 열을 전체 단어 집합의 단어들로 구성하고, `i` 단어의 윈도우 크기(Window Size) 내에서 `k` 단어가 등장한 횟수를 `i`행 `k`열에 기재한 행렬을 말합니다.
	- 위 행렬은 행렬을 전치(Transpose)해도 동일한 행렬이 된다는 특징이 있습니다. 그 이유는 `i` 단어의 윈도우 크기 내에서 `k` 단어가 등장한 빈도는 반대로 `k` 단어의 윈도우 크기 내에서 `i` 단어가 등장한 빈도와 동일하기 때문입니다.
- Co-occurence Probability
	- 동시 등장 확률 P(k|i)는 동시 등장 행렬로부터 특정 단어 `i`의 전체 등장 횟수를 카운트하고, 특정 단어 `i`가 등장했을 때 어떤 단어 `k`가 등장한 횟수를 카운트하여 계산한 조건부 확률입니다.
	- `i`를 중심 단어(Center Word), `k`를 주변 단어(Context Word)라고 했을 때, 위에서 배운 동시 등장 행렬에서 중심 단어 `i`의 행의 모든 값을 더한 값을 분모로 하고 `i`행 `k`열의 값을 분자로 한 값이라고 볼 수 있겠습니다.
- GloVe는 Center word와 Context word의 dot product가 log of Co-occurence probability가 되도록 학습됩니다.
```python
from glove import Corpus, Glove
# pip install glove_python_binary

corp = Corpus()
# `corpus`로부터 Co-occurence matrix를 생성합니다.
corp.fit(corpus, window=5)

# `no_components`: Dimension of embedding vectors.
model = Glove(no_components=100, learning_rate=0.05)
model.fit(corp.matrix, epochs=20, no_threads=4, verbose=True)
model.add_dictionary(corp.dictionary)

# word2idx = corp.dictionary
```
- Pre-trained Word Embedding
	```python
	Reference: "http://nlp.stanford.edu/data/glove.6B.zip"
	file_name = "D:/glove.6B.zip"
	if not os.path.exists(file_name):
		urllib.request.urlretrieve(source, filename=file_name)
	file_name = "D:/glove.6B.100d.txt"
	if not os.path.exists(file_name):
		zipfile.ZipFile(file_name).extractall(path="D:/")

	token2vec = dict()
	with open(file_name, mode="r", encoding="utf-8") as f:
		for line in tqdm(f):
			line = line.split()
			word = line[0]
			emb_vec = np.array(line[1:], dtype="float32")
			token2vec[word] = emb_vec
		f.close()

	emb_dim = 100
	emb_mat = np.zeros(shape=(vocab_size + 2, emb_dim))
	for word, idx in word2idx.items():
		try:
			emb_mat[idx] = token2vec[word]
		except:
			continue
	```
## Skip-Gram with Negative Sampling (SGNS)
- Reference: https://wikidocs.net/69141
- Word2Vec의 출력층에서는 소프트맥스 함수를 지난 단어 집합 크기의 벡터와 실제값인 원-핫 벡터와의 오차를 구하고 이로부터 임베딩 테이블에 있는 모든 단어에 대한 임베딩 벡터 값을 업데이트합니다. 만약 단어 집합의 크기가 수만 이상에 달한다면 이 작업은 굉장히 무거운 작업이므로, Word2Vec은 꽤나 학습하기에 무거운 모델이 됩니다.
- Word2Vec은 역전파 과정에서 모든 단어의 임베딩 벡터값의 업데이트를 수행하지만, 만약 현재 집중하고 있는 중심 단어와 주변 단어가 '강아지'와 '고양이', '귀여운'과 같은 단어라면, 사실 이 단어들과 별 연관 관계가 없는 '돈가스'나 '컴퓨터'와 같은 수많은 단어의 임베딩 벡터값까지 업데이트하는 것은 비효율적입니다.
- 네거티브 샘플링은 Word2Vec이 학습 과정에서 전체 단어 집합이 아니라 일부 단어 집합에만 집중할 수 있도록 하는 방법입니다. 가령, 현재 집중하고 있는 주변 단어가 '고양이', '귀여운'이라고 해봅시다. 여기에 '돈가스', '컴퓨터', '회의실'과 같은 단어 집합에서 무작위로 선택된 주변 단어가 아닌 단어들을 일부 가져옵니다. 이렇게 하나의 중심 단어에 대해서 전체 단어 집합보다 훨씬 작은 단어 집합을 만들어놓고 마지막 단계를 이진 분류 문제로 변환합니다. 주변 단어들을 긍정(positive), 랜덤으로 샘플링 된 단어들을 부정(negative)으로 레이블링한다면 이진 분류 문제를 위한 데이터셋이 됩니다. 이는 기존의 단어 집합의 크기만큼의 선택지를 두고 다중 클래스 분류 문제를 풀던 Word2Vec보다 훨씬 연산량에서 효율적입니다.