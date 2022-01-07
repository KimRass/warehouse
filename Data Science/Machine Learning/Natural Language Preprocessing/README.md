# Text
- Source: https://en.wikipedia.org/wiki/Text_(literary_theory)
- In literary theory, *a text is any object that can be "read", whether this object is a work of literature, a street sign, an arrangement of buildings on a city block, or styles of clothing.*
## Corpus (plural Corpora)
- Source: https://21centurytext.wordpress.com/home-2/special-section-window-to-corpus/what-is-corpus/
- *A corpus is a collection of texts, written or spoken, usually stored in a computer database.* A corpus may be quite small, for example, containing only 50,000 words of text, or very large, containing many millions of words.
- *Written texts in corpora might be drawn from books, newspapers, or magazines that have been scanned or downloaded electronically. Other written corpora might contain works of literature, or all the writings of one author (e.g., William Shakespeare).* Such corpora help us to see how language is used in contemporary society, how our use of language has changed over time, and how language is used in different situations.
- People build corpora of different sizes for specific reasons. For example, a very large corpus would be required to help in the preparation of a dictionary. It might contain tens of millions of words – because it has to include many examples of all the words and expressions that are used in the language. A medium-sized corpus might contain transcripts of lectures and seminars and could be used to write books for learners who need academic language for their studies. Such corpora range in size from a million words to five or ten million words. Other corpora are more specialized and much smaller. These might contain the transcripts of business meetings, for instance, and could be used to help writers design materials for teaching business language.
	
# Language Model (LM)
## Statistical Language Model
- Source: https://en.wikipedia.org/wiki/Language_model
- ***A statistical language model is a probability distribution over sequences of words. Given such a sequence it assigns a probability to the whole sequence.***
- ***Data sparsity is a major problem in building language models. Most possible word sequences are not observed in training. One solution is to make the assumption that the probability of a word only depends on the previous n words. This is known as an n-gram model or unigram model when n equals to 1. The unigram model is also known as the bag of words model.***
## Bidirectional Language Model
- Bidirectional representations condition on both pre- and post- context (e.g., words) in all layers.

# Part-of-Speech
## Part-of-Speech Tagging
- Source: https://en.wikipedia.org/wiki/Text_corpus
- A corpus may contain texts in a single language (monolingual corpus) or text data in multiple languages (multilingual corpus).
- In order to make the corpora more useful for doing linguistic research, they are often subjected to a process known as annotation. *An example of annotating a corpus is part-of-speech tagging, or POS-tagging, in which information about each word's part of speech (verb, noun, adjective, etc.) is added to the corpus in the form of tags. Another example is indicating the lemma (base) form of each word. When the language of the corpus is not a working language of the researchers who use it, interlinear glossing is used to make the annotation bilingual.*

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
	```python
	import gensim
	
	if os.path.exists("ted_en_w2v"):.
		model = gensim.models.KeyedVectors.load_word2vec_format("ted_en_w2v")
	else:
		# `vector_size`: Dimensionality of the word vectors.
		# `min_count`: Ignores all words with total frequency lower than this.
		# `workers`: Use these many worker threads to train the model (=faster training with multicore machines).
		# `sg=0`: CBOW, `sg=1`: Skip-gram.
		model = gensim.models.Word2Vec(corpus, vector_size=100, window=5, min_count=5, workers=4, sg=0)
		model.wv.save_word2vec_format("ted_en_w2v")
		
	sim_words = model.wv.most_similar("man")
	```
## FastText
## GloVe
## SGNS (Skip-Gram with Negative Sampling)

# Syntactic & Semantic Analysis
- Source: https://builtin.com/data-science/introduction-nlp
- Syntactic analysis (syntax) and semantic analysis (semantic) are the two primary techniques that lead to the understanding of natural language. Language is a set of valid sentences, but what makes a sentence valid? Syntax and semantics.
- ***Syntax is the grammatical structure of the text, whereas semantics is the meaning being conveyed. A sentence that is syntactically correct, however, is not always semantically correct. For example, “cows flow supremely” is grammatically valid (subject — verb — adverb) but it doesn't make any sense.***

# Preprocessing
## Tokenization
- Source: https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/
- Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual words or terms. Each of these smaller units are called tokens.
- The tokens could be words, numbers or punctuation marks. In tokenization, smaller units are created by locating word boundaries. Wait – what are word boundaries?
- These are the ending point of a word and the beginning of the next word. These tokens are considered as a first step for stemming and lemmatization.
### Word Tokenization
```python
text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
```
- Using `nltk.tokenize.word_tokenize()`
	```python
	from nltk.tokenize import word_tokenize
	
	word_tokens = word_tokenize(text)
	```
	- `Don't` -> `Do` + `n't`, `Jone's` -> `Jone` + `'s`
- Using `nltk.tokenize.WordPunctTokenizer().tokenize()`
	```python
	from nltk.tokenize import WordPunctTokenizer
	
	text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
	
	word_tokens = WordPunctTokenizer().tokenize(text)
	```
	- `Don't` -> `Don` + `'` + `t`, `Jone's` -> `Jone` + `'` + `s`
- Using `nltk.tokenize.TreebankWordTokenizer().tokenize()`
	```python
	from nltk.tokenize import TreebankWordTokenizer
	
	text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
	
	word_tokens = TreebankWordTokenizer().tokenize(text)
	```
- Using `konlpy` (for Korean language)
	```python
	from konlpy.tag import *

	okt = Okt()
	# kkm = Kkma()
	# kmr = Komoran()
	# hnn = Hannanum()
	
	nouns = okt.nouns() # kkm.nouns(), kmr.nouns(), hnn.nouns()
	morphs = okt.morphs() # kkm.morphs(), kmr.morphs(), hnn.morphs()
	pos = okt.pos() # kkm.pos(), kmr.pos(), hnn.pos()
	```
### Sentence Tokenization
- Using `nltk.tokenize.sent_tokenize()`
	```python
	from nltk.tokenize import sent_tokenize

	text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
	
	sent_tokens = sent_tokenize(text)
	```
- Using `kss.split_sentences()` (for Korean language)
	```python
	import kss
	
	text = "딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?"
	
	sent_tokens = kss.split_sentences(text)
	```
## Word Divider
```python
from pykospacing import spacing

text = "오지호는극중두얼굴의사나이성준역을맡았다.성준은국내유일의태백권전승자를가리는결전의날을앞두고20년간동고동락한사형인진수(정의욱분)를찾으러속세로내려온인물이다."

sent_div = spacing(text)
```
## Stemming & Lemmatization
### Stemming
- Source: https://builtin.com/data-science/introduction-nlp
- Basically, stemming is the process of reducing words to their word stem. A "stem" is the part of a word that remains after the removal of all affixes. For example, the stem for the word "touched" is "touch." "Touch" is also the stem of "touching," and so on.
- You may be asking yourself, why do we even need the stem? Well, *the stem is needed because we're going to encounter different variations of words that actually have the same stem and the same meaning.*Now, imagine all the English words in the vocabulary with all their different fixations at the end of them. To store them all would require a huge database containing many words that actually have the same meaning. This is solved by focusing only on a word’s stem. Popular algorithms for stemming include the Porter stemming algorithm from 1979, which still works well.
- Using `nltk.stem.PorterStemmer().stem()`
	```python
	from nltk.stem import PorterStemmer
	
	ps = PorterStemmer()

	words = ["formalize", "allowance", "electricical"]	
	
	stems = [ps.stem(w) for w in words]
	```
### Lemmatization
- Source: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
-  Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.
- Using `nltk.stem.WordNetLemmatizer().lemmatize()`
	```python
	import nltk
	from nltk.stem import WordNetLemmatizer
	
	nltk.download("wordnet")
	wnl = WordNetLemmatizer()
	
	words = ["policy", "doing", "organization", "have", "going", "love", "lives", "fly", "dies", "watched", "has", "starting"]
	
	lemmas = [wnl.lemmatize(w) for w in words])
	```
## Check Spelling
- Using `hanspell.spell_checker.check()`
	```python
	from hanspell import spell_checker
	
	sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
	
	sent_checked = spell_checker.check(sent)
	```
## Stopwords
- Using `nltk.corpus.stopwords.words()`
	```python
	import nltk
	from nltk.corpus import stopwords
	
	nltk.download("stopwords")
	
	sw = stopwords.words("english")
	```
	
# Tasks
## NER (Named Entity Recognition)
- Source: https://builtin.com/data-science/introduction-nlp
- *Named entity recognition (NER) concentrates on determining which items in a text (i.e. the "named entities") can be located and classified into pre-defined categories. These categories can range from the names of persons, organizations and locations to monetary values and percentages.*
## Sentiment Analysis
- *With sentiment analysis we want to determine the attitude (i.e. the sentiment) of a speaker or writer with respect to a document, interaction or event. Therefore it is a natural language processing problem where text needs to be understood in order to predict the underlying intent. The sentiment is mostly categorized into positive, negative and neutral categories.*

# 너무 짧은 문장 자르기
```python
lens = sorted([len(doc) for doc in train_X])
ratio = 0.99
max_len = int(np.quantile(lens, ratio))
print(f"가장 긴 문장의 길이는 {np.max(lens)}입니다.")
print(f"길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```

# NLU
# NLG

# Bag-of-Words Model
- Source: https://en.wikipedia.org/wiki/Bag-of-words_model
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

	vect = CountVectorizer()
	```
	```python
	vect.fit(corpus)
	dtm = vect.transform(corpus).toarray()
	```
	```python
	dtm = vect.fit_transform(corpus).toarray())
	```
	```python
	token2id = vect.vocabulary_
	```
- Using `tf.keras.preprocessing.text.Tokenizer().texts_to_matrix(mode="count")`
	```python
	import tensorflow as tf
	
	tokenizer = tf.keras.preprocessing.text.Tokenizer()
	tokenizer.fit_on_texts(corpus)
	
	# token2idx = tokenizer.word_index
	dtm = tokenizer.texts_to_matrix(corpus, mode="count").round(3)
	```
## TF-IDF(Term Frequency-Inverse Document Frequency)
- Source: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
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
	import tensorflow as tf
	
	tokenizer = tf.keras.preprocessing.text.Tokenizer()
	tokenizer.fit_on_texts(corpus)
	
	# token2idx = tokenizer.word_index
	tfidf = tokenizer.texts_to_matrix(corpus, mode="tfidf").round(3)
	```
- Using `gensim.models.TfidfModel()`
	```python
	tfidf = gensim.models.TfidfModel(dtm)[dtm]
	```
# Latent Dirichlet Allocation
## `gensim.models.ldamodel.Ldamodel()`
```python
model = gensim.models.ldamodel.LdaModel(dtm, num_topics=n_topics, id2word=id2word, alpha="auto", eta="auto")
```
## `pyLDAvis`
```python
import pyLDAvis
```
## `pyLDAvis.enable_notebook()`
- `pyLDAvis`를 Jupyter Notebook에서 실행할 수 있게 활성화합니다.
## `pyLDAvis.gensim`
```python
import pyLDAvis.gensim
```
### `pyLDAvis.gensim.prepare()`
```python
pyldavis = pyLDAvis.gensim.prepare(model, dtm, id2word)
```
- Source: https://wikidocs.net/30708
- 도수 기반의 표현 방법인 BoW의 행렬 DTM 또는 TF-IDF 행렬을 입력으로 함.
## LDA의 가정
- 각각의 문서는 다음과 같은 과정을 거쳐서 작성되었다고 가정.
1. 문서에 사용할 단어의 개수 N을 정합니다.
    - Ex) 5개의 단어를 정하였습니다.
2. 문서에 사용할 토픽의 혼합을 확률 분포에 기반하여 결정합니다.
    - Ex) 위 예제와 같이 토픽이 2개라고 하였을 때 강아지 토픽을 60%, 과일 토픽을 40%와 같이 선택할 수 있습니다.
3. 문서에 사용할 각 단어를 (아래와 같이) 정합니다.
	1. 토픽 분포에서 토픽 T를 확률적으로 고릅니다.
		- Ex) 60% 확률로 강아지 토픽을 선택하고, 40% 확률로 과일 토픽을 선택할 수 있습니다.
    2. 선택한 토픽 T에서 단어의 출현 확률 분포에 기반해 문서에 사용할 단어를 고릅니다.
		- Ex) 강아지 토픽을 선택하였다면, 33% 확률로 강아지란 단어를 선택할 수 있습니다. 이제 3)을 반복하면서 문서를 완성합니다.
## LDA의 수행
1. 사용자는 알고리즘에게 토픽의 개수 k를 알려줍니다.
2. 모든 문서의 모든 단어에 대해서 k개 중 하나의 토픽을 랜덤으로 할당.
랜덤으로 할당하였기 때문에 사실 이 결과는 전부 틀린 상태입니다. 만약 한 단어가 한 문서에서 2회 이상 등장하였다면, 각 단어는 서로 다른 토픽에 할당되었을 수도 있습니다.
3. 모든 단어에 대해 다음을 반복.
어떤 문서의 각 단어 w는 자신은 잘못된 토픽에 할당되어져 있지만, 다른 단어들은 전부 올바른 토픽에 할당되어져 있는 상태라고 가정합니다. 이에 따라 단어 w는 아래의 두 가지 기준에 따라서 토픽이 재할당됩니다.
    - 단어 w가 속한 문서 d에서 어떤 토픽이 가장 큰 비중을 차지하는지.
    - 전체 문서들 속에서 단어 w가 어떤 토픽에 주로 속하는지.
- 즉, 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률을 결합확률로 추정하여 토픽을 추출.
