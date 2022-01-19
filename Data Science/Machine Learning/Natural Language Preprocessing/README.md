# Text
- Source: https://en.wikipedia.org/wiki/Text_(literary_theory)
- In literary theory, *a text is any object that can be "read", whether this object is a work of literature, a street sign, an arrangement of buildings on a city block, or styles of clothing.*
## Corpus (plural Corpora)
- Source: https://21centurytext.wordpress.com/home-2/special-section-window-to-corpus/what-is-corpus/
- *A corpus is a collection of texts, written or spoken, usually stored in a computer database.* A corpus may be quite small, for example, containing only 50,000 words of text, or very large, containing many millions of words.
- *Written texts in corpora might be drawn from books, newspapers, or magazines that have been scanned or downloaded electronically. Other written corpora might contain works of literature, or all the writings of one author (e.g., William Shakespeare).* Such corpora help us to see how language is used in contemporary society, how our use of language has changed over time, and how language is used in different situations.
- People build corpora of different sizes for specific reasons. For example, a very large corpus would be required to help in the preparation of a dictionary. It might contain tens of millions of words – because it has to include many examples of all the words and expressions that are used in the language. A medium-sized corpus might contain transcripts of lectures and seminars and could be used to write books for learners who need academic language for their studies. Such corpora range in size from a million words to five or ten million words. Other corpora are more specialized and much smaller. These might contain the transcripts of business meetings, for instance, and could be used to help writers design materials for teaching business language.

# Puctuation
```python
import string

sw = {i for i in string.punctuation}
```
- Output: `"!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"`
	
# Part-of-Speech
## Part-of-Speech Tagging
- Source: https://en.wikipedia.org/wiki/Text_corpus
- A corpus may contain texts in a single language (monolingual corpus) or text data in multiple languages (multilingual corpus).
- In order to make the corpora more useful for doing linguistic research, they are often subjected to a process known as annotation. *An example of annotating a corpus is part-of-speech tagging, or POS-tagging, in which information about each word's part of speech (verb, noun, adjective, etc.) is added to the corpus in the form of tags. Another example is indicating the lemma (base) form of each word. When the language of the corpus is not a working language of the researchers who use it, interlinear glossing is used to make the annotation bilingual.*

# Datasets
## `sklearn.datasets.fetch_20newsgroups()`
## Steam Reviews
- Source: https://github.com/bab2min/corpus/tree/master/sentiment
## Naver Shopping
- Source: https://github.com/bab2min/corpus/tree/master/sentiment
## NLP Challenge
## fra-eng
- Source: https://www.kaggle.com/myksust/fra-eng/activity
## IMDb
## Annotated Corpus for NER
## Chatbot Data for Korean
- Source: https://github.com/songys/Chatbot_data
## Natural Language Understanding benchmark
## Naver Sentiment Movie Corpus
- Source: https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt
## TED

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
	
# BLEU Score
- BLEU는 기계 번역 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하여 번역에 대한 성능을 측정하는 방법입니다. 측정 기준은 n-gram에 기반합니다.
- 번역된 문장을 `cand`(candidate), 완벽한 번역 문장을 `ref`(Reference)라고 하겠습니다.
- Using `nltk.translate.bleu_score.sentence_bleu()`
- Brevity Penalty

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
- Using `tensorflow.keras.preprocessing.text.Tokenizer()`
	```python
	tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="UNK")
	tokenizer.fit_on_texts(corpus)

	token2idx = tokenizer.word_index
	idx2token = tokenizer.index_word
	tokenizer.word_counts
	tokenizer.texts_to_sequences()
	tokenizer.sequences_to_texts()
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
- Using `pykospacing.spacing()`
	```python
	!pip install git+https://github.com/haven-jeon/PyKoSpacing.git --user
	```
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
	!pip install git+https://github.com/ssut/py-hanspell.git
	```
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
## Padding
```python
tr_X = tf.keras(preprocessing.sequence.pad_sequences(tr_X, padding="post", maxlen_max_len)
tr_y = tf.keras(preprocessing.sequence.pad_sequences(tr_y, padding="post", maxlen_max_len)

# tr_X = tf.keras.utils.to_categorical(tr_X)
# tr_y = tf.keras.utils.to_categorical(tr_y)
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
print(f"길이가 가장 긴 문장의 길이는 {np.max(lens)}이고 길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```

# Subword Tokenizer
- Using `sentencepiece`
	- Reference: https://pypi.org/project/sentencepiece/, https://github.com/google/sentencepiece
	- 사전 토큰화 작업(pretokenization)없이 전처리를 하지 않은 데이터(raw data)에 바로 단어 분리 토크나이저를 사용할 수 있다면, 이 토크나이저는 그 어떤 언어에도 적용할 수 있는 토크나이저가 될 것입니다. 센텐스피스는 이 이점을 살려서 구현되었습니다. 센텐스피스는 사전 토큰화 작업없이 단어 분리 토큰화를 수행하므로 언어에 종속되지 않습니다.
	- Sentencepiece의 학습 데이터로는 빈 칸이 포함되지 않은 문서 집합이어야 합니다.
	- Training
		```python
		# `--input` : 학습시킬 파일
		# `--model_type` : (`"unigram"`, `"bpe"`, `"char"`, `"word"`, default `"unigram"`). The input sentence must be pretokenized when using `word`.
		# `--character_coverage`: Amount of characters covered by the model, good defaults are: `0.9995` for languages with rich character set like Japanese or Chinese and `1.0` for other languages with small character set.
		# `--model_prefix.model` and `--model_prefix.vocab` are generated.
		input_ = "./NSMC_document.txt"
		model_prefix = "NSMC"
		vocab_size = 5000
		model_type = "bpe"
		spm.SentencePieceTrainer.train(f"--input={input_} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type}")
		
		# By default, `sentencepiece` uses Unknown (`"<unk>"`), BOS (`"<s>"`) and EOS (`"</s>"`) tokens which have the ids of `0`, `1`, and `2` respectively.
		subwords = pd.read_csv("--model_prefix.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
		```
	- Segmentation
		```python
		import sentencepiece as sp
		
		sp = spm.SentencePieceProcessor(model_file="--model_prefix.model")
		
		piece_size = sp.get_piece_size()
		
		# `enable_sampling=True`: Applys Drop-out. `sents` are segmented differently on each `encode()` calls.
		# `alpha`: Drop-out rate
		ids = sp.encode(sents, out_type=int, enable_sampling=True, alpha=0.1, nbest_size=-1)
		# ids = spp.encode_as_ids(sents)
		pieces = sp.encode(sents, out_type=str)
		# pieces = spp.encode_as_pieces(sents)
		
		sents = sp.decode(ids)
		sents = sp.decode(pieces)
		
		pieces = sp.id_to_piece(ids)
		ids = sp.piece_to_id(pieces)
		# ids = sp[pieces]
		```
- Using `tfds.deprecated.text.SubwordTextEncoder`
	- Reference: https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder
	```python
	import tensorflow_datasets as tfds

	# Build
	enc = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus_generator, target_vocab_size=2**15)
	# Load
	enc = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_fname)

	subwords = enc.subwords
	# Encodes text into a list of integers.
	encoded = enc.encode(sents)
	# Decodes a list of integers into text.
	sents = enc.decode(decoded)

	# Save the vocabulary to a file.
	enc.save_to_file(filename_prefix)

	# Extracts list of subwords from file.
	enc.load_from_file(filename_prefix)
	```

# NLU
# NLG

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

# Language Model (LM)
## Statistical Language Model
- Source: https://en.wikipedia.org/wiki/Language_model
- ***A statistical language model is a probability distribution over sequences of words. Given such a sequence it assigns a probability to the whole sequence.***
- ***Data sparsity is a major problem in building language models. Most possible word sequences are not observed in training. One solution is to make the assumption that the probability of a word only depends on the previous n words. This is known as an n-gram model or unigram model when n equals to 1. The unigram model is also known as the bag of words model.***
## Bidirectional Language Model
- Bidirectional representations condition on both pre- and post- context (e.g., words) in all layers.

# seq2seq
- seq2seq는 크게 두 개로 구성된 아키텍처로 구성되는데, 바로 인코더와 디코더입니다. 인코더는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보들을 압축해서 하나의 벡터로 만드는데, 이를 컨텍스트 벡터(context vector)라고 합니다. 입력 문장의 정보가 하나의 컨텍스트 벡터로 모두 압축되면 인코더는 컨텍스트 벡터를 디코더로 전송합니다. 디코더는 컨텍스트 벡터를 받아서 번역된 단어를 한 개씩 순차적으로 출력합니다.
- 디코더는 초기 입력으로 문장의 시작을 의미하는 심볼 `<sos>`가 들어갑니다. 디코더는 `<sos>`가 입력되면, 다음에 등장할 확률이 높은 단어를 예측합니다. 첫번째 시점(time step)의 디코더 RNN 셀은 다음에 등장할 단어로 je를 예측하였습니다. 첫번째 시점의 디코더 RNN 셀은 예측된 단어 je를 다음 시점의 RNN 셀의 입력으로 입력합니다. 그리고 두번째 시점의 디코더 RNN 셀은 입력된 단어 je로부터 다시 다음에 올 단어인 suis를 예측하고, 또 다시 이것을 다음 시점의 RNN 셀의 입력으로 보냅니다. 디코더는 이런 식으로 기본적으로 다음에 올 단어를 예측하고, 그 예측한 단어를 다음 시점의 RNN 셀의 입력으로 넣는 행위를 반복합니다. 이 행위는 문장의 끝을 의미하는 심볼인 `<eos>`가 다음 단어로 예측될 때까지 반복됩니다. 지금 설명하는 것은 테스트 과정 동안의 이야기입니다.
## Character-Level seq2seq

# Bidirectional LSTM Sentiment Analysis
```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size+2, output_dim=64))
hidden_size = 128
model.add(Bidirectional(LSTM(units=hidden_size)))
model.add(Dense(units=1, activation="sigmoid"))

es = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=2)
model_path = "steam_reviews_bilstm.h5"
mc = ModelCheckpoint(filepath=model_path, monitor="val_binary_accuracy", mode="auto", verbose=1, save_best_only=True)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["binary_accuracy"])

batch_size = 256
hist = model.fit(x=tr_X, y=tr_y, validation_split=0.2, batch_size=batch_size, epochs=10, verbose=1, callbacks=[es, mc])
```

# `tokenization_kobert`
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/monologg/KoBERT-NER/master/tokenization_kobert.py", filename="tokenization_kobert.py")
```
## `KoBertTokenizer`
```python
from tokenization_kobert import KoBertTokenizer
```
- `KoBertTokenizer` 파일 안에 `from transformers import PreTrainedTokenizer`가 이미 되어있습니다.
### `KoBertTokenizer.from_pretrained()`
```python
tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
```
#### `tokenier.tokenize()`
```python
tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역")
```
#### `tokenizer.encode()`
```python
tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역")
```
- `max_length`
- `padding="max_length"`
#### `tokenizer.convert_tokens_to_ids()`
```python
tokenizer.convert_tokens_to_ids("[CLS]")
```
- Unknown Token: `0`, `"[PAD]"`: `1`, `"[CLS]"`: `2`, `"[SEP]"`: `3`

# `transformers`
```python
!pip install --target=$my_path transformers==3.5.0
```
## `BertModel`
```python
from transformers import BertModel
```
```python
model = BertModel.from_pretrained("monologg/kobert")
```
## `TFBertModel`
```python
from transformers import TFBertModel
```
### `TFBertModel.from_pretrained()`
```python
model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True,
                                    num_labels=len(tag2idx), output_attentions=False,
                                    output_hidden_states = False)
```
```python
bert_outputs = model([token_inputs, mask_inputs])
```
#### `model.save()`
```python
model.save("kobert_navermoviereview.h5", save_format="tf")
```
### `BertModel.from_pretrained()`
```python
model = BertModel.from_pretrained("monologg/kobert")
```

# `pyLDAvis`
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

# `soynlp`
## `soynlp.normalizer`
```python
from soynlp.normalizer import *
```
### `emoticon_normalize()`
```python
emoticon_normalize("앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ", num_repeats=2)
```
### `repeat_normalize()`
```python
repeat_normalize("와하하하하하하하하하핫", num_repeats=2)
```

# `khaiii`
## `KhaiiiApi`
```python
from khaiii import KhaiiiApi
```
```python
api = KhaiiiApi()
```
### `api.analyze()`
#### `word.morphs`
##### `morph.lex`
##### `morph.tag`
```python
morphs = []
sentence = "하스스톤 전장이 새로 나왔는데 재밌어요!"
for word in api.analyze(sentence):
    for morph in word.morphs:
        morphs.append((morph.lex, morph.tag))
```

# `nltk`
```python
import nltk
```
## `nltk.tokenize`
### `nltk.tokenize.word_tokenize()`
```python
nltk.tokenize.word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.")
```
### `nltk.tokenize.sent_tokenize()`
```python
nltk.tokenize.sent_tokenize("I am actively looking for Ph.D. students and you are a Ph.D student.")
```
### `WordPunctTokenizer()`
```python
from nltk.tokenize import WordPunctTokenizer
```
#### `WordPunctTokenizer().tokenize()`
```python
WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.")
```
### `TreebankWordTokenizer()`
```python
from nltk.tokenize import TreebankWordTokenizer
```
#### `TreebankWordTokenizer().tokenize()`
```python
TreebankWordTokenizer().tokenize("Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own.")
```
- Penn Treebank Tokenization.
## `nltk.stem`
### `PorterStemmer()`
```python
from nltk.stem import PorterStemmer
```
```python
ps = PorterStemmer()
```
#### `ps.stem()`
```python
[ps.stem(word) for word in ["formalize", "allowance", "electricical"]]
```
### `WordNetLemmatizer()`
```python
from nltk.stem import WordNetLemmatizer
```
```python
wnl = WordNetLemmatizer()
```
#### `wnl.lemmatize()`
```python
wnl.lemmatize("watched", "v")
```
## `nltk.Text()`
```python
text = nltk.Text(total_tokens, name="NMSC")
```
### `text.tokens`
### `text.vocab()`
- Returns frequency distribution
#### `text.vocab().most_common()`
```python
text.vocab().most_common(10)
```
### `text.plot()`
## `nltk.download()`
- (`"punkt"`, `"wordnet"`, `"stopwords"`, `"movie_reviews"`)
## `nltk.corpus`
### `stopwords`
```python
from nltk.corpus import stopwords
```
#### `stopwords.words()`
```python
stopwords.words("english")
```
### `movie_reviews`
```python
from nltk.corpus import movie_reviews
```
#### `movie_reviews.sents()`
```python
sentences = [sent for sent in movie_reviews.sents()]
```
### `nltk.corpus.treebank`
#### `nltk.corpus.treebank.tagged_sents()`
```python
tagged_sents = nltk.corpus.treebank.tagged_sents()
```
## `nltk.translate`
### `nltk.translate.bleu_score`
```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
```
#### `sentence_bleu()`
```python
ref = [["this", "is", "a", "test"], ["this", "is" "test"]]
cand = ["this", "is", "a", "test"]
score = nltk.translate.bleu_score.sentence_bleu(ref, cand)
```
- `weights`: e.g., `(1/2, 1/2, 0, 0)`
#### `corpus_bleu()`
```python
refs = [[["this", "is", "a", "test"], ["this", "is" "test"]]]
cands = [["this", "is", "a", "test"]]
score = nltk.translate.bleu_score.corpus_bleu(refs, cands)
```
- `weights`=(1/2, 1/2, 0, 0)
#### `SmoothingFunction()`
## `nltk.ngrams()`
```python
nltk.ngrams("I am a boy", 3)
```

# `MeCab`
- Install on Google Colab
	```python
	!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
	%cd Mecab-ko-for-Google-Colab
	!bash install_mecab-ko_on_colab190912.sh
	```
- Install on Microsoft Windows
	- Source: https://cleancode-ws.tistory.com/97, https://github.com/Pusnow/mecab-python-msvc/releases/tag/mecab_python-0.996_ko_0.9.2_msvc-2
	```python
	!pip install mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl
	```
```python
import MeCab
```
```python
class Mecab:
    def pos(self, text):
        p = re.compile(".+\t[A-Z]+")
        return [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
    
    def morphs(self, text):
        p = re.compile(".+\t[A-Z]+")
        return [p.match(line).group().split("\t")[0] for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
    
    def nouns(self, text):
        p = re.compile(".+\t[A-Z]+")
        temp = [tuple(p.match(line).group().split("\t")) for line in MeCab.Tagger().parse(text).splitlines()[:-1]]
        nouns=[]
        for word in temp:
            if word[1] in ["NNG", "NNP", "NNB", "NNBC", "NP", "NR"]:
                nouns.append(word[0])
        return nouns
    
mcb = Mecab()
```

# `konlpy`
## `konlpy.tag`
```python
from konlpy.tag import *

okt = Okt()
kkm = Kkma()
kmr = Komoran()
hnn = Hannanum()
```
#### `okt.nouns()`, `kkm.nouns()`, `kmr.nouns()`, `hnn.nouns()`
#### `okt.morphs()`, `kkm.morphs()`, `kmr.morphs()`, `hnn.morphs()`
- `stem`: (bool)
- `norm`: (bool)
#### `okt.pos()`, `kkm.pos()`, `kmr.pos()`, `hnn.pos()`
- `stem`: (bool)
- `norm`: (bool)

# `ckonlpy`
```python
!pip install customized_konlpy
```
```python
from ckonlpy.tag import Twitter

twt = Twitter()
```
## `twt.add_dictionary()`
```python
twt.add_dictionary("은경이", "Noun")
```

# `glove`
```python
!pip install glove_python
```
### `glove.Corpus`
### `glove.Glove`

# `gensim`
```python
import gensim
```
## `gensim.corpora`
### `gensim.corpora.Dictionary()`
```python
id2word = gensim.corpora.Dictionary(docs_tkn)
```
#### `id2word.id2token`
- `dict(id2word)` is same as `dict(id2word.id2token)`
#### `id2word.token2id`
#### `id2word.doc2bow()`
```python
dtm = [id2word.doc2bow(doc) for doc in docs_tkn]
```
#### `gensim.corpora.Dictionary.load()`
```python
id2word = gensim.corpora.Dictionary.load("kakaotalk id2word")
```
### `gensim.corpora.BleiCorpus`
#### `gensim.corpora.BleiCorpus.serizalize()`
```python
gensim.corpora.BleiCorpus.serialize("kakotalk dtm", dtm)
```
### `gensim.corpora.bleicorpus`
#### `gensim.corpora.bleicorpus.BleiCorpus()`
```python
dtm = gensim.corpora.bleicorpus.BleiCorpus("kakaotalk dtm")
```
## `gensim.models`
### `gensim.models.TfidfModel()`
```python
tfidf = gensim.models.TfidfModel(dtm)[dtm]
```
### `gensim.models.AuthorTopicModel()`
```python
model = gensim.models.AuthorTopicModel(corpus=dtm, id2word=id2word, num_topics=n_topics, author2doc=aut2doc, passes=1000)
```
#### `gensim.models.AuthorTopicModel.load()`
```python
model = gensim.models.AuthorTopicModel.load("kakaotalk model")
```
## `gensim.models.ldamodel.Ldamodel()`
```python
model = gensim.models.ldamodel.LdaModel(dtm, num_topics=n_topics, id2word=id2word, alpha="auto", eta="auto")
```
## `gensim.models.Word2Vec()`
```python
model = gensim.models.Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)
```
- `size` : 임베딩 벡터의 차원.
- `min_count` : 단어 최소 빈도 수(빈도가 적은 단어들은 학습하지 않는다)
- `workers` : 학습을 위한 프로세스 수  
- `sg=0` :cBoW
- `sg=1` : Skip-gram.  
### `gensim.models.FastText()`
```python
model = gensim.models.FastText(sentences, min_count=5, sg=1, size=300, workers=4, min_n=2, max_n=7, alpha=0.05, iter=10, window=7)
```
### `gensim.models.KeyedVectors`
### `gensim.models.KeyedVectors.load_word2vect_format()`
```python
model = gensim.models.KeyedVectors.load_word2vec_format("eng_w2v")
```
```python
model_google = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)  
```
- Loads a model.
#### `model.vectors`
#### `model.save()`
```python
model.save("kakaotalk model")
```
#### `model.show_topic()`
```python
model.show_topic(1, topn=20)
```
- Arguments : (the index of the topic, number of words to print)
#### `model.wv`
##### `model.wv.vecotrs`
##### `model.wv.most_similar()`
```python
model.wv.most_similar("안성기")
```
##### `model.wv.Save_word2vec_format()`
```python
model.wv.save_word2vec_format("eng_w2v")
```

##### `pad_sequences()`
```python
train_X = pad_sequences(train_X, maxlen=max_len)
```
```python
X_char = [pad_sequences([[char2idx[char] if char in chars else 1 for char in word] for word in sent]) for sent in corpus]
```
```python
train_X = pad_sequences([tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_lists], 
                        maxlen=max_len, value=tokenizer.convert_tokens_to_ids("[PAD]"),
                        truncating="post", padding="post")
```
- `padding`: (`"pre"`, `"post"`)
- `truncating`: (`"pre"`, `"post"`)
- `value=` : padding에 사용할 value를 지정합니다.
#### `tf.keras.preprocessing.text`
##### `tf.keras.preprocessing.text.Tokenizer()`
```python
tkn = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size+2, oov_token="UNK", lower=True)
```
- `lower=False`: 대문자를 유지합니다.
##### `tkn.fit_on_texts()`
```python
tkn.fit_on_texts(["나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"])
```
##### `tkn.word_index`
```python
word2idx = tkn.word_index
```
##### `tkn.index_word`
##### `tkn.word_counts`
```python
word2cnt = dict(sorted(tkn.word_counts.items(), key=lambda x:x[1], reverse=True))

cnts = list(word2cnt.values())

for vocab_size, value in enumerate(np.cumsum(cnts)/np.sum(cnts)):
    if value >= ratio:
        break

print(f"{vocab_size:,}개의 단어로 전체 data의 {ratio:.0%}를 표현할 수 있습니다.")
print(f"{len(word2idx):,}개의 단어 중 {vocab_size/len(word2idx):.1%}에 해당합니다.")
```
##### `tkn.texts_to_sequences()`
```python
train_X = tkn.texts_to_sequences(train_X)
```
- `num_words`가 적용됩니다.
##### `tkn.sequences_to_texts()`
##### `tkn.texts_to_matrix()`
```python
tkn.texts_to_matrix(["먹고 싶은 사과", "먹고 싶은 바나나", "길고 노란 바나나 바나나", "저는 과일이 좋아요"], mode="count"))
```
- `mode`: (`"count"`, `"binary"`, `"tfidf"`, `"freq"`)
- `num_words`가 적용됩니다.
# `tensorflow_hub`
```python
import tensorflow_hub as hub
```
## `hub.Module()`
```python
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
```
### `elmo()`
```python
embeddings = elmo(["the cat is on the mat", "dogs are in the fog"], signature="default", as_dict=True)["elmo"]
```