# Datasets
## Reuters Newswire Classification Dataset
```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=None, test_split=0.2)
```
## 20 Newsgroups Classification Dataset
```python
data = sklearn.datas.fetch_20newsgroups(shuffle=True, [random_state], remove=("headers", "footers", "quotes"), [subset])
corpus = data["data"]
```
- `subset`: (`"all"`, `"train"`, `"test"`)
## Steam Reviews
- Source: https://github.com/bab2min/corpus/tree/master/sentiment
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename="./Datasets/Steam Reviews/steam.txt")
data = pd.read_table("./Datasets/Steam Reviews/steam.txt", names=["label", "review"])

data["review"] = data["review"].str.replace(f"[{string.punctuation}]", " ")
data["review"] = data["review"].str.replace(r" +", " ")
data = data[data["review"]!=" "]
data = data.dropna(axis=0)
data = data.drop_duplicates(["review"], keep="first")
```
## `tensorflow.keras.datasets.imdb`
```python
from tensorflow.keras.datasets import imdb

vocab_size = 10000
(tr_X, tr_y), (te_X, te_y) = imdb.load_data(num_words=vocab_size)
```
## Naver Shopping
- Source: https://github.com/bab2min/corpus/tree/master/sentiment
## NLP Challenge
## fra-eng
- Source: https://www.kaggle.com/myksust/fra-eng/activity
```python
data = pd.read_table("./Datasets/fra-eng/fra.txt", usecols=[0, 1], names=["src", "tar"])
# data = pd.read_csv("./Datasets/fra-eng/fra.txt", usecols=[0, 1], names=["src", "tar"], sep="\t")
```
## IMDb
## Annotated Corpus for NER
## Chatbot Data for Korean
- Source: https://github.com/songys/Chatbot_data
## Natural Language Understanding Benchmark
- Source: https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines, https://github.com/ajinkyaT/CNN_Intent_Classification
- Labels: (`"AddToPlaylist"`, `"BookRestaurant"`, `"GetWeather"`, `"RateBook"`, `"SearchCreativeWork"`, `"SearchScreeningEvent"`)
 'BookRestaurant',
 'GetWeather',
 'RateBook',
 'SearchCreativeWork',
 'SearchScreeningEvent'}
```python
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/train_text.npy", filename="./Datasets/NLU Benchmark/train_text.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/train_label.npy", filename="./Datasets/NLU Benchmark/train_label.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/test_text.npy", filename="./Datasets/NLU Benchmark/test_text.npy")
urllib.request.urlretrieve("https://github.com/ajinkyaT/CNN_Intent_Classification/raw/master/data/test_label.npy", filename="./Datasets/NLU Benchmark/test_label.npy")

train_text = np.load("./Datasets/NLU Benchmark/train_text.npy", allow_pickle=True).tolist()
train_label = np.load("./Datasets/NLU Benchmark/train_label.npy", allow_pickle=True).tolist()
test_text = np.load("./Datasets/NLU Benchmark/test_text.npy", allow_pickle=True).tolist()
label_test = np.load("./Datasets/NLU Benchmark/test_label.npy", allow_pickle=True).tolist()
```
## Naver Sentiment Movie Corpus
- Source: https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt
```python
ratings_tr = pd.read_table("./Datasets/Naver Sentiment Movie Corpus/ratings_train.txt")
ratings_te = pd.read_table("./Datasets/Naver Sentiment Movie Corpus/ratings_test.txt")

ratings_tr = ratings_tr.dropna(subset=["document"])
ratings_te = ratings_te.dropna(subset=["document"])
```
## TED
```python
filename = "./Datasets/TED/ted_en-20160408.xml"
if not os.path.exists(filename):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/\
	GaoleMeng/RNN-and-FFNN-textClassification/master/\
	ted_en-20160408.xml", filename=filename)
with open(filename, mode="r", encoding="utf-8") as f:
    tree = etree.parse(f)
    raw_data = "\n".join(tree.xpath("//content/text()"))
```
## Portuguese-English Translation Dataset from The TED Talks Open Translation Project.
- Reference: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
```python
# `with_info`: If `True`, `tfds.load()` will return the tuple `(tf.data.Dataset, tfds.core.DatasetInfo)`, the latter containing the info associated with the builder.
# `as_supervised`: If `True`, the returned `tf.data.Dataset` will have a 2-tuple structure `(input, label)` according to `builder.info.supervised_keys`. If `False`, the returned `tf.data.Dataset` will have a dictionary with all the features.
dataset, metadata = tfds.load("ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True)
dataset_tr, dataset_val, dataset_te = dataset["train"], dataset["validation"], dataset["test"]

tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in dataset_tr), target_vocab_size=2**13)
tokenizer_tar = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in dataset_tr), target_vocab_size=2**13)

max_len = 40
buffer_size = tf.data.experimental.cardinality(dataset_tr).numpy()
batch_size = 64

def encode(lang1, lang2):
    lang1 = [tokenizer_src.vocab_size] + tokenizer_src.encode(lang1.numpy()) + [tokenizer_src.vocab_size + 1]
    lang2 = [tokenizer_tar.vocab_size] + tokenizer_tar.encode(lang2.numpy()) + [tokenizer_tar.vocab_size + 1]
    return lang1, lang2

def tf_encode(pt, en):
    # `func`: A Python function that accepts `inp` as arguments, and returns a value (or list of values) whose type is described by `Tout`.
    # `inpt`: Input arguments for func. A list whose elements are Tensors or a single Tensor.
    result_pt, result_en = tf.py_function(func=encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en

def filter_max_len(x, y):
    return tf.logical_and(tf.size(x) <= max_len, tf.size(y) <= max_len)

dataset_tr = dataset_tr.map(tf_encode)
dataset_tr = dataset_tr.filter(filter_max_len)
dataset_tr = dataset_tr.cache()
dataset_tr = dataset_tr.shuffle(buffer_size)
dataset_tr = dataset_tr.padded_batch(batch_size)
dataset_tr = dataset_tr.prefetch(tf.data.AUTOTUNE)

dataset_val = dataset_val.map(tf_encode)
dataset_val = dataset_val.filter(filter_max_len)
dataset_val = dataset_val.padded_batch(batch_size)
```

# Text
- Source: https://en.wikipedia.org/wiki/Text_(literary_theory)
- In literary theory, *a text is any object that can be "read", whether this object is a work of literature, a street sign, an arrangement of buildings on a city block, or styles of clothing.*
## Corpus (plural Corpora)
- Source: https://21centurytext.wordpress.com/home-2/special-section-window-to-corpus/what-is-corpus/
- *A corpus is a collection of texts, written or spoken, usually stored in a computer database.* A corpus may be quite small, for example, containing only 50,000 words of text, or very large, containing many millions of words.
- *Written texts in corpora might be drawn from books, newspapers, or magazines that have been scanned or downloaded electronically. Other written corpora might contain works of literature, or all the writings of one author (e.g., William Shakespeare).* Such corpora help us to see how language is used in contemporary society, how our use of language has changed over time, and how language is used in different situations.
- People build corpora of different sizes for specific reasons. For example, a very large corpus would be required to help in the preparation of a dictionary. It might contain tens of millions of words – because it has to include many examples of all the words and expressions that are used in the language. A medium-sized corpus might contain transcripts of lectures and seminars and could be used to write books for learners who need academic language for their studies. Such corpora range in size from a million words to five or ten million words. Other corpora are more specialized and much smaller. These might contain the transcripts of business meetings, for instance, and could be used to help writers design materials for teaching business language.

# NLU

# NLG

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

# Out of Vocabulary (OOV) Problem
- Used in computational linguistics and natural language processing for terms encountered in input which are not present in a system's dictionary or database of known terms.

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

	# Ignore if frequency of the token is greater than `max_df` or lower than `min_df`.
	vect = CountVectorizer(min_df, max_df, max_features)
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
	from tensorflow.keras.preprocessing.text import Tokenizer
	
	tokenizer = Tokenizer()
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
	
# BLEU (BiLingual Evaluation Understudy)
- Sources: https://en.wikipedia.org/wiki/BLEU, https://towardsdatascience.com/bleu-bilingual-evaluation-understudy-2b4eab9bcfd1
- ***BLEU is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU.*** BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.
- ***Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness are not taken into account.***
- N-gram precision: (Number of n-grams from the cadidate found in any of the reference)/(The total number of n-grams in the candidate)
	```python
	def count_ngram(cand, n):
		return Counter(nltk.ngrams(cand, n))

	def ngram_precision(refs, cand, n):
		counter_refs = Counter()
		for ref in refs:
			counter_refs += count_ngram(ref, n)
		
		ngrams_cand = count_ngram(cand, n)
		tot_cnt = 0
		for ngram, cnt in ngrams_cand.items():
			if ngram in counter_refs:
				tot_cnt += cnt 
		return tot_cnt/len(cand) - n + 1
	```
- Modified n-grams precision
	- min(Number of n-grams from the candidate found in any of the reference, Maximum total count of n-grams in any of the reference)/(The total number of n-grams in the candidate)
	```python
	def max_ref_count(ngram, refs, n):
		maxim = 0
		for ref in refs:
			ngram2cnt_ref = count_ngram(ref, n)
			if ngram2cnt_ref[ngram] > maxim:
				maxim = ngram2cnt_ref[ngram]
		return maxim

	def count_clip(ngram, cand, refs, n):
		return min(count_ngram(cand, n)[ngram], max_ref_count(ngram, refs, n))

	def modified_ngram_precision(refs, cand, n):
		sum_count_clip = 0
		for ngram, cnt in count_ngram(cand, n).items():
			sum_count_clip += count_clip(ngram, cand, refs, n)
		return sum_count_clip/(len(cand) - n + 1)
	```
- *In practice, however, using individual words as the unit of comparison is not optimal. Instead, BLEU computes the same modified precision metric using n-grams. The length which has the "highest correlation with monolingual human judgements" was found to be four.*
- *To produce a score for the whole corpus, the modified precision scores for the segments are combined using the geometric mean multiplied by a brevity penalty to prevent very short candidates from receiving too high a score.* Let `r` be the total length of the reference corpus, and `c` the total length of the translation corpus. If `c<=r`, the brevity penalty applies, defined to be `np.exp(1 - r/c)`.
- *In the case of multiple reference sentences, `r` is taken to be the minimum of the lengths of the sentences whose lengths are closest to the lengths of the candidate sentences. ("best match length")*)
	```python
	def best_match_length(refs, cand):
		ref_lens = [len(ref) for ref in refs]
		return min(ref_lens, key=lambda x:(abs(x - len(cand)), x))

	def brevity_penalty(refs, cand):
		c = len(cand)
		r = best_match_length(refs, cand)

		if c == 0:
			return 0
		else:
			if c <= r:
				return np.exp(1 - r/c)
			else:
				return 1
	```
- Implementation
	```python
	def bleu_score(refs, cand, weights=[0.25, 0.25, 0.25, 0.25]):
		ps = [modified_ngram_precision(refs, cand, n=k + 1) for k, _ in enumerate(weights)]
		score = sum([w*np.log(p) if p != 0 else 0 for w, p in zip(weights, ps)])
		return brevity_penalty(refs, cand)*np.exp(score)
	```
- Using `nltk.translate.bleu_score.sentence_bleu()`
	- Reference: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
	```python
	from nltk.translate.bleu_score import sentence_bleu
	
	# The default BLEU calculates a score for up to 4-grams using uniform
    weights (this is called BLEU-4).
	score = sentence_bleu(refs, cand, [weights=(0.25, 0.25, 0.25, 0.25)])
	```
- Using `nltk.translate.bleu_score.corpus_bleu()`
	- Reference: https://www.nltk.org/_modules/nltk/translate/bleu_score.html
	```python
	from nltk.translate.bleu_score import corpus_bleu()
	
	# The default BLEU calculates a score for up to 4-grams using uniform
    weights (this is called BLEU-4).
	score = corpus_bleu([refs], [cand], [weights=(0.25, 0.25, 0.25, 0.25)])
	```

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
	
	tokens = word_tokenize(sent)
	```
	- `"Don't"` -> `"Do"` + `"n't"`, `"Jone's"` -> `"Jone"` + `"'s"`
- Using `nltk.tokenize.TreebankWordTokenizer().tokenize()`
	```python
	from nltk.tokenize import TreebankWordTokenizer
	
	tokens = TreebankWordTokenizer().tokenize(sent)
	```
	- `"Don't"` -> `"Do"` + `"n't"`, `"Jone's"` -> `"Jone"` + `"'s"`
- Using `nltk.tokenize.WordPunctTokenizer().tokenize()`
	```python
	from nltk.tokenize import WordPunctTokenizer
	
	tokens = WordPunctTokenizer().tokenize(sent)
	```
	- `"Don't"` -> `"Don"` + `"'"` + `"t"`, `"Jone's"` -> `"Jone"` + `"'"` + `"s"`
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
	from tensorflow.keras.preprocessing.text import Tokenizer
	
	tokenizer = Tokenizer(oov_token="UNK")
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
	
	nltk.download("punkt")

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
## Split Hangul Syllables
```python
import re

# 유니코드 한글 시작 : 44032, 끝 : 55199
a, b, c = 44032, 588, 28

onsets = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
nuclei = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
codas = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

def split(string):
    res = ""
    for char in string:
     # 한글 여부 check 후 분리
        if re.match(".*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*", char) != None:
            code = ord(char) - a
            # 초성1
            onset_idx = int(code/b)
            res += onsets[onset_idx]

            # 중성
            nucleus_idx = int((code - (onset_idx*b)) / c)
            res += nuclei[nucleus_idx]

            # 종성
            coda_idx = int((code - (b*onset_idx) - (c*nucleus_idx)))
            res += codas[coda_idx]
        else:
            res += char
    return res
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
## Determine Vocabulary Size
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tr_X)
word2idx = tokenizer.word_index
cnts = sorted(tokenizer.word_counts.values(), reverse=True)
ratio = 0.99
for vocab_size, value in enumerate(np.cumsum(cnts)/np.sum(cnts)):
    if value >= ratio:
        break
print(f"{vocab_size:,}개의 단어로 전체 data의 {ratio:.0%}를 표현할 수 있습니다.")
print(f"{len(word2idx):,}개의 단어 중 {vocab_size/len(word2idx):.1%}에 해당합니다.")
```
## Determine Sequence Length
- Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
```python
# `num_words`: The maximum number of words to keep, based on word frequency. Only the most common `num_words - 1` words will be kept.
# `filters`: A string where each element is a character that will be filtered from the texts. The default is all punctuation, plus tabs and line breaks, minus the `'` character.
# `oov_token`: If given, it will be added to `word_index` and used to replace out-of-vocabulary words during `texts_to_sequence()` calls
tokenizer = Tokenizer(num_words=vocab_size + 2, oov_token="UNK")
tokenizer.fit_on_texts(train_text)
word2idx = tokenizer.word_index
word2cnt = dict(sorted(tokenizer.word_counts.items(), key=lambda x:x[1], reverse=True))

X_tr = tokenizer.texts_to_sequences(train_text)
X_te = tokenizer.texts_to_sequences(test_text)

lens = sorted([len(doc) for doc in X_tr])
ratio = 0.99
max_len = int(np.quantile(lens, 0.99))
print(f"길이가 가장 긴 문장의 길이는 {np.max(lens)}이고 길이가 {max_len} 이하인 문장이 전체의 {ratio:.0%}를 차지합니다.")
```
## Padding
```python
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

tr_X = pad_sequences(tr_X, padding="post", maxlen=max_len)
tr_y = pad_sequences(tr_y, padding="post", maxlen=max_len)

# tr_X = to_categorical(tr_X)
# tr_y = to_categorical(tr_y)
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
- Source: https://wikidocs.net/22885
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
	source: "http://nlp.stanford.edu/data/glove.6B.zip"
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
- Source: https://wikidocs.net/69141
- Word2Vec의 출력층에서는 소프트맥스 함수를 지난 단어 집합 크기의 벡터와 실제값인 원-핫 벡터와의 오차를 구하고 이로부터 임베딩 테이블에 있는 모든 단어에 대한 임베딩 벡터 값을 업데이트합니다. 만약 단어 집합의 크기가 수만 이상에 달한다면 이 작업은 굉장히 무거운 작업이므로, Word2Vec은 꽤나 학습하기에 무거운 모델이 됩니다.
- Word2Vec은 역전파 과정에서 모든 단어의 임베딩 벡터값의 업데이트를 수행하지만, 만약 현재 집중하고 있는 중심 단어와 주변 단어가 '강아지'와 '고양이', '귀여운'과 같은 단어라면, 사실 이 단어들과 별 연관 관계가 없는 '돈가스'나 '컴퓨터'와 같은 수많은 단어의 임베딩 벡터값까지 업데이트하는 것은 비효율적입니다.
- 네거티브 샘플링은 Word2Vec이 학습 과정에서 전체 단어 집합이 아니라 일부 단어 집합에만 집중할 수 있도록 하는 방법입니다. 가령, 현재 집중하고 있는 주변 단어가 '고양이', '귀여운'이라고 해봅시다. 여기에 '돈가스', '컴퓨터', '회의실'과 같은 단어 집합에서 무작위로 선택된 주변 단어가 아닌 단어들을 일부 가져옵니다. 이렇게 하나의 중심 단어에 대해서 전체 단어 집합보다 훨씬 작은 단어 집합을 만들어놓고 마지막 단계를 이진 분류 문제로 변환합니다. 주변 단어들을 긍정(positive), 랜덤으로 샘플링 된 단어들을 부정(negative)으로 레이블링한다면 이진 분류 문제를 위한 데이터셋이 됩니다. 이는 기존의 단어 집합의 크기만큼의 선택지를 두고 다중 클래스 분류 문제를 풀던 Word2Vec보다 훨씬 연산량에서 효율적입니다.

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
## Intent Classification
## Chatbot
## Neural Machine Translation (NMT)
## Natural Language Inferencing (NLI)
- SBERT를 학습하는 첫번째 방법은 문장 쌍 분류 태스크. 대표적으로는 NLI(Natural Language Inferencing) 문제를 푸는 것입니다. 다음 챕터에서 한국어 버전의 NLI 데이터인 KorNLI 문제를 BERT로 풀어볼 예정입니다. NLI는 두 개의 문장이 주어지면 수반(entailment) 관계인지, 모순(contradiction) 관계인지, 중립(neutral) 관계인지를 맞추는 문제입니다. 다음은 NLI 데이터의 예시입니다.
## Semantic Textual Similarity (STS)
- STS란 두 개의 문장으로부터 의미적 유사성을 구하는 문제를 말합니다. 다음은 STS 데이터의 예시입니다. 여기서 레이블은 두 문장의 유사도로 범위값은 0~5입니다.

# Subword Tokenizer
- Using `sentencepiece`
	- Reference: https://pypi.org/project/sentencepiece/, https://github.com/google/sentencepiece
	- 사전 토큰화 작업(pretokenization)없이 전처리를 하지 않은 데이터(raw data)에 바로 단어 분리 토크나이저를 사용할 수 있다면, 이 토크나이저는 그 어떤 언어에도 적용할 수 있는 토크나이저가 될 것입니다. 센텐스피스는 이 이점을 살려서 구현되었습니다. 센텐스피스는 사전 토큰화 작업없이 단어 분리 토큰화를 수행하므로 언어에 종속되지 않습니다.
	- Sentencepiece의 학습 데이터로는 빈 칸이 포함되지 않은 문서 집합이어야 합니다.
	- Training
		```python
		# `--input`: one-sentence-per-line raw corpus file.
		# `--model_type`: (`"unigram"`, `"bpe"`, `"char"`, `"word"`, default `"unigram"`). The input sentence must be pretokenized when using `word`.
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
		import sentencepiece as spm
		
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

	vocab_size = enc.vocab_size

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
## Byte Pair Encoding (BPE)

# Topic Modeling
- Source: https://wikidocs.net/30707
- 토픽(Topic)은 한국어로는 주제라고 합니다. 토픽 모델링(Topic Modeling)이란 기계 학습 및 자연어 처리 분야에서 토픽이라는 문서 집합의 추상적인 주제를 발견하기 위한 통계적 모델 중 하나로, 텍스트 본문의 숨겨진 의미 구조를 발견하기 위해 사용되는 텍스트 마이닝 기법입니다.
## LSA (Latent Semantic Analysis)
- Source: https://wikidocs.net/24949
- LSA는 기본적으로 DTM이나 TF-IDF 행렬에 절단된 SVD(truncated SVD)를 사용하여 차원을 축소시키고, 단어들의 잠재적인 의미를 끌어낸다는 아이디어를 갖고 있습니다.
## LDA (Latent Dirichlet Allocation)
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
### `pyLDAvis.gensim.prepare()`
```python
pyldavis = pyLDAvis.gensim.prepare(model, dtm, id2word)
```

# Language Model (LM)
## Statistical Language Model
- Source: https://en.wikipedia.org/wiki/Language_model
- ***A statistical language model is a probability distribution over sequences of words. Given such a sequence it assigns a probability to the whole sequence.***
- ***Data sparsity is a major problem in building language models. Most possible word sequences are not observed in training. One solution is to make the assumption that the probability of a word only depends on the previous n words. This is known as an n-gram model or unigram model when n equals to 1. The unigram model is also known as the bag of words model.***
## Bidirectional Language Model
- Bidirectional representations condition on both pre- and post- context (e.g., words) in all layers.

# Seq2Seq
- Source: https://en.wikipedia.org/wiki/Seq2seq
- Seq2seq turns one sequence into another sequence (sequence transformation). It does so by use of a recurrent neural network (RNN) or *more often LSTM or GRU to avoid the problem of vanishing gradient.* The context for each item is the output from the previous step. The primary components are one encoder and one decoder network. *The encoder turns each item into a corresponding hidden vector containing the item and its context. The decoder reverses the process, turning the vector into an output item, using the previous output as the input context.*
- Attention: The input to the decoder is a single vector which stores the entire context. *Attention allows the decoder to look at the input sequence selectively.*
- Beam Search: *Instead of picking the single output (word) as the output, multiple highly probable choices are retained, structured as a tree (using a Softmax on the set of attention scores). Average the encoder states weighted by the attention distribution.*
- Bucketing: Variable-length sequences are possible because of padding with 0s, which may be done to both input and output. However, if the sequence length is 100 and the input is just 3 items long, expensive space is wasted. Buckets can be of varying sizes and specify both input and output lengths.
- seq2seq는 크게 두 개로 구성된 아키텍처로 구성되는데, 바로 인코더와 디코더입니다. 인코더는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보들을 압축해서 하나의 벡터로 만드는데, 이를 컨텍스트 벡터(context vector)라고 합니다. 입력 문장의 정보가 하나의 컨텍스트 벡터로 모두 압축되면 인코더는 컨텍스트 벡터를 디코더로 전송합니다. 디코더는 컨텍스트 벡터를 받아서 번역된 단어를 한 개씩 순차적으로 출력합니다.
- 디코더는 초기 입력으로 문장의 시작을 의미하는 심볼 `<sos>`가 들어갑니다. 디코더는 `<sos>`가 입력되면, 다음에 등장할 확률이 높은 단어를 예측합니다. 첫번째 시점(time step)의 디코더 RNN 셀은 다음에 등장할 단어로 je를 예측하였습니다. 첫번째 시점의 디코더 RNN 셀은 예측된 단어 je를 다음 시점의 RNN 셀의 입력으로 입력합니다. 그리고 두번째 시점의 디코더 RNN 셀은 입력된 단어 je로부터 다시 다음에 올 단어인 suis를 예측하고, 또 다시 이것을 다음 시점의 RNN 셀의 입력으로 보냅니다. 디코더는 이런 식으로 기본적으로 다음에 올 단어를 예측하고, 그 예측한 단어를 다음 시점의 RNN 셀의 입력으로 넣는 행위를 반복합니다. 이 행위는 문장의 끝을 의미하는 심볼인 `<eos>`가 다음 단어로 예측될 때까지 반복됩니다. 지금 설명하는 것은 테스트 과정 동안의 이야기입니다.
- Applications: Chatbot, NMT, Text summarization, STT, Image captioning
- 정상적으로 정수 인코딩이 수행된 것을 볼 수 있습니다. 아직 정수 인코딩을 수행해야 할 데이터가 하나 더 남았습니다. 디코더의 예측값과 비교하기 위한 실제값이 필요합니다. 그런데 이 실제값에는 시작 심볼에 해당되는 `<sos>`가 있을 필요가 없습니다. 이해가 되지 않는다면 이전 페이지의 그림으로 돌아가 Dense와 Softmax 위에 있는 단어들을 다시 보시기 바랍니다. 그래서 이번에는 정수 인코딩 과정에서 `<sos>`를 제거합니다. 즉, 모든 프랑스어 문장의 맨 앞에 붙어있는 '\t'를 제거하도록 합니다
- 이미 RNN에 대해서 배운 적이 있지만, 다시 복습을 해보도록 하겠습니다. 하나의 RNN 셀은 각각의 Time step마다 두 개의 입력을 받습니다.
- 현재 Time step을 `t`라고 할 때, RNN 셀은 `t - 1`에서의 Hidden state와 `t`에서의 입력 벡터를 입력으로 받고, `t`에서의 Hidden state를 만듭니다. 이때 `t`에서의 Hidden state는 바로 위에 또 다른 은닉층이나 출력층이 존재할 경우에는 위의 층으로 보내거나, 필요 없으면 값을 무시할 수 있습니다. 그리고 RNN 셀은 다음 시점에 해당하는 `t + 1`의 RNN 셀의 입력으로 현재 `t`에서의 Hidden state를 입력으로 보냅니다.
- RNN 챕터에서도 언급했지만, 이런 구조에서 현재 시점 `t`에서의 Hidden state는 과거 시점의 동일한 RNN 셀에서의 모든 Hidden state의 값들의 영향을 누적해서 받아온 값이라고 할 수 있습니다. 그렇기 때문에 앞서 우리가 언급했던 Context vector는 사실 인코더에서의 마지막 RNN 셀의 Hidden state값을 말하는 것이며, 이는 입력 문장의 모든 단어 토큰들의 정보를 요약해서 담고있다고 할 수 있습니다.
- 디코더는 인코더의 마지막 RNN 셀의 Hidden state인 컨텍스트 벡터를 첫번째 Hidden state의 값으로 사용합니다. 디코더의 첫번째 RNN 셀은 이 첫번째 Hidden state의 값과, 현재 `t`에서의 입력값인 `"<SOS>"`로부터, 다음에 등장할 단어를 예측합니다.
## Teacher Forcing
- 모델을 설계하기 전에 혹시 의아한 점은 없으신가요? 현재 시점의 디코더 셀의 입력은 오직 이전 디코더 셀의 출력을 입력으로 받는다고 설명하였는데 `dec_input`이 왜 필요할까요?
- 훈련 과정에서는 이전 시점의 디코더 셀의 출력을 현재 시점의 디코더 셀의 입력으로 넣어주지 않고, 이전 시점의 실제값을 현재 시점의 디코더 셀의 입력값으로 하는 방법을 사용할 겁니다. 그 이유는 이전 시점의 디코더 셀의 예측이 틀렸는데 이를 현재 시점의 디코더 셀의 입력으로 사용하면 현재 시점의 디코더 셀의 예측도 잘못될 가능성이 높고 이는 연쇄 작용으로 디코더 전체의 예측을 어렵게 합니다. 이런 상황이 반복되면 훈련 시간이 느려집니다. 만약 이 상황을 원하지 않는다면 이전 시점의 디코더 셀의 예측값 대신 실제값을 현재 시점의 디코더 셀의 입력으로 사용하는 방법을 사용할 수 있습니다.
## Character-Level seq2seq
- Training
	```python
	
	```
- Inference
	- 앞서 seq2seq는 훈련할 때와 동작할 때의 방식이 다르다고 언급한 바 있습니다. 이번에는 입력한 문장에 대해서 기계 번역을 하도록 모델을 조정하고 동작시켜보도록 하겠습니다.
	- 전체적인 번역 동작 단계를 정리하면 아래와 같습니다.
		1. 번역하고자 하는 입력 문장이 인코더에 들어가서 Hidden state와 셀 상태를 얻습니다.
		2. 상태와 `<SOS>`를 디코더로 보냅니다.
		3. 디코더가 `<EOS>`가 나올 때까지 다음 문자를 예측하는 행동을 반복합니다.
		
# Greedy Search & Beam Search
## Greedy Search
```python
def greedy_search(data):
    return np.argmax(data, axis=1)
```
## Beam Search
- Source: https://en.wikipedia.org/wiki/Beam_search, https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24
- In computer science, beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. Best-first search is a graph search which orders all partial solutions (states) according to some heuristic. But in beam search, *only a predetermined number of best partial solutions are kept as candidates. It is thus a greedy algorithm*.
- *Beam search uses breadth-first search to build its search tree.* At each level of the tree, it generates all successors of the states at the current level, sorting them in increasing order of heuristic cost. However, it only stores a predetermined number of best states at each level (called the beam width). Only those states are expanded next. The greater the beam width, the fewer states are pruned. With an infinite beam width, no states are pruned and beam search is identical to breadth-first search.* The beam width bounds the memory required to perform the search.
- For example, *beam search has been used in many machine translation systems.* To select the best translation, each part is processed, and many different ways of translating the words appear. *The top best translations according to their sentence structures are kept, and the rest are discarded. The translator then evaluates the translations according to a given criterion, choosing the translation which best keeps the goals.*
```python
def beam_search(data, k):
    seq_score = [[list(), 0]]
    for probs in data:
        cands = list()
        for seq, score in seq_score:
            for i, prob in enumerate(probs):
                cands.append([seq + [i], score - np.log(prob)])
        seq_score.extend(cands)
        seq_score = sorted(seq_score, key=lambda x:x[1])[:k]
    return [np.array(i[0]) for i in seq_score]
```

# Attention
- Sources: https://en.wikipedia.org/wiki/Attention_(machine_learning), https://wikidocs.net/22893
- In neural networks, attention is a technique that mimics cognitive attention. *The effect enhances some parts of the input data while diminishing other parts — the thought being that the network should devote more focus to that small but important part of the data. Learning which part of the data is more important than others depends on the context and is trained by gradient descent.*
- To build a machine that translates English-to-French, one starts with an Encoder-Decoder and grafts an attention unit to it. *In practice, the attention unit consists of 3 fully connected neural network layers that needs to be trained. The 3 layers are called Query, Key, and Value.*
- Self Attention: Query, Key, Value의 출처가 서로 동일한 경우를 말합니다.
- Multi-head Attention: Attention을 Parallel하게 수행한다는 의미입니다.
## Dot-Product Attention (= Luong Attention)
- Implementation
	```python
	def dot_product_attention(queries, keys, values, mask=None):
		attn_scores = tf.matmul(queries, keys, transpose_b=True)
		# (batch_size, seq_len, seq_len)
		attn_weights = tf.nn.softmax(attn_scores, axis=-1)
		# (batch_size, seq_len, dk)
		context_vec = tf.matmul(attn_weights, values)
		return context_vec, attn_weights
	```
## Scaled Dot-Product Attention (for Transformer)
- Implementation
	```python
	# 패딩 마스킹을 써야하는 경우에는 스케일드 닷 프로덕트 어텐션 함수에 패딩 마스크를 전달하고
	# 룩-어헤드 마스킹을 써야하는 경우에는 스케일드 닷 프로덕트 어텐션 함수에 룩-어헤드 마스크를 전달합니다.
	def scaled_dot_product_attention(queries, keys, values, mask):
		attn_scores = tf.matmul(queries, keys, transpose_b=True)/dk**0.5
		if mask is not None:
			attn_scores = attn_scores + (mask*-1e9)
		attn_weights = tf.nn.softmax(attn_scores, axis=-1)
		context_vec = tf.matmul(attn_weights, values)
		return context_vec, attn_weights
	```
## Bahdanau Attention (= Concat Attention)
- Implementation
	```python
	class BahdanauAttention(Model):
		def __init__(self, units):
			super(BahdanauAttention, self).__init__()
			self.W1 = Dense(units=units)
			self.W2 = Dense(units=units)
			self.W3 = Dense(units=1)

		# The keys is same as the values 
		def call(self, values, query):
			# (batch_size, h_size) -> (batch_size, 1, h_size)
			query = tf.expand_dims(query, 1)

			attn_scores = self.W3(tf.nn.tanh(self.W1(values) + self.W2(query)))
			attn_weights = tf.nn.softmax(attn_scores, axis=1)

			# Attention value
			# (batch_size, h_size)
			context_vec = tf.reduce_sum(attn_weights*values, axis=1)

			return context_vec, attn_weights
	```

# Transformer
- Sources: https://en.wikipedia.org/wiki/Transformer_(machine_learning_model), https://wikidocs.net/31379, https://www.tensorflow.org/text/tutorials/transformer
- *A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the field of natural language processing (NLP) and in computer vision (CV).*
- Like recurrent neural networks (RNNs), transformers are designed to handle sequential input data, such as natural language, for tasks such as translation and text summarization. However, ***unlike RNNs, transformers do not necessarily process the data in order. Rather, the attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not need to process the beginning of the sentence before the end. Rather, it identifies the context that confers meaning to each word in the sentence. This feature allows for more parallelization than RNNs and therefore reduces training times.***
- *The additional training parallelization allows training on larger datasets than was once possible. This led to the development of pretrained systems such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer), which were trained with large language datasets, such as the Wikipedia Corpus and Common Crawl, and can be fine-tuned for specific tasks.*
- Before transformers, most state-of-the-art NLP systems relied on gated RNNs, such as LSTM and gated recurrent units (GRUs), with added attention mechanisms. *Transformers are built on these attention technologies without using an RNN structure, highlighting the fact that attention mechanisms alone can match the performance of RNNs with attention.*
- *Sequential processing: Gated RNNs process tokens sequentially, maintaining a state vector that contains a representation of the data seen after every token.* To process the `n`th token, the model combines the state representing the sentence up to token `n - 1` with the information of the new token to create a new state, representing the sentence up to token `n`. Theoretically, the information from one token can propagate arbitrarily far down the sequence, if at every point the state continues to encode contextual information about the token. In practice this mechanism is flawed: ***the vanishing gradient problem leaves the model's state at the end of a long sentence without precise, extractable information about preceding tokens.*** *The dependency of token computations on results of previous token computations also makes it hard to parallelize computation on modern deep learning hardware. This can make the training of RNNs inefficient.*
- Attention
	- These problems were addressed by attention mechanisms. Attention mechanisms let a model draw from the state at any preceding point along the sequence. The attention layer can access all previous states and weigh them according to a learned measure of relevancy, providing relevant information about far-away tokens.*
	- A clear example of the value of attention is in language translation, where context is essential to assign the meaning of a word in a sentence. In an English-to-French translation system, the first word of the French output most probably depends heavily on the first few words of the English input. *However, in a classic LSTM model, in order to produce the first word of the French output, the model is given only the state vector of the last English word. Theoretically, this vector can encode information about the whole English sentence, giving the model all necessary knowledge. In practice, this information is often poorly preserved by the LSTM. An attention mechanism can be added to address this problem: the decoder is given access to the state vectors of every English input word, not just the last, and can learn attention weights that dictate how much to attend to each English input state vector.*
- *When added to RNNs, attention mechanisms increase performance. The development of the Transformer architecture revealed that attention mechanisms were powerful in themselves and that sequential recurrent processing of data was not necessary to achieve the quality gains of RNNs with attention. Transformers use an attention mechanism without an RNN, processing all tokens at the same time and calculating attention weights between them in successive layers. Since the attention mechanism only uses information about other tokens from lower layers, it can be computed for all tokens in parallel, which leads to improved training speed.*
- Scaled dot-product attention
	- Whenever we are required to calculate the Attention of a target word with respect to the input embeddings, we should use the Query of the target and the Key of the input to calculate a matching score, and these matching scores then act as the weights of the Value vectors during summation.
	- Reference: https://wikidocs.net/31379
	- ![Scaled Dot-Product Attention](https://wikidocs.net/images/page/31379/transformer16.PNG)
	- `mask`를 사용하는 이유
		- 패딩한 부분을 어텐션 연산에 참여시키지 않기 위해
		- Seq2Seq에서는 인코딩 부분과 단어 일부로 다음 단어를 예측하기 때문에 단어 전체가 아닌 일부분만을 학습시키기 위해
	- Implementation
		```python
		def scaled_dot_product_attention(queries, keys, values, mask=None):
			attn_scores = tf.matmul(queries, keys, transpose_b=True)/dk**0.5
			if mask is not None:
				attn_scores = attn_scores + (mask*-1e9)
			# (batch_size, seq_len_dec, seq_len_enc)
			attn_weights = tf.nn.softmax(attn_scores, axis=-1)
			# (batch_size, seq_len_dec, dk) (Same shape as queries)
			context_vec = tf.matmul(attn_weights, values)
			return context_vec, attn_weights
		```
- Multi-head attention
	- One set of {\displaystyle \left(W_{Q},W_{K},W_{V}\right)}{\displaystyle \left(W_{Q},W_{K},W_{V}\right)} matrices is called an attention head, and each layer in a transformer model has multiple attention heads. While each attention head attends to the tokens that are relevant to each token, with multiple attention heads the model can do this for different definitions of "relevance". In addition the influence field representing relevance can become progressively dilated in successive layers. Many transformer attention heads encode relevance relations that are meaningful to humans. For example, attention heads can attend mostly to the next word, while others mainly attend from verbs to their direct objects.[8] The computations for each attention head can be performed in parallel, which allows for fast processing. The outputs for the attention layer are concatenated to pass into the feed-forward neural network layers.
	- 멀티 헤드 어텐션은 전체 어텐션을 분리하여 병렬적으로 어텐션을 수행하는 기법입니다. 즉 `(batch_size, 50, 64*8)` 의 텐서가 있다면 이것을 `(batch_size, 50, 64)`의 8개의 텐서로 나눈다음에 개별적으로 어텐션을 수행하고 (각각을 Attention head라고 부름), 다시 `(batch_size, 50, 64*8)`의 텐서로 Concat하게 됩니다. 이렇게 하는 이유는, 깊은 차원을 한번에 어텐션을 수행하는 것보다, 병렬로 각각 수행하는 것이 더 심도있는 언어들간의 관계를 학습할 수 있기 때문입니다.
	- 예를 들어보겠습니다. 앞서 사용한 예문 '그 동물은 길을 건너지 않았다. 왜냐하면 그것은 너무 피곤하였기 때문이다.'를 상기해봅시다. 단어 그것(it)이 쿼리였다고 해봅시다. 즉, it에 대한 Q벡터로부터 다른 단어와의 연관도를 구하였을 때 첫번째 어텐션 헤드는 '그것(it)'과 '동물(animal)'의 연관도를 높게 본다면, 두번째 어텐션 헤드는 '그것(it)'과 '피곤하였기 때문이다(tired)'의 연관도를 높게 볼 수 있습니다. 각 어텐션 헤드는 전부 다른 시각에서 보고있기 때문입니다.
	- `d_model`을 `n_heads`로 나눈 값을 각 Q, K, V의 차원을 결정합니다.
	- Implementation
		```python
		class MultiheadAttention(Layer):
			def __init__(self):
				super().__init__()

			def split_heads(self, x):
				x = tf.reshape(x, shape=(batch_size, -1, n_heads, dk))
				return tf.transpose(x, perm=[0, 2, 1, 3])

			def call(self, values, keys, queries, mask):
				queries = Dense(units=d_model)(queries)
				keys = Dense(units=d_model)(keys)
				values = Dense(units=d_model)(values)

				batch_size = tf.shape(queries)[0]
				# (batch_size, n_heads, seq_len_dec, dk)
				queries = self.split_heads(queries)
				# (batch_size, n_heads, seq_len_enc, dk)
				keys = self.split_heads(keys)
				# (batch_size, n_heads, seq_len_enc, dk)
				values = self.split_heads(values)

				# (batch_size, n_heads, seq_len_dec, dk)
				context_vec, attn_weights = scaled_dot_product_attention(queries, keys, values, mask)
				# (batch_size, seq_len_dec, n_heads, dk)
				z = tf.transpose(context_vec, perm=[0, 2, 1, 3])
				# (batch_size, seq_len_dec, d_model)
				z = tf.reshape(z, shape=(batch_size, -1, d_model))
				z = Dense(units=d_model)(z)
				return z, attn_weights
		```
- Encoder
	- *The first encoder takes positional information and embeddings of the input sequence as its input, rather than encodings. The positional information is necessary for the transformer to make use of the order of the sequence, because no other part of the transformer makes use of this.*
- Decoder
	- *Each decoder consists of three major components: a self-attention mechanism, an attention mechanism over the encodings, and a feed-forward neural network.* The decoder functions in a similar fashion to the encoder, but an additional attention mechanism is inserted which instead draws relevant information from the encodings generated by the encoders.
	- *Like the first encoder, the first decoder takes positional information and embeddings of the output sequence as its input, rather than encodings. The transformer must not use the current or future output to predict an output, so the output sequence must be partially masked to prevent this reverse information flow.* The last decoder is followed by a final linear transformation and softmax layer, to produce the output probabilities over the vocabulary.
	- 디코더는 인코더랑 유사하지만, 구조가 약간 다릅니다. 이번 Seq2Seq는 포르투갈어를 영어로 바꾸는 문제입니다.  디코더에서는 두단계의 멀티 헤드 어텐션 구조를 거치는데, 첫번째 멀티 헤드 어텐션은 영어문장과 영어문장의 셀프 어텐션을 하여 영어 문장간의 관계를 배우게 됩니다. 두 번째 멀티 헤드 어텐션은 포르투갈어가 인코딩 된 것과 영어 문장간의 셀프 어텐션된 결과를 다시 어텐션 해서 포르투갈 어와 영어의 관계를 학습하게 됩니다.
- ![Transformer Architecture](https://i.imgur.com/Tl2zsFL.png)
- ![Transformer Architecture (2)](https://i.imgur.com/w4n19Rs.png)
- Positional Encoding
	- Implementation
		```python
		# `d_model` is the number of dimensions, `seq_len` is the length of input sequence.
		def positional_encoding_matrix(seq_len, d_model):
			a, b = np.meshgrid(np.arange(d_model), np.arange(seq_len))
			pe_mat = b/10000**(2*(a//2)/d_model)
			pe_mat[:, 0::2] = np.sin(pe_mat[:, 0::2])
			pe_mat[:, 1::2] = np.cos(pe_mat[:, 1::2])
			pe_mat = pe_mat[None, :]
			return pe_mat
		```
	- Visualization
		```python
		pe_mat = positional_encoding_matrix(seq_len, d_model)[0]

		plt.figure(figsize=(10, 6))
		plt.pcolormesh(pe_mat, cmap="RdBu");
		plt.gca().invert_yaxis()
		plt.colorbar();
		```
- 셀프 어텐션은 인코더의 초기 입력인 `d_model`의 차원을 가지는 단어 벡터들을 사용하여 셀프 어텐션을 수행하는 것이 아니라 우선 각 단어 벡터들로부터 Q벡터, K벡터, V벡터를 얻는 작업을 거칩니다. 이때 이 Q벡터, K벡터, V벡터들은 초기 입력인 `d_model`의 차원을 가지는 단어 벡터들보다 더 작은 차원을 가지는데, 논문에서는 512의 차원을 가졌던 각 단어 벡터들을 64의 차원을 가지는 Q벡터, K벡터, V벡터로 변환하였습니다.
- Look-ahead Mask
	- To prevent the model from peeking at the expected output the model uses a look-ahead mask.
	- 포르투갈어가 암호화된 것과, 영어 문장 한단어 한단어를 보면서 다음 단어를 예측하게 되기 때문에, look_ahead_mask를 사용하게 됩니다. 만약 영어 문장이 (I love you) 로 이루어져 있다면, look_ahead_mask를 사용하면, (I, 0, 0) -> Love 예측, (I love, 0) -> You 예측, (I love you) -> 단어의 끝인 [SEP] 예측을 합니다.
	- 즉 look_ahead_mask는 다음 단어를 예측할 때, 전에 있던 단어만으로 예측할수 있도록 앞에 있는 단어는 가리는 것입니다. 이러한 역할을 가능하게 하는 mask가 look_ahead_mask 입니다.
	- 위 그림과 같이 디코더도 인코더와 동일하게 임베딩 층과 포지셔널 인코딩을 거친 후의 문장 행렬이 입력됩니다. 트랜스포머 또한 seq2seq와 마찬가지로 교사 강요(Teacher Forcing)을 사용하여 훈련되므로 학습 과정에서 디코더는 번역할 문장에 해당되는 <sos> je suis étudiant의 문장 행렬을 한 번에 입력받습니다. 그리고 디코더는 이 문장 행렬로부터 각 시점의 단어를 예측하도록 훈련됩니다.
	- 여기서 문제가 있습니다. seq2seq의 디코더에 사용되는 RNN 계열의 신경망은 입력 단어를 매 시점마다 순차적으로 입력받으므로 다음 단어 예측에 현재 시점을 포함한 이전 시점에 입력된 단어들만 참고할 수 있습니다. 반면, 트랜스포머는 문장 행렬로 입력을 한 번에 받으므로 현재 시점의 단어를 예측하고자 할 때, 입력 문장 행렬로부터 미래 시점의 단어까지도 참고할 수 있는 현상이 발생합니다. 가령, suis를 예측해야 하는 시점이라고 해봅시다. RNN 계열의 seq2seq의 디코더라면 현재까지 디코더에 입력된 단어는 <sos>와 je뿐일 것입니다. 반면, 트랜스포머는 이미 문장 행렬로 <sos> je suis étudiant를 입력받았습니다.
	- 이를 위해 트랜스포머의 디코더에서는 현재 시점의 예측에서 현재 시점보다 미래에 있는 단어들을 참고하지 못하도록 룩-어헤드 마스크(look-ahead mask)를 도입했습니다. 직역하면 '미리보기에 대한 마스크'입니다.
	- 룩-어헤드 마스크(look-ahead mask)는 디코더의 첫번째 서브층에서 이루어집니다. 디코더의 첫번째 서브층인 멀티 헤드 셀프 어텐션 층은 인코더의 첫번째 서브층인 멀티 헤드 셀프 어텐션 층과 동일한 연산을 수행합니다. 오직 다른 점은 어텐션 스코어 행렬에서 마스킹을 적용한다는 점만 다릅니다. 우선 다음과 같이 셀프 어텐션을 통해 어텐션 스코어 행렬을 얻습니다.
	- 이제 자기 자신보다 미래에 있는 단어들은 참고하지 못하도록 다음과 같이 마스킹합니다.
	- 인코더의 셀프 어텐션 : 패딩 마스크를 전달
	- 디코더의 첫번째 서브층인 마스크드 셀프 어텐션 : 룩-어헤드 마스크를 전달 <-- 지금 설명하고 있음.
	- 디코더의 두번째 서브층인 인코더-디코더 어텐션 : 패딩 마스크를 전달
	- 참고로 패딩은 1로 하겠습니다. 왜냐하면 어텐션 부분에서 mask * (-1e9)를 하는데, 패딩이 1이어야 -1e9가 곱해져서 상당히 음수로 큰 수가 되는 것이고, 이게 소프트 맥스에 들어가면 0이 되기 때문입니다.(지수함수라 지수함수에 -음수는 0으로 수렴)

# BERT (Bidirectional Encoder Representations from Transformers)
- ![BERT Embeddings](https://mino-park7.github.io/images/2019/02/bert-input-representation.png)
	- Token embeddings: 버트 모형에 들어가는 인풋은 일정한 길이를 가져야 합니다.(본 예제에서는 64)
	따라서 남는 부분은 1로 채워지게 됩니다(패딩)
	- Segment embeddings: 세그멘트 인풋은 문장이 앞문장인지 뒷문장인지 구분해주는 역할을 하는데요 본 문장에서는 문장 하나만 인풋으로 들어가기 때문에 0만 들어가게 되고, 문장 길이만큼의 0이 인풋으로 들어가게 됩니다.
	- Position embeddings: 패딩이 아닌 부분은 1, 패딩인 부분은 0.
- Using `tokenization_kobert` (KoBERT)
	```python
	!pip install transformers==4.4.2
	!pip install sentencepiece

	import urllib	urllib.request.urlretrieve("https://raw.githubusercontent.com/monologg/KoBERT-Transformers/master/kobert_transformers/tokenization_kobert.py", filename="tokenization_kobert.py")

	from tokenization_kobert import KoBertTokenizer

	tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

	# `"[PAD]"`: `1`, `"[CLS]"`: `2`, `"[SEP]"`: `3`
		# 모든 Sentence의 첫번째 Token은 언제나 `"[CLS]"`(special classification token)입니다. 이것은 ㅆransformer 전체층을 다 거치고 나면 ㅆoken sequence의 결합된 의미를 가지게 되는데, 여기에 간단한 Classifier를 붙이면 단일 문장, 또는 연속된 문장의 classification을 쉽게 할 수 있게 됩니다. 만약 classification task가 아니라면 이 token은 무시하면 됩니다. (https://tmaxai.github.io/post/BERT/)
	pieces = tokenizer.tokenize(text)
	# `max_length`
	# `truncation`
		# `True`: Explicitly truncate examples to `max_length`
		# 'longest_first'
	# `padding`
		# `True` or `"longest"`: Pad to the longest sequence in the batch.
		# `"max_length"`: Pad to `max_length`.
	ids = tokenizer.encode(text)
	sents = tokenizer.decode(ids)
	id = tokenizer.convert_tokens_to_ids(piece)
	```

# Sentence BERT (= SBERT)
- Paper: https://arxiv.org/abs/1908.10084
- Source: https://www.sbert.net/, https://wikidocs.net/156176
- 사전 학습된 BERT로부터 문장 벡터를 얻는 방법은 다음과 같이 세 가지가 있습니다.
	- BERT의 [CLS] 토큰의 출력 벡터를 문장 벡터로 간주한다.
	- BERT의 모든 단어의 출력 벡터에 대해서 평균 풀링을 수행한 벡터를 문장 벡터로 간주한다.
	- BERT의 모든 단어의 출력 벡터에 대해서 맥스 풀링을 수행한 벡터를 문장 벡터로 간주한다.
- SBERT는 기본적으로 BERT의 문장 임베딩의 성능을 우수하게 개선시킨 모델입니다. SBERT는 위에서 언급한 BERT의 문장 임베딩을 응용하여 BERT를 파인 튜닝합니다.
- Source: https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
- Each of those 512 tokens has a respective 768 values. This pooling operation will take the mean of all token embeddings and compress them into a single 768 vector space — creating a "sentence vector".
```python
# https://huggingface.co/models?library=sentence-transformers
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
```

# GPT (Generative Pre-trained Transformer)

# ElMo
```python
import tensorflow_hub as hub

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
embeddings = elmo(["the cat is on the mat", "dogs are in the fog"], signature="default", as_dict=True)["elmo"]
```

# `soynlp`
```python
from soynlp.normalizer import *
```
## `emoticon_normalize(num_repeats)`
## `repeat_normalize(num_repeats)`

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
	- Source: https://github.com/Pusnow/mecab-python-msvc/releases/tag/mecab_python-0.996_ko_0.9.2_msvc-2
	```python
	!pip install mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl
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

# `gensim`
```python
import gensim
```
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
#### `gensim.corpora.BleiCorpus.serizalize()`
```python
gensim.corpora.BleiCorpus.serialize("kakotalk dtm", dtm)
```
#### `gensim.corpora.bleicorpus.BleiCorpus()`
```python
dtm = gensim.corpora.bleicorpus.BleiCorpus("kakaotalk dtm")
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

# Regular Expression
```python
 import re
```
- The meta-characters which do not match themselves because they have special meanings are: `.`, `^`, `$`, `*`, `+`, `?`, `{`, `}`, `[`, `]`, `(`, `)`, `\`, `|`.
- `.`: Match any single character except newline.
- `\n`: Newline
- `\r`: Return
- `\t`: Tab
- `\f`: Form
- `\w`, `[a-zA-Z0-9_]`: Match any single "word" character: a letter or digit or underbar. 
- `\W`, `[^a-zA-Z0-9_]`: Match any single non-word character.
- `\s`, `[ \n\r\t\f]`: Match any single whitespace character(space, newline, return, tab, form).
- `\S`: Match any single non-whitespace character.
- `[ㄱ-ㅣ가-힣]`: 어떤 한글
- `\d`, `[0-9]`: Match any single decimal digit.
- `\D`, `[^0-9]`: Match any single non-decimal digit.
- `\*`: 0개 이상의 바로 앞의 character(non-greedy way)
- `\+`: 1개 이상의 바로 앞의 character(non-greedy way)
- `\?`: 1개 이하의 바로 앞의 character
- `{m, n}`: m개~n개의 바로 앞의 character(생략된 m은 0과 동일, 생략된 n은 무한대와 동일)(non-greedy way)
- `{n}`: n개의 바로 앞의 character
- `^`: Match the start of the string.
- `$`: Match the end of the string.
## `re.search()`
- Scan through string looking for the first location where the regular expression pattern produces a match, and return a corresponding match object. Return None if no position in the string matches the pattern; note that this is different from finding a zero-length match at some point in the string.
## `re.match()`
- If zero or more characters at the beginning of string match the regular expression pattern, return a corresponding match object. Return None if the string does not match the pattern; note that this is different from a zero-length match.
## `re.findall()`
- Return all non-overlapping matches of pattern in string, as a list of strings. The string is scanned left-to-right, and matches are returned in the order found. If one or more groups are present in the pattern, return a list of groups; this will be a list of tuples if the pattern has more than one group. Empty matches are included in the result.
## `re.split(maxsplit)`
## `re.sub(expr, count)`
## `re.compile()`

# Install `khaiii` on Google Colab
```
!git clone https://github.com/kakao/khaiii.git
!pip install cmake
!mkdir build
!cd build && cmake /content/khaiii
!cd /content/build/ && make all
!cd /content/build/ && make resource
!cd /content/build && make install
!cd /content/build && make package_python
!pip install /content/build/package_python
```
```
!git clone https://github.com/kakao/khaiii.git
!pip install cmake
!mkdir build
# !cd build
!cd build && cmake /content/drive/MyDrive/Libraries/khaiii
!cd build && make all
# !cd build && make resource
# !cd build && make install
# !cd build && make package_python
# !cd package_python
# !cd build/package_python && !pip install package_python
```