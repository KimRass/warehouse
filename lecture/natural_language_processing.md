# Out of Vocabulary (OOV) Problem
- Used in computational linguistics and natural language processing for terms encountered in input which are not present in a system's dictionary or database of known terms.

# Preprocess
## Regularization
### Word Regularization
```python
text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
```
- Using `nltk.tokenize.word_tokenize()`
	```python
	from nltk.tokenize import word_tokenize
	
	nltk.download("punkt")
	nltk.download("averaged_perceptron_tagger")

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
### Sentence Regularization
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
	# Reference: https://github.com/hyunwoongko/kss
	# `use_heuristic`: If you set it `True`, Kss conduct open-ended segmentation. If you set it `False`, Kss conduct punctuation-only segmentation.
	# `use_quotes_brackets_processing`: This parameter indicates whether to segment the parts enclosed in brackets or quotations marks. If you set it `True`, Kss does not segment these parts, If you set it `False`, Kss segments the even in the parts that are enclosed in brackets and quotations marks.
	# `backend`: (`"pynori"`, `"mecab"`)
	# `num_workers`
	sent_tokens = kss.split_sentences(text)
	```
- Using `kiwipiepy.Kiwi().split_into_sents()` (for Korean language)
## Spacing
- Using `pykospacing.spacing()`
	```python
	!pip install git+https://github.com/haven-jeon/PyKoSpacing.git --user
	```
	```python
	from pykospacing import spacing

	# Example
	text = "오지호는극중두얼굴의사나이성준역을맡았다.성준은국내유일의태백권전승자를가리는결전의날을앞두고20년간동고동락한사형인진수(정의욱분)를찾으러속세로내려온인물이다."
	sent_div = spacing(text)
	```
## Stemming & Lemmatization
### Stemming
- Reference: https://builtin.com/data-science/introduction-nlp
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
- Reference: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
-  Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.
- Using `nltk.stem.WordNetLemmatizer().lemmatize()`
	```python
	import nltk
	from nltk.stem import WordNetLemmatizer
	
	nltk.download("wordnet")
	wnl = WordNetLemmatizer()
	
	words = ["policy", "doing", "organization", "have", "going", "love", "lives", "fly", "dies", "watched", "has", "starting"]
	
	lemmas = [wnl.lemmatize(w) for w in words]
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

# Language Model (LM)
## Statistical Language Model
- Reference: https://en.wikipedia.org/wiki/Language_model
- ***A statistical language model is a probability distribution over sequences of words. Given such a sequence it assigns a probability to the whole sequence.***
- ***Data sparsity is a major problem in building language models. Most possible word sequences are not observed in training. One solution is to make the assumption that the probability of a word only depends on the previous n words. This is known as an n-gram model or unigram model when n equals to 1. The unigram model is also known as the bag of words model.***
## Bidirectional Language Model
- Bidirectional representations condition on both pre- and post- context (e.g., words) in all layers.

# Seq2Seq
- Reference: https://en.wikipedia.org/wiki/Seq2seq
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
- Reference: https://en.wikipedia.org/wiki/Beam_search, https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24
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
