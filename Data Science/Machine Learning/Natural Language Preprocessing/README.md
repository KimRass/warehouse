# Language Model (LM)
## Statistical Language Model
- Source: https://en.wikipedia.org/wiki/Language_model
- ***A statistical language model is a probability distribution over sequences of words. Given such a sequence it assigns a probability to the whole sequence.***
- ***Data sparsity is a major problem in building language models. Most possible word sequences are not observed in training. One solution is to make the assumption that the probability of a word only depends on the previous n words. This is known as an n-gram model or unigram model when n equals to 1. The unigram model is also known as the bag of words model.***
## Bidirectional Language Model
- Bidirectional representations condition on both pre- and post- context (e.g., words) in all layers.

# Word Embedding
- In natural language processing (NLP), word embedding is a term used for the representation of words for text analysis, ***typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning.*** Word embeddings can be obtained using a set of language modeling and feature learning techniques where words or phrases from the vocabulary are mapped to vectors of real numbers. ***Conceptually it involves the mathematical embedding from space with many dimensions per word to a continuous vector space with a much lower dimension.***
- Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing and sentiment analysis.

# Syntactic & Semantic Analysis
- Source: https://builtin.com/data-science/introduction-nlp
- Syntactic analysis (syntax) and semantic analysis (semantic) are the two primary techniques that lead to the understanding of natural language. Language is a set of valid sentences, but what makes a sentence valid? Syntax and semantics.
- ***Syntax is the grammatical structure of the text, whereas semantics is the meaning being conveyed. A sentence that is syntactically correct, however, is not always semantically correct. For example, “cows flow supremely” is grammatically valid (subject — verb — adverb) but it doesn't make any sense.***

# Preprocessing
## Stemming
- Source: https://builtin.com/data-science/introduction-nlp
- Basically, stemming is the process of reducing words to their word stem. A "stem" is the part of a word that remains after the removal of all affixes. For example, the stem for the word "touched" is "touch." "Touch" is also the stem of "touching," and so on.
- You may be asking yourself, why do we even need the stem? Well, *the stem is needed because we're going to encounter different variations of words that actually have the same stem and the same meaning.*Now, imagine all the English words in the vocabulary with all their different fixations at the end of them. To store them all would require a huge database containing many words that actually have the same meaning. This is solved by focusing only on a word’s stem. Popular algorithms for stemming include the Porter stemming algorithm from 1979, which still works well.

## NER (Named Entity Recognition)
- Source: https://builtin.com/data-science/introduction-nlp
- *Named entity recognition (NER) concentrates on determining which items in a text (i.e. the "named entities") can be located and classified into pre-defined categories. These categories can range from the names of persons, organizations and locations to monetary values and percentages.*
## Sentiment Analysis
- *With sentiment analysis we want to determine the attitude (i.e. the sentiment) of a speaker or writer with respect to a document, interaction or event. Therefore it is a natural language processing problem where text needs to be understood in order to predict the underlying intent. The sentiment is mostly categorized into positive, negative and neutral categories.*

# NLU
# NLG

# TF-IDF(Term Frequency-Inverse Document Frequency)
- Source: https://wikidocs.net/31698
- TF-IDF는 특정 문서에서 자주 등장하는 단어는 그 문서 내에서 중요한 단어로 판단
## tf(d, t)
- 특정 문서 d에서의 특정 단어 t의 등장 횟수.
- tf를 이어 붙인 것은 DTM과 같다.
## df(t)
- 특정 단어 t가 등장한 문서 d의 수.
- 여기서 특정 단어가 각 문서, 또는 문서들에서 몇 번 등장했는지는 관심가지지 않으며 오직 등장한 문서의 수에만 관심을 가집니다.
## idf(d, t)
- ![image.png](/wikis/2670857615939396646/files/2805180087290841767)
- n: 문서의 전체 개수.

# Latent Dirichlet Allocation
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
