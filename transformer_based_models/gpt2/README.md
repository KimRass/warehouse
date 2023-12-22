# Paper Summary
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
## Related Works
- The current best performing systems on language tasks utilize a combination of pre-training and supervised fine-tuning. This approach has a long history with a trend towards more flexible forms of transfer. First, word vectors were learned and used as inputs to task-specific architec-tures, then the contextual representations of recurrent networks were transferred [1], and recent work suggests that task-specific architectures are no longer necessary and transferring many self-attention blocks is sufficient [2] [3].
- These methods still require supervised training in order to perform a task. When only minimal or no supervised data is available, another line of work has demonstrated the promise of language models to perform specific tasks, such as commonsense reasoning and sentiment analysis [2].
## Methodology
- Unsupervised multitask & zero-shot learning
    - We would like to move towards more general systems which can perform many tasks – eventually without the need to manually create and label a training dataset for each one.
    - Our suspicion is that the prevalence of single task training on single domain datasets is a major contributor to the lack of generalization observed in current systems.
    - We demonstrate language models can perform down-stream tasks in a zero-shot setting – without any parameter or architecture modification.
    - Learning to perform a single task can be expressed in a probabilistic framework as estimating a conditional distribution $p(output \mid input)$. ***Since a general system should be able to perform many different tasks, even for the same input, it should condition not only on the input but also on the task to be performed. That is, it should model*** $p(output \mid input, task)$***.***
    - Language provides a flexible way to specify tasks, inputs, and outputs all as a sequence of symbols. For example, a translation training example can be written as the sequence '(translate to french, english text, french text)'. Likewise, a reading comprehension training example can be written as '(answer the question, document, question, answer)'.
    - Our speculation is that a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement. If a language model is able to do this it will be, in effect, performing unsupervised multitask learning. We test whether this is the case by analyzing the performance of language models in a zero-shot setting on a wide variety of tasks.
- Tokenization
    - ***Current large scale LMs include pre-processing steps such as lower-casing, tokenization, and out-of-vocabulary tokens which restrict the space of model-able strings.*** While processing Unicode strings as a sequence of UTF-8 bytes elegantly fulfills this requirement as exemplified in work such as [5], ***current byte-level LMs are not competitive with word-level LMs on large scale datasets. We observed a similar performance gap in our own attempts to train standard byte-level LMs on WebText.***
    - ***A byte-level version of BPE only requires a base vocabulary of size 256. However, directly applying BPE to the byte sequence results in suboptimal merges due to BPE using a greedy frequency based heuristic for building the token vocabulary.*** This results in a suboptimal allocation of limited vocabulary slots and model capacity. To avoid this, we prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.
    - ***Since our approach can assign a probability to any Unicode string, this allows us to evaluate our LMs on any dataset regardless of pre-processing, tokenization, or vocab size.***
- Language modeling
    - Language modeling is usually framed as unsupervised distribution estimation from a set of examples $(x_{1}, x_{2}, ..., x_{n})$ each composed of variable length sequences of symbols $(s_{1}, s_{2}, ..., s_{n})$. Since language has a natural sequential ordering, it is common to factorize the joint probabilities over symbols as the product of conditional probabilities:
    $$p(x) = \prod_{i = 1}^{n}p(s_{i}|s_{1}, s_{2}, ..., s_{n - i})$$
## Architecture
- We use a Transformer [4] based architecture for our LMs. The model largely follows the details of the OpenAI GPT model [2] with a few modifications. ***Layer normalization [6] was moved to the input of each sub-block, similar to a pre-activation residual network [7] and an additional layer normalization was added after the final self-attention block. A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of $\frac{1}{\sqrt{N}}$ where $N$ is the number of residual layers. The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.***
- Table 2. Model variants
    - <img src="https://user-images.githubusercontent.com/67457712/225798715-26a03efa-847d-48a4-b75e-3e7acd0d2d36.png" width="200">
    - The smallest model is equivalent to the original GPT, and the second smallest equivalent to the largest model from BERT [3]. ***Our largest model, which we call GPT-2***, has over an order of magnitude more parameters than GPT.
## Training
### Datasets
- WebText
    - We created a new web scrape which emphasizes document quality. To do this we only scraped web pages which have been curated/filtered by humans.
    - Contains slightly over 8 million documents for a total of 40 GB of text. We removed all Wikipedia documents from WebText since it is a common data source for other datasets and could complicate analysis due to over lapping training data with test evaluation tasks.
## Experiments
- Table 3. Zero-shot transfer evaluation
    - <img src="https://raw.githubusercontent.com/terrifyzhao/terrifyzhao.github.io/master/assets/img/2019-02-19-GPT2.0%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic2.jpg" width="700">
    - ***Large improvements are noticed on small datasets such as Penn Treebank and WikiText-2 which have only 1 to 2 million training tokens. Large improvements are also noticed on datasets created to measure long-term dependencies like LAMBADA and the Children’s Book Test.***
    - Our model is still significantly worse than prior work on the One Billion Word Benchmark. ***This is likely due to a combination of it being both the largest dataset and having some of the most destructive pre-processing - 1BW’s sentence level shuffling removes all long-range structure.***
    - CBT
        - The Children’s Book Test (CBT) was created to examine the performance of LMs on different categories of words: named entities, nouns, verbs, and prepositions. Rather than reporting perplexity as an evaluation metric, CBT reports accuracy on an automatically constructed cloze test where the task is to predict which of 10 possible choices for an omitted word is correct.
    - LAMBADA
        - The LAMBADA dataset tests the ability of systems to model long-range dependencies in text. The task is to predict the final word of sentences which require at least 50 tokens of context for a human to successfully predict.
- Winsograd Schema Challenge
    - The Winograd Schema challenge was constructed to measure the capability of a system to perform commonsense reasoning by measuring its ability to resolve ambiguities in text.
- Reading Comprehension
    - The Conversation Question Answering dataset (CoQA) consists of documents from 7 different domains paired with natural language dialogues between a question asker and a question answerer about the document. CoQA tests reading comprehension capabilities and also the ability of models to answer questions that depend on conversation history (such as "Why?").
- Summarization
    - Table 4. Summarization evaluation
        - <img src="https://user-images.githubusercontent.com/67457712/225798858-f7d16f81-9277-4e85-befd-064af4fb4224.png" width="300">
        - We test GPT-2’s ability to perform summarization on the CNN and Daily Mail dataset. ***To induce summarization behavior we add the text "TL;DR:" after the article and generate 100 tokens with Top-***$k$ ***random sampling with*** $k = 2$ ***which reduces repetition and encourages more abstractive summaries than greedy decoding. (Comment: Beam width를 2로 하는 Beam search를 사용했다는 것으로 이해됩니다.) We use the first 3 generated sentences in these 100 tokens as the summary.***
        - ***While qualitatively the generations resemble summaries, as shown in Table 14, they often focus on recent content from the article or confuse specific details such as how many cars were involved in a crash or whether a logo was on a hat or shirt. On the commonly reported ROUGE 1, 2, L metrics the generated summaries only begin to approach the performance of classic neural baselines and just barely outperforms selecting 3 random sentences from the article.***
        - GPT-2’s performance drops by 6.4 points on the aggregate metric when the task hint is removed which demonstrates the ability to invoke task specific behavior in a language model with natural language. (Comment: R-AVG를 보면 'GPT-2 TL; DR:' 대비 'GPT-2 no hint'의 성능이 하락했습니다.)
- Translation
    - ***In order to help GPT-2 infer that this is the desired task, we condition the language model on a context of example pairs of the format "english sentence = french sentence" and then after a final prompt of "english sentence =" we sample from the model with greedy decoding and use the first generated sentence as the translation.***
    - On the WMT-14 English-French test set, GPT-2 gets 5 BLEU, which is slightly worse than a word-by-word substitution with a bilingual lexicon inferred in previous work on unsupervised word translation.
    - On the WMT-14 French-English test set, 11.5 BLEU. This is much worse than the 33.5 BLEU of the current best unsupervised machine translation approach. ***Performance on this task was surprising to us, since we deliberately removed non-English webpages from WebText as a filtering step.***
## References
- [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Multilingual Language Processing From Bytes](https://arxiv.org/pdf/1512.00103.pdf)
- [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

# Official Repository
- https://github.com/openai/gpt-2