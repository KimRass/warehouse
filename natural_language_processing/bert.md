# Paper Summary
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- **The pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks without substantial task-specific architecture modifications.**
## Related Works
- Feature-based approach
- Fine-tuning approach
  - **The fine-tuning approach, such as the GPT, introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters.**
  - In OpenAI GPT, the authors use a left-to-right architecture, where every token can only at- tend to previous tokens in the self-attention layers of the Transformer. Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.
  - The BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.
## Train
- There are two steps in our framework: pre-training and fine-tuning.
### Pre-training
- During pre-training, the model is trained on unlabeled data over different pre-training tasks.
#### Masked Language Model (MLM)
- BERT alleviates the previously mentioned unidirectionality constraint by using a "masked lan- guage model" (MLM) pre-training objective.
- **The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer.**
- In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random.
- We do not always replace "masked" words with the actual $[MASK]$ token. The training data generator chooses 15% of the token positions at random for prediction. If the $i$-th token is chosen, we replace the $i$-th token with (1) the $[MASK]$ token 80% of the time (2) a random token 10% of the time (3) the unchanged $i$-th token 10% of the time. Then, $T_{i}$ will be used to predict the original token with cross entropy loss.
#### Next Sentence Prediction (NSP)
- In addition to the masked language model, we also use a "next sentence prediction" task that jointly pre-trains text-pair representations.
### Fine-tuning
- **For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters.**
## Architecture
- A distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.
- In this work, we denote the number of layers (i.e., Transformer blocks) as $L$, the hidden size as $H$, and the number of self-attention heads as $A$. We primarily report results on two model sizes: BERT-BASE ($L$ = 12, $H$ = 768, $A$ = 12, Total Parameters = 110M) and BERT-LARGE ($L$ = 24, $H$ = 1024, $A$ = 16, Total Parameters = 340M). BERT BASE was chosen to have the same model size as OpenAI GPT for comparison purposes.
- To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences in one token sequence. Throughout this work, a "sentence" can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.
- We use WordPiece embeddings with a 30,000 token vocabulary. (Comment: Vocab size)
- BERT pre-training and fine-tuning
  - <img src="https://production-media.paperswithcode.com/methods/new_BERT_Overall.jpg" width="500">
- BERT input
- We denote input embedding as $E$, The final hidden vector of the special $[CLS]$ token as $C \in \mathbb{R}^{H}$, and the final hidden vector for the $i$th input token as $T_{i} \in \mathbb{R}^{H}$.
  - <img src="https://lh3.googleusercontent.com/stK9CWIWiSuF_aq75q7_6wUqyqfePKzeLxqVet9IVNqrcyJqqg9hXkhuFXBXXbIjaGY15gSF9Yr7kyjceVXs5HbDMpmkhet49fhbtLsm9-4E4iCYckzGTsYSxOqRaVGNTkkhWykg" width="300">
- The first token of every sequence is always a special classification token ($[CLS]$). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
- Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ($[SEP]$). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B.
- For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings.


- when choosing the sentences A and B for each pre- training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). As we show in Figure 1, C is used for next sentence predic- tion (NSP).5
- For the pre-training corpus we use the BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is criti- cal to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark (Chelba et al., 2013) in order to extract long contiguous sequences.
- For each task, we simply plug in the task- specific inputs and outputs into BERT and fine- tune all the parameters end-to-end. At the in- put, sentence A and sentence B from pre-training are analogous to (1) sentence pairs in paraphras- ing, (2) hypothesis-premise pairs in entailment, (3) question-passage pairs in question answering, and 4) a degenerate text-∅ pair in text classification or sequence tagging. At the output, the token rep- resentations are fed into an output layer for token- level tasks, such as sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as en- tailment or sentiment analysis.
- Compared to pre-training, fine-tuning is rela- tively inexpensive.