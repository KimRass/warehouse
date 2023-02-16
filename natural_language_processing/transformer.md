# Transformer
- References: https://en.wikipedia.org/wiki/Transformer_(machine_learning_model), https://wikidocs.net/31379, https://www.tensorflow.org/text/tutorials/transformer
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
	- TensorFlow implementation
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

# Paper Summary
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.
- In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
- Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
- To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution.
- Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder
- Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.
## Attention
- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
- We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √ dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √ dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of √ 1