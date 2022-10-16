# Attention
- References: https://en.wikipedia.org/wiki/Attention_(machine_learning), https://wikidocs.net/22893
- In neural networks, attention is a technique that mimics cognitive attention. *The effect enhances some parts of the input data while diminishing other parts — the thought being that the network should devote more focus to that small but important part of the data. Learning which part of the data is more important than others depends on the context and is trained by gradient descent.*
- To build a machine that translates English-to-French, one starts with an Encoder-Decoder and grafts an attention unit to it. *In practice, the attention unit consists of 3 fully connected neural network layers that needs to be trained. The 3 layers are called Query, Key, and Value.*
- Self Attention: Query, Key, Value의 출처가 서로 동일한 경우를 말합니다.
- Multi-head Attention: Attention을 Parallel하게 수행한다는 의미입니다.
## Dot-Product Attention (= Luong Attention)
- TensorFlow implementation
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
- TensorFlow implementation
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
- TensorFlow implementation
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