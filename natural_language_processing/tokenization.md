# Subword Tokenization
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

## Byte Pair Encoding (BPE)