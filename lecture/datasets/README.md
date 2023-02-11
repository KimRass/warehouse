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
- Reference: https://github.com/bab2min/corpus/tree/master/sentiment
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
- Reference: https://github.com/bab2min/corpus/tree/master/sentiment
## NLP Challenge
## fra-eng
- Reference: https://www.kaggle.com/myksust/fra-eng/activity
```python
data = pd.read_table("./Datasets/fra-eng/fra.txt", usecols=[0, 1], names=["src", "tar"])
# data = pd.read_csv("./Datasets/fra-eng/fra.txt", usecols=[0, 1], names=["src", "tar"], sep="\t")
```
## IMDb
## Annotated Corpus for NER
## Chatbot Data for Korean
- Reference: https://github.com/songys/Chatbot_data
## Natural Language Understanding Benchmark
- Reference: https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines, https://github.com/ajinkyaT/CNN_Intent_Classification
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
- Reference: https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt
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
dataset, metadata = tfds.load(
	"ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
)
dataset_tr, dataset_val, dataset_te = dataset["train"], dataset["validation"], dataset["test"]

tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
	(pt.numpy() for pt, en in dataset_tr), target_vocab_size=2**13
)
tokenizer_tar = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
	(en.numpy() for pt, en in dataset_tr), target_vocab_size=2**13
)

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
    result_pt, result_en = tf.py_function(
		func=encode, inp=[pt, en], Tout=[tf.int64, tf.int64]
	)
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
