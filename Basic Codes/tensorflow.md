# tensorflow
```python
import tensorflow as tf
```
## tf.multiply()

## tf.square()

* 각 arguments를 제곱하여 ndarray 생성

## tf.reduce\_sum()

* axis=0 \| 1

## tf.reduce\_mean()

* axis=0 \| 1

## tf.argmax()

* axis=0 \| 1

## assign

```python
W.assign(W - tf.multiply(lr, dW))
```

## assign_sub

```python
W.assign_sub(tf.multiply(lr, dW))
```

## tf.sign

```python
tf.sign(tf.reduce_sum(self.w * x) + self.b)
```

## tf.exp()

## tf.math.log()

## tf.sigmoid()

## tf.constant()

```python
image = tf.constant([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=tf.float32)
```

## tf.convert\_to\_tensor()

```python
img = tf.convert_to_tensor(img)
```

## tf.Variable()

## tf.zeros()

```python
W = tf.Variable(tf.zeros([2, 1], dtype=tf.float32), name = "weight")
```

## tf.transpose()

## tf.zeros()

```python
tf.zeros([2, 1])
```

## tf.cast()

```python
pred = tf.cast(h > 0.5, dtype=tf.float32)
```

* 조건이 True면 1, False면 0 반환.
* 혹은 단순히 Tensor의 자료형 변환.

## tf.equal()

```python
acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), dtype=tf.float32))
```

## tf.concat()

```python
layer3 = tf.concat([layer1, layer2], axis=1)
```

## tf.reshape()

```python
layer3 = tf.reshape(layer3, shape=[-1, 2])
```

## tf.constant\_initializer()

```
weight_init = tf.constant_initializer(weight)
```

# tf.random

## tf.random.set\_seed()

## tf.random.normal()

```python
x = tf.Variable(tf.random.normal([784, 200], 1, 0.35))
```

## tf.keras

### tf.keras.utils
#### tf.keras.utils.get_file()
```python
base_url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens/datasets/"

movies_path = tf.keras.utils.get_file("movies.csv", os.path.join(base_url, "movies.csv"))
movie_df = pd.read_csv(movies_path)
```
- 인터넷의 파일을 로컬 컴퓨터의 홈 디렉토리 아래 .keras/datasets 디렉토리로 다운로드.
#### tf.keras.utils.to_categorical()
```python
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
```
* one-hot encoding

### @tf.function

* 자동 그래프 생성
* 함수 정의문 직전에 사용

### tf.GradientTape()

```python
with tf.GradientTape() as tape:
    hyp = W * X + b
    loss = tf.reduce_mean(tf.square(hyp - y))

dW, db = tape.gradient(loss, [W, b])
```

### tf.keras.datasets

#### tf.keras.datasets.mnist

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### tf.keras.losses

#### tf.keras.losses.categorical_crossentropy()
```python
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True))
    return loss
```
* 출처 : [https://hwiyong.tistory.com/335](https://hwiyong.tistory.com/335)
* 딥러닝에서 쓰이는 logit은 매우 간단합니다. 모델의 출력값이 문제에 맞게 normalize 되었느냐의 여부입니다. 예를 들어, 10개의 이미지를 분류하는 문제에서는 주로 softmax 함수를 사용하는데요. 이때, 모델이 출력값으로 해당 클래스의 범위에서의 확률을 출력한다면, 이를 logit=False라고 표현할 수 있습니다(이건 저만의 표현인 점을 참고해서 읽어주세요). 반대로 모델의 출력값이 sigmoid 또는 linear를 거쳐서 만들어지게 된다면, logit=True라고 표현할 수 있습니다.
* 결론: 클래스 분류 문제에서 softmax 함수를 거치면 from\_logits = False(default값), 그렇지 않으면 from\_logits = True.
* 텐서플로우에서는 softmax 함수를 거치지 않고, from\_logits = True를 사용하는게 numerical stable하다고 설명하고 있다.
* training=True : tf.keras.layers.Dropout() 적용
- 정답 레이블이 one-hot encoding 형태일 경우 사용.
#### tf.keras.losses.sparse_categorical_crossentropy()
```python
def loss_fn(model, x, y):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=model(x), from_logits=True))
```
- 정답 레이블이 one-hot encoding 형태가 아닐 경우 사용.
### tf.keras.optimizers

#### tf.keras.optimizers.Adam()

```python
optimizer = tf.keras.optimizers.Adam()
```

#### tf.keras.optimizers.SGD()

```python
optimizer = tf.keras.optimizers.SGD(lr=0.01)
```

#### optimizer.apply\_gradients()

```python
optimizer.apply_gradients(zip([dW, db], [W, b]))
```

```python
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

### tf.keras.metrics

```python
tf.keras.metrics.Mean(name = 'test_loss')
```

```python
tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
```

### tf.keras.layers

#### tf.keras.layers.Flatten()

```python
def flatten() :
    return tf.keras.layers.Flatten()
```

#### tf.keras.layers.Dense()

```python
def dense(label_dim, weight_init) :
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init)
```

* units : 출력 값 차원

#### tf.keras.layers.Activation()

* tf\.keras\.activations\.sigmoid \| tf\.keras\.activations\.relu

#### tf.keras.layers.Dropout()

```python
def dropout(rate)
    return tf.keras.layers.Dropout(rate)
```

#### tf.keras.layers.Dropout()

```python
def dropout(rate):
    return tf.keras.layers.Dropout(rate)
```

* rate : dropout을 적용할 perceptron의 비율

#### tf.keras.layers.BatchNormalization()

```python
def batch_norm()
    return tf.keras.layers.BatchNormalization()
```

#### tf,keras.layers.Conv2D()

```python
conv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding="same", data_format="channels_last", kernel_initializer=weight_init)(image)
```

* image : (batch, height, width, number of channels)
* filter : (height, width, number of channels, number of filters)
* convolution : (batch, height, width, number of filters)
* image의 number of channels와 filter의 number of filters의 값은 동일
* fliters : filter 개수.
* kernal\_size : filter의 사이즈(int, tuple, list 가능).
* strides : stride(int, tuple, list 가능).
* padding : "valid" \| "same"

#### tf.keras.layers.MaxPool2D()
```python
pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="valid", data_format="channels_last")(image)
```
#### tf.keras.layers.SimpleRNNCell()
```python
cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
```
#### tf.keras.layers.RNN()
```python
rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)

outputs, states = rnn(x_data)
```
#### tf.keras.layers.SimpleRNN()
```python
rnn = tf.keras.layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)

outputs, states = rnn(x_data)
```
- tf.keras.layers.SimpleRNN() = tf.keras.layers.SimpleRNNCell() + tf.keras.layers.RNN()
#### tk.keras.layers.Embedding()
```python
one_hot = np.eye(len(char2idx))

model.add(tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, trainable=False, mask_zero=True, input_length=max_sequence, embeddings_initializer=tf.keras.initializers.Constant(one_hot)))
```
- input dim : 단어의 개수. 입력되는 정수의 개수
- output_dim : 텍스트의 개수. 출력되는 embedding vector의 크기
- input_length : 입력 sequence의 길이
- mask_zero : 0인 값 무시 여부. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
- embeddings_initializer : Initializer for the embeddings matrix
### tf.keras.initializers

#### tf.keras.initializers.RandomNormal()

#### tf.keras.initializers.glorot_uniform()

#### tf.keras.initializers.he_uniform()
#### tf.keras.initializers.Constant()
* tf.keras.layers.Activation(tf.keras.activations.relu) 사용 시 선택
```python
model.add(tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, trainable=False, mask_zero=True, input_length=max_sequence, embeddings_initializer=tf.keras.initializers.Constant(one_hot)))
```
### tf.keras.Model, tf.keras.Sequential()

```python
class CreateModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(32, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(64, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(256, activation = 'relu')
        self.dense5 = tf.keras.layers.Dense(10, activation = 'softmax')

    def __call__(self, x, training=None, mask=None):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)
```

```python
class CreateModel(tf.keras.Model):
    def __init__(self, label_dim):
        super(CreateModel, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(flatten()) 

        rand_norm = tf.keras.initializers.RandomNormal()
        for i in range(2):
            self.model.add(dense(units=256, rand_norm))
            self.model.add(sigmoid())

        self.model.add(dense(label_dim, rand_norm))

    def call(self, x, training=None, mask=None):
        y = self.model(x)
        return y
```

### tf.keras.Input()

```python
inputs = tf.keras.Input(shape=(28, 28, 1))
```
### tf.keras.preprocessing.sequence.pad_sequences()
```python
x_data = tf.keras.preprocessing.sequence.pad_sequences(sequences=x_data, maxlen=max_sequence, padding="post", truncating="post", value=0)
```
- padding="pre" | "post"
- truncating="pre" | "post"
## tf.nn

### tf.nn.softmax()

```python
h = tf.nn.softmax(tf.matmul(train_X, W) + b)
```

### tf.nn.relu

```python
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)))
```

## tf.data.Dataset

### tf.data.Dataset.from\_tensor\_slices()

```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).batch(batch_size, drop_remainder=True).prefetch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(len(test_x)).batch(len(test_x)).prefetch(len(test_x))
```

* shuffle() : 지정한 개수의 데이터를 무작위로 섞어서 출력.
* batch() : 지정한 개수의 데이터를 묶어서 출력.
* prefetch() : This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.

## tf.train

### tf.train.Checkpoint()

```python
ckpt = tf.train.Checkpoint(cnn=model)
...
ckpt.save(file_prefix=ckpt_prefix)
```
