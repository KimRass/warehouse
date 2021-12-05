# class
## instance attribute(instance variables)
```python
class CLASS:
    def __init__(self):
        self.ATTRIBUTE = VALUE
```
* INSTANCE.ATTRIBUTE로 사용
## class attriubute(class variables)
```python
class CLASS:
    ATTRIBUTE = VALUE
```
* CLASS.ATTRIBUTE로 사용
* 모든 INSTANCE가 ATTRIBUTE 값을 공유.
* 동일한 instance attribute와 class attribute가 있으면 instance attribute -> class attribute 순으로 method를 탐색.
* INSTANCE.ATTRIBUTE로 사용 시 INSTANCE의 namespace에서 ATTRIBUTE를 찾고 없으면 CLASS의 namespace로 이동한 후 다시 ATTRIBUTE를 찾아 그 값을 반환.
## \_\_init\_\_
```python
INSTANCE = CLASS() #instance를 initiate 할 때 실행
```
## \_\_init\_\_, \_\_call\_\_
```python
class CLASS:
    def __init__(self, parameter1, parameter2, ...)
        ...
        
    def __call__(self, parameter3, parameter4, ...)
        ...
        return ...
        
    def FUNCTION(self, parameter5, parameter6, ...)
        ...
        return ...
```
- `INSTANCE = CLASS(parameter1, parameter2, ...)` : \_\_init\_\_문은 instance를 initiate할 때 실행
- `INSTANCE(parameter3, parameter4, ...)` : \_\_call\_\_문은 instance를 call할 때 실행
- `INSTANCE.FUNCTION(parameter5, parameter6, ...)`
- method : class 정의문 안에서 정의된 함수
- method의 첫번째 parameter는 항상 self여야 함
- method의 첫 번째 parameter는 self지만 호출할 때는 아무것도 전달하지 않는 이유는 첫 번째 parameter인 self에 대한 값은 파이썬이 자동으로 넘겨주기 때문입니다.
## \_\_iter\_\_()
```python
def __iter__(self):
	return self
```
## \_\_next\_\_()
## StopIteration
### class variables
- class 정의문 안에서 정의된 variables
### instance variables
- self가 붙어 있는 variables
## override
- 출처 : https://rednooby.tistory.com/55
## super()
- 출처 : https://rednooby.tistory.com/56?category=633023
```python
class Bahdanau(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.W3 = Dense(1)
        
        # key와 value는 같습니다.
    def call(self, query, keys):
        # query.shape : (batch_size, h_size) --> (batch_size, 1, h_size)
        query = tf.expand_dims(query, axis=1)

        # att_scores.shape : (batch_size, max_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.W3 : (batch_size, max_len, units)
        att_scores = self.W3(tf.nn.tanh(self.W1(query) + self.W2(keys)))

        # att_weights.shape : (batch_size, max_len, 1)
        att_weights = tf.nn.softmax(att_scores, axis=1)

        # context_vector.shape : (batch_size, h_size)
        context_vector = att_weights*keys
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, att_weights
```