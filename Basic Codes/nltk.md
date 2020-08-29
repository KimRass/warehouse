```python
import nltk
```
# nltk.Text()
```python
text = nltk.Text(total_tokens, name="NMSC")
```
## text.tokens
## text.vocab() : returns frequency distribution
### text.vocab().most_common()
```python
text.vocab().most_common(10)
```
## text.plot()
```python
text.plot(50)
```
# nltk.download()
```python
nltk.download("movie_reviews")
```
```python
nltk.download("punkt")
```
# nltk.corpus
```python
from nltk.corpus import movie_reviews
```
## movie_reviews
### movie_reviews.sents()
```python
sentences = [sent for sent in movie_reviews.sents()]
```